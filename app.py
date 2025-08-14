from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from PIL import Image
import io
import base64
import time
import traceback
import logging
from dotenv import load_dotenv
import boto3
import ipaddress
import socket
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import uuid

# Environment variables yükle
load_dotenv()

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ✅ GÜVENLİ CORS - Sadece belirli origin'lere izin ver
ALLOWED_ORIGINS = [
    "https://seninapp.com",  # Production domain
    "capacitor://localhost",  # Capacitor
    "ionic://localhost",      # Ionic
    "http://localhost:3000",  # Development
    "http://localhost:8080"   # Development
]

CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# ✅ İstek boyutu sınırı (10MB)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# ✅ Güvenli URL kontrolü için allowlist - GÜNCELLENDİ
ALLOWED_HOSTS = {
    # R2 bucket domain'iniz
    "pub-c83b988f47e44b5490a605d85ec8d0e8.r2.dev",
    # TODO: CDN domain'inizi buraya ekleyin (varsa)
    # "cdn.yourdomain.com"
}

# ✅ Güvenli URL kontrolü fonksiyonu - GÜNCELLENDİ
def host_allowed(host, allowed_hosts):
    """Host'un allowlist'te olup olmadığını kontrol eder"""
    if not host:
        return False
    host = host.lower()
    
    # Tam eşleşme kontrolü - parametreyi kullan
    if host in (h.lower() for h in allowed_hosts):
        return True
    
    # Suffix kontrolü (R2 ve CDN için) - GÜNCELLENDİ
    return (host.endswith(".r2.dev") or  # R2 domain suffix
            host.endswith(".r2.cloudflarestorage.com") or  # Standart R2 suffix
            host.endswith(".your-actual-cdn.com"))  # TODO: Gerçek CDN domain'iniz

def is_url_safe(url, allowed_hosts=None):
    """URL'nin güvenli olup olmadığını kontrol eder - DÜZELTİLDİ"""
    try:
        parsed = urlparse(url)
        
        # Sadece HTTPS'e izin ver
        if parsed.scheme != 'https':
            return False
            
        host = parsed.hostname
        if not host:
            return False
            
        # Allowlist kontrolü - parametreyi geçir
        if allowed_hosts is not None and not host_allowed(host, allowed_hosts):
            return False
            
        # DNS/IP kontrolü - tüm A/AAAA kayıtlarını çöz
        try:
            # Tüm IP'leri çöz (IPv4 + IPv6)
            infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
            
            for family, socktype, proto, canonname, sockaddr in infos:
                ip = sockaddr[0]
                ip_obj = ipaddress.ip_address(ip)
                
                # Özel ağları reddet
                if (ip_obj.is_private or ip_obj.is_loopback or 
                    ip_obj.is_link_local or ip_obj.is_multicast):
                    return False
                    
        except (socket.gaierror, ValueError):
            # DNS çözülemezse güvenli değil
            return False
            
        return True
        
    except Exception:
        return False

# ✅ Görsel küçültme ve yeniden kodlama - DÜZELTİLDİ
def shrink_image(image_bytes, max_side=1024, format='JPEG', quality=85):
    """Görseli küçültür ve yeniden kodlar - MIME ve boyut kontrolü eklendi"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # Format kontrolü
        if img.format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError(f"Desteklenmeyen format: {img.format}")
            
        # Piksel sayısı kontrolü (20MP limit)
        if img.width * img.height > 20_000_000:
            raise ValueError(f"Görsel çok büyük: {img.width}x{img.height} px")
            
        img = img.convert('RGB')
        
        # En-boy oranını koruyarak küçült
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        
        # Yeni boyutları logla
        logger.info(f"Image boyutu: {img.size[0]}x{img.size[1]} px")
        
        # Yeniden kodla
        buf = io.BytesIO()
        img.save(buf, format=format, quality=quality, optimize=True)
        compressed_bytes = buf.getvalue()
        
        # Sıkıştırma oranını logla
        compression_ratio = len(compressed_bytes) / len(image_bytes)
        logger.info(f"Compression ratio: {compression_ratio:.2%}")
        
        return compressed_bytes
        
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise  # Hata durumunda exception fırlat

# ✅ Retry/backoff ile HTTP session - DÜZELTİLDİ
def create_http_session():
    """Retry ve backoff ile HTTP session oluşturur"""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "OPTIONS"]  # Gereksiz metodlar kaldırıldı
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    # HTTP mount kaldırıldı (güvenlik için sadece HTTPS)
    
    return session

# ✅ HTTP session'ı global olarak oluştur
http_session = create_http_session()

# ✅ HUGGING FACE API KONFİGÜRASYONU - DEEPFAKE DETECTION
HF_API_URL = "https://api-inference.huggingface.co/models/dima806/deepfake_vs_real_image_detection"
HF_TOKEN = os.getenv("HF_TOKEN")

# Token kontrolü - Geçici olarak public model kullan
if not HF_TOKEN:
    logger.warning("⚠️ HF_TOKEN bulunamadı, public model kullanılıyor...")
    HF_API_URL = "https://api-inference.huggingface.co/models/dima806/deepfake_vs_real_image_detection"
    HF_TOKEN = ""  # Public model için token gerekmez
else:
    logger.info("✅ HF_TOKEN ayarlandı, private model kullanılıyor...")

# ✅ GEMINI API KONFİGÜRASYONU - PROXY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

if not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY bulunamadı, Gemini servisleri devre dışı...")
    logger.warning("⚠️ Render.com'da GEMINI_API_KEY environment variable'ını ayarlayın!")
else:
    logger.info("✅ GEMINI_API_KEY ayarlandı, Gemini servisleri aktif...")

# ✅ SIGHTENGINE API KONFİGÜRASYONU - PROXY
SIGHTENGINE_API_USER = os.getenv("SIGHTENGINE_API_USER")
SIGHTENGINE_API_SECRET = os.getenv("SIGHTENGINE_API_SECRET")
SIGHTENGINE_BASE_URL = "https://api.sightengine.com/1.0"

if not SIGHTENGINE_API_USER or not SIGHTENGINE_API_SECRET:
    logger.warning("⚠️ SIGHTENGINE API key'leri bulunamadı, Sightengine servisleri devre dışı...")
    logger.warning("⚠️ Render.com'da SIGHTENGINE_API_USER ve SIGHTENGINE_API_SECRET environment variable'larını ayarlayın!")
else:
    logger.info("✅ SIGHTENGINE API key'leri ayarlandı, Sightengine servisleri aktif...")

headers = {
    "Content-Type": "application/json"  # ✅ JSON formatı için
}

# Token varsa Authorization header'ı ekle
if HF_TOKEN:
    headers["Authorization"] = f"Bearer {HF_TOKEN}"

logger.info(f"✅ Hugging Face API yapılandırıldı: {HF_API_URL}")
logger.info(f"✅ Token durumu: {'Ayarlandı' if HF_TOKEN != 'hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' else 'Fallback'}")

# === Cloudflare R2 ayarları (opsiyonel ama önerilir) ===
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_BUCKET = os.getenv("R2_BUCKET")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")

def r2_client():
    if not all([R2_ACCOUNT_ID, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
        return None
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )

# ✅ Rate limiting için basit middleware
from functools import wraps
from collections import defaultdict
import threading

# Basit in-memory rate limiter (production'da Redis kullanın)
request_counts = defaultdict(int)
request_timestamps = defaultdict(list)
rate_limit_lock = threading.Lock()

def rate_limit(max_requests=100, window_seconds=60):
    """Basit rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            
            with rate_limit_lock:
                now = time.time()
                # Eski istekleri temizle
                request_timestamps[client_ip] = [
                    ts for ts in request_timestamps[client_ip] 
                    if now - ts < window_seconds
                ]
                
                # Limit kontrolü
                if len(request_timestamps[client_ip]) >= max_requests:
                    return jsonify({"error": "Rate limit exceeded"}), 429
                    
                # Yeni isteği ekle
                request_timestamps[client_ip].append(now)
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ✅ Dağıtık rate-limit için Redis opsiyonu - DÜZELTİLDİ
def create_redis_rate_limiter():
    """Redis tabanlı rate limiter oluşturur (opsiyonel)"""
    try:
        import redis
        redis_url = os.getenv('REDIS_URL')
        if redis_url:
            r = redis.from_url(redis_url)
            r.ping()  # Bağlantı testi
            logger.info("Redis rate limiter enabled")
            return r
    except ImportError:
        logger.warning("Redis not available, using in-memory limiter")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}, using in-memory limiter")
    
    return None

# Redis client'ı oluştur
redis_client = create_redis_rate_limiter()

def rate_limit_redis(max_requests=100, window_seconds=60):
    """Redis tabanlı rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not redis_client:
                # Redis yoksa in-memory kullan
                return rate_limit(max_requests, window_seconds)(f)(*args, **kwargs)
            
            client_ip = request.remote_addr
            key = f"rate_limit:{client_ip}:{f.__name__}"
            
            try:
                # Sliding window ile rate limit
                current = redis_client.get(key)
                if current and int(current) >= max_requests:
                    return jsonify({"error": "Rate limit exceeded"}), 429
                
                pipe = redis_client.pipeline()
                pipe.incr(key)
                pipe.expire(key, window_seconds)
                pipe.execute()
                
            except Exception as e:
                logger.error(f"Redis rate limit error: {e}")
                # Redis hatası durumunda in-memory'ye fallback
                return rate_limit(max_requests, window_seconds)(f)(*args, **kwargs)
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ✅ Koşullu limiter seçimi - DÜZELTİLDİ
limiter = rate_limit_redis if redis_client else rate_limit

@app.route("/health", methods=["GET"])
def health_check():
    """Sağlık kontrolü endpoint'i"""
    try:
        return jsonify({
            "status": "healthy",
            "model": "dima806/deepfake_vs_real_image_detection",
            "model_loaded": True,
            "timestamp": time.time(),
            "server": "Deepfake Detection API Server",
            "hf_token_configured": HF_TOKEN != "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "gemini_configured": bool(GEMINI_API_KEY)
        })
    except Exception as e:
        logger.error(f"❌ Health check hatası: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/model-info", methods=["GET"])
def model_info():
    """Model bilgilerini döndür"""
    try:
        return jsonify({
            "model_name": "Deepfake vs Real Image Detection",
            "model_type": "ViT (Vision Transformer)",
            "author": "dima806",
            "size": "Medium",
            "description": "Deepfake vs Real image detection using Vision Transformer",
            "accuracy": "99.27%",
            "url": "https://huggingface.co/dima806/deepfake_vs_real_image_detection"
        })
    except Exception as e:
        logger.error(f"❌ Model info hatası: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    # Base64 kabul etmiyoruz; URL tabanlı akışa yönlendir
    try:
        if not request.is_json:
            return jsonify({"error": "JSON bekleniyor"}), 400
        data = request.get_json(force=True) or {}
        if data.get("url"):
            # İstemci yanlış endpoint'i çağırdıysa nazikçe yönlendir
            return jsonify({
                "error": "Lütfen /analyze-url endpoint'ini kullanın",
                "hint": "Body: { url: 'https://...' }"
            }), 400
        return jsonify({
            "error": "Bu endpoint base64 kabul etmiyor. Lütfen önce /upload-url ile yükleyip /analyze-url kullanın."
        }), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload-url", methods=["POST"])
def upload_url():
    """
    Cloudflare R2 için presigned PUT/GET URL üretir.
    Body: { "contentType": "image/jpeg", "key": "uploads/xyz.jpg" (ops.) }
    """
    try:
        client = r2_client()
        if client is None:
            return jsonify({"error": "R2 not configured"}), 500

        data = request.get_json(force=True) or {}
        content_type = data.get("contentType", "image/jpeg")
        key = data.get("key") or f"uploads/{int(time.time()*1000)}.jpg"

        put_url = client.generate_presigned_url(
            "put_object",
            Params={"Bucket": R2_BUCKET, "Key": key, "ContentType": content_type},
            ExpiresIn=600,
        )
        get_url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": R2_BUCKET, "Key": key},
            ExpiresIn=600,
        )
        return jsonify({"key": key, "putUrl": put_url, "getUrl": get_url})
    except Exception as e:
        logger.error(f"❌ upload-url hatası: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ TEK TANIM + KOŞULLU DEKORATÖR - DÜZELTİLDİ
@app.route("/analyze-url", methods=["POST"])
@limiter(max_requests=50, window_seconds=60)
def analyze_url():
    """
    Body: { "url": "https://..." }
    Görseli URL'den indirir, Hugging Face'e binary olarak gönderir.
    """
    start_time = time.time()
    
    try:
        data = request.get_json(force=True) or {}
        url = data.get("url")
        if not url:
            return jsonify({"error": "url required"}), 400

        # ✅ 1) URL güvenlik kontrolü - parametreyi geçir
        if not is_url_safe(url, ALLOWED_HOSTS):
            return jsonify({"error": "Forbidden URL"}), 403

        # ✅ 2) İndir (boyut guard ile)
        try:
            r = http_session.get(url, stream=True, timeout=15)
            r.raise_for_status()
            
            # Content-Length kontrolü
            content_length = int(r.headers.get("Content-Length") or 0)
            if content_length and content_length > 10 * 1024 * 1024:  # 10MB
                return jsonify({"error": "File too large"}), 413
                
            # Stream olarak oku (maksimum 10MB)
            image_bytes = b""
            max_size = 10 * 1024 * 1024
            
            for chunk in r.iter_content(chunk_size=8192):
                image_bytes += chunk
                if len(image_bytes) > max_size:
                    return jsonify({"error": "File too large"}), 413
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"URL download error: {e}")
            return jsonify({"error": f"Failed to download image: {str(e)}"}), 400

        if len(image_bytes) < 1000:
            return jsonify({"error": f"Image too small: {len(image_bytes)} bytes"}), 400

        # ✅ 3) Görseli küçült ve yeniden kodla
        try:
            original_size = len(image_bytes)
            image_bytes = shrink_image(image_bytes, max_side=1024, format='JPEG', quality=85)
            compressed_size = len(image_bytes)
            
            logger.info(f"Image size: {original_size} -> {compressed_size} bytes")
            
        except ValueError as e:
            # MIME veya boyut hatası
            if "Desteklenmeyen format" in str(e):
                return jsonify({"error": "Unsupported media type"}), 415
            elif "çok büyük" in str(e):
                return jsonify({"error": "Image too large"}), 413
            else:
                return jsonify({"error": str(e)}), 400

        # ✅ 4) HF Inference - binary upload + fallback - DÜZELTİLDİ
        try:
            files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
            headers_hf = {}
            if HF_TOKEN:
                headers_hf["Authorization"] = f"Bearer {HF_TOKEN}"
                
            # Tercih 1: multipart upload
            response = http_session.post(
                HF_API_URL, 
                headers=headers_hf, 
                files=files, 
                timeout=60
            )
            
            # Fallback: raw bytes (400/415 durumunda)
            if response.status_code in (400, 415):
                logger.warning("Multipart failed, trying raw bytes...")
                response = http_session.post(
                    HF_API_URL,
                    headers={**headers_hf, "Content-Type": "application/octet-stream"},
                    data=image_bytes,
                    timeout=60
                )
            
            # ✅ 503 durumunda retry + Retry-After kontrolü
            if response.status_code == 503:
                retry_after = response.headers.get('Retry-After')
                wait_time = int(retry_after) if retry_after and retry_after.isdigit() else 2.0
                
                logger.warning(f"HF model loading, retrying in {wait_time}s...")
                time.sleep(wait_time)
                
                # Retry'da da aynı fallback mantığı
                response = http_session.post(
                    HF_API_URL, 
                    headers=headers_hf, 
                    files=files, 
                    timeout=60
                )
                
                if response.status_code in (400, 415):
                    response = http_session.post(
                        HF_API_URL,
                        headers={**headers_hf, "Content-Type": "application/octet-stream"},
                        data=image_bytes,
                        timeout=60
                    )
                
        except Exception as e:
            logger.error(f"HF API error: {e}")
            return jsonify({"error": f"Hugging Face API error: {str(e)}"}), 502

        if response.status_code != 200:
            logger.error(f"HF API: {response.status_code} - {response.text}")
            return jsonify({ 
                "error": f"Hugging Face API hatası: {response.status_code}", 
                "detail": response.text 
            }), 502

        result = response.json()
        if len(result) >= 2:
            result1, result2 = result[0], result[1]
            if 'fake' in result1['label'].lower():
                fake_prob = result1['score'] * 100
                real_prob = result2['score'] * 100
            else:
                real_prob = result1['score'] * 100
                fake_prob = result2['score'] * 100
            prediction = "Sahte" if fake_prob > real_prob else "Gerçek"
            confidence = max(fake_prob, real_prob)
        else:
            prediction, confidence, real_prob, fake_prob = "Bilinmiyor", 0, 0, 0

        # ✅ İşlem süresini hesapla
        elapsed_ms = int((time.time() - start_time) * 1000)

        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": {"real": round(real_prob, 2), "fake": round(fake_prob, 2)},
            "model_used": "dima806/deepfake_vs_real_image_detection",
            "model_info": "ViT-based Deepfake vs Real detection",
            "processing_time_ms": elapsed_ms,
            "image_stats": {
                "original_size_bytes": original_size,
                "compressed_size_bytes": compressed_size,
                "compression_ratio": round(compressed_size / original_size, 3)
            }
        })
        
    except Exception as e:
        logger.error(f"analyze-url error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ✅ TEK TANIM + KOŞULLU DEKORATÖR - DÜZELTİLDİ
@app.route("/api/gemini/text", methods=["POST"])
@limiter(max_requests=30, window_seconds=60)
def gemini_text():
    """Gemini metin oluşturma proxy endpoint'i - optimize edilmiş"""
    start_time = time.time()
    
    try:
        if not GEMINI_API_KEY:
            return jsonify({
                "success": False,
                "error": "Gemini API key yapılandırılmamış"
            }), 500

        logger.info("Gemini text generation request received")
        
        if not request.is_json:
            return jsonify({"error": "JSON formatında veri bekleniyor"}), 400
        
        data = request.json
        prompt = data.get('prompt')
        image_data = data.get('imageData')
        image_url = data.get('imageUrl')
        image_mime = data.get('imageMime', 'image/jpeg')
        
        if not prompt:
            return jsonify({"error": "Prompt gerekli"}), 400

        # Gemini API isteği hazırla
        parts = []
        if image_url:
            # ✅ URL güvenlik kontrolü - parametreyi geçir
            if not is_url_safe(image_url, ALLOWED_HOSTS):
                return jsonify({"error": "Forbidden image URL"}), 403
                
            # URL'den indir ve optimize et
            try:
                r = http_session.get(image_url, timeout=20)
                r.raise_for_status()
                
                # Boyut kontrolü
                if len(r.content) > 5 * 1024 * 1024:  # 5MB
                    return jsonify({"error": "Image too large"}), 413
                    
                # Görseli küçült - DÜZELTİLDİ: MIME sabitlendi
                optimized_bytes = shrink_image(r.content, max_side=1024, format='JPEG', quality=85)
                b64data = base64.b64encode(optimized_bytes).decode('utf-8')
                
                # ✅ DÜZELTİLDİ: MIME her zaman image/jpeg (shrink sonrası)
                parts.append({
                    "inlineData": {
                        "mimeType": "image/jpeg",  # Sabit - shrink sonrası JPEG
                        "data": b64data
                    }
                })
                
            except ValueError as e:
                if "Desteklenmeyen format" in str(e):
                    return jsonify({"error": "Unsupported media type"}), 415
                elif "çok büyük" in str(e):
                    return jsonify({"error": "Image too large"}), 413
                else:
                    return jsonify({"error": f"Image processing error: {str(e)}"}), 400
            except Exception as e:
                return jsonify({"success": False, "error": f"Görsel indirilemedi: {str(e)}"}), 400
        elif image_data:
            # Base64'ten decode et ve optimize et
            try:
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                    
                image_bytes = base64.b64decode(image_data)
                if len(image_bytes) > 5 * 1024 * 1024:  # 5MB
                    return jsonify({"error": "Image too large"}), 413
                    
                # Görseli küçült - DÜZELTİLDİ: MIME sabitlendi
                optimized_bytes = shrink_image(image_bytes, max_side=1024, format='JPEG', quality=85)
                b64data = base64.b64encode(optimized_bytes).decode('utf-8')
                
                # ✅ DÜZELTİLDİ: MIME her zaman image/jpeg (shrink sonrası)
                parts.append({
                    "inlineData": {
                        "mimeType": "image/jpeg",  # Sabit - shrink sonrası JPEG
                        "data": b64data
                    }
                })
                
            except ValueError as e:
                if "Desteklenmeyen format" in str(e):
                    return jsonify({"error": "Unsupported media type"}), 415
                elif "çok büyük" in str(e):
                    return jsonify({"error": "Image too large"}), 413
                else:
                    return jsonify({"error": f"Image processing error: {str(e)}"}), 400
            except Exception as e:
                return jsonify({"success": False, "error": f"Base64 decode hatası: {str(e)}"}), 400
        
        parts.append({"text": prompt})
        
        payload = {
            "contents": [{"parts": parts}]
        }

        # Gemini API'ye istek gönder
        response = http_session.post(
            f"{GEMINI_BASE_URL}/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )

        if not response.ok:
            logger.error(f"Gemini API error: {response.status_code} - {response.text}")
            return jsonify({
                "success": False,
                "error": f"Gemini API hatası: {response.status_code}"
            }), 502

        gemini_data = response.json()
        
        if gemini_data.get('candidates') and gemini_data['candidates'][0].get('content'):
            text = gemini_data['candidates'][0]['content']['parts'][0]['text']
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            logger.info(f"Gemini text generated: {text[:50]}...")
            return jsonify({
                "success": True,
                "text": text.strip(),
                "processing_time_ms": elapsed_ms
            })
        else:
            return jsonify({
                "success": False,
                "error": "Gemini geçersiz yanıt döndü"
            }), 500

    except Exception as e:
        logger.error(f"Gemini text generation error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/gemini/image", methods=["POST"])
@limiter(max_requests=20, window_seconds=60)
def gemini_image():
    """Gemini görsel oluşturma proxy endpoint'i - DÜZELTİLDİ"""
    try:
        if not GEMINI_API_KEY:
            return jsonify({
                "success": False,
                "error": "Gemini API key yapılandırılmamış"
            }), 500

        logger.info("Gemini image generation request received")
        
        if not request.is_json:
            return jsonify({"error": "JSON formatında veri bekleniyor"}), 400
        
        data = request.json
        prompt = data.get('prompt')
        input_image_data = data.get('inputImageBase64')
        input_image_url = data.get('inputImageUrl')
        input_image_mime = data.get('inputImageMime', 'image/jpeg')
        
        if not prompt:
            return jsonify({"error": "Prompt gerekli"}), 400

        # Gemini API isteği hazırla
        parts = []
        if input_image_url:
            # ✅ URL güvenlik kontrolü - parametreyi geçir
            if not is_url_safe(input_image_url, ALLOWED_HOSTS):
                return jsonify({"error": "Forbidden input image URL"}), 403
                
            try:
                r = http_session.get(input_image_url, timeout=20)
                r.raise_for_status()
                
                # Boyut kontrolü
                if len(r.content) > 5 * 1024 * 1024:  # 5MB
                    return jsonify({"error": "Input image too large"}), 413
                    
                # Görseli optimize et - DÜZELTİLDİ: MIME sabitlendi
                optimized_bytes = shrink_image(r.content, max_side=1024, format='JPEG', quality=85)
                b64data = base64.b64encode(optimized_bytes).decode('utf-8')
                
                # ✅ DÜZELTİLDİ: MIME her zaman image/jpeg (shrink sonrası)
                parts.append({
                    "inlineData": {
                        "mimeType": "image/jpeg",  # Sabit - shrink sonrası JPEG
                        "data": b64data
                    }
                })
                
            except ValueError as e:
                if "Desteklenmeyen format" in str(e):
                    return jsonify({"error": "Unsupported input image format"}), 415
                elif "çok büyük" in str(e):
                    return jsonify({"error": "Input image too large"}), 413
                else:
                    return jsonify({"error": f"Input image processing error: {str(e)}"}), 400
            except Exception as e:
                return jsonify({"success": False, "error": f"Input image indirilemedi: {str(e)}"}), 400
                
            parts.append({"text": f"Edit this image: {prompt}"})
        elif input_image_data:
            try:
                # ✅ DÜZELTİLDİ: Debug logging ekle
                logger.info(f"Processing input_image_data, type: {type(input_image_data)}")
                logger.info(f"Input data length: {len(input_image_data) if input_image_data else 0}")
                logger.info(f"Input data starts with: {str(input_image_data)[:50] if input_image_data else 'None'}")
                
                if input_image_data.startswith('data:image'):
                    input_image_data = input_image_data.split(',')[1]
                    logger.info("Removed data URI prefix")
                    
                image_bytes = base64.b64decode(input_image_data)
                logger.info(f"Base64 decoded, size: {len(image_bytes)} bytes")
                
                if len(image_bytes) > 5 * 1024 * 1024:  # 5MB
                    return jsonify({"error": "Input image too large"}), 413
                    
                # Görseli optimize et - DÜZELTİLDİ: MIME sabitlendi
                optimized_bytes = shrink_image(image_bytes, max_side=1024, format='JPEG', quality=85)
                b64data = base64.b64encode(optimized_bytes).decode('utf-8')
                logger.info(f"Image optimized, final size: {len(optimized_bytes)} bytes")
                
                # ✅ DÜZELTİLDİ: MIME her zaman image/jpeg (shrink sonrası)
                parts.append({
                    "inlineData": {
                        "mimeType": "image/jpeg",  # Sabit - shrink sonrası JPEG
                        "data": b64data
                    }
                })
                
            except ValueError as e:
                if "Desteklenmeyen format" in str(e):
                    return jsonify({"error": "Unsupported input image format"}), 415
                elif "çok büyük" in str(e):
                    return jsonify({"error": "Input image too large"}), 413
                else:
                    return jsonify({"error": f"Input image processing error: {str(e)}"}), 400
            except Exception as e:
                return jsonify({"success": False, "error": f"Input image decode hatası: {str(e)}"}), 400
                
            parts.append({"text": f"Edit this image: {prompt}"})
        else:
            parts.append({"text": f"Create an image: {prompt}"})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"]
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
            ]
        }

        # Gemini API'ye istek gönder
        response = http_session.post(
            f"{GEMINI_BASE_URL}/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )

        if not response.ok:
            logger.error(f"Gemini API error: {response.status_code} - {response.text}")
            return jsonify({
                "success": False,
                "error": f"Gemini API hatası: {response.status_code}"
            }), 502

        gemini_data = response.json()
        
        # ✅ BETTER LOGGING EKLE
        logger.info(f"Gemini response status: {response.status_code}")
        logger.info(f"Gemini response keys: {list(gemini_data.keys())}")
        
        # Safety check
        if gemini_data.get('candidates') and gemini_data['candidates'][0]:
            candidate = gemini_data['candidates'][0]
            
            if candidate.get('finishReason') == 'IMAGE_SAFETY':
                return jsonify({
                    "success": False,
                    "error": "Görsel güvenlik politikaları nedeniyle oluşturulamadı",
                    "errorType": "SAFETY"
                }), 400
            
            if candidate.get('finishReason') and candidate['finishReason'] != 'STOP':
                return jsonify({
                    "success": False,
                    "error": f"Görsel oluşturma tamamlanamadı: {candidate['finishReason']}",
                    "errorType": "FINISH_REASON"
                }), 400
            
            # Image data ara
            if candidate.get('content'):
                logger.info(f"Candidate content parts count: {len(candidate['content']['parts'])}")
                for i, part in enumerate(candidate['content']['parts']):
                    logger.info(f"Part {i} keys: {list(part.keys())}")
                    if 'inlineData' in part:
                        logger.info(f"Part {i} inlineData keys: {list(part['inlineData'].keys())}")
                        logger.info(f"Part {i} data length: {len(part['inlineData'].get('data', ''))}")
                    # ✅ DÜZELTİLDİ: inlineData ile uyumlu
                    if part.get('inlineData') and part['inlineData'].get('data'):
                        inline_data_b64 = part['inlineData']['data']
                        mime_type = part['inlineData'].get('mimeType', 'image/png')
                        
                        # ✅ BASE64 VALIDATION EKLE
                        if not inline_data_b64 or inline_data_b64 == 'undefined' or len(inline_data_b64) < 100:
                            logger.error(f"Invalid base64 data: length={len(inline_data_b64) if inline_data_b64 else 0}, data={inline_data_b64[:50] if inline_data_b64 else 'None'}")
                            continue  # Sonraki part'a geç
                        
                        # Base64 decode test et
                        try:
                            test_bytes = base64.b64decode(inline_data_b64)
                            if len(test_bytes) == 0:
                                logger.error("Empty image data after base64 decode")
                                continue
                            logger.info(f"✅ Gemini image generated - size: {len(test_bytes)} bytes")
                        except Exception as e:
                            logger.error(f"Invalid base64 format: {e}")
                            continue

                        # Eğer R2 yapılandırılmışsa görüntüyü yükleyip URL döndür
                        try:
                            client = r2_client()
                            if client is not None:
                                file_ext = 'png' if 'png' in mime_type else 'jpg'
                                object_key = f"generated/{int(time.time()*1000)}_{uuid.uuid4().hex}.{file_ext}"
                                image_bytes = base64.b64decode(inline_data_b64)
                                client.put_object(
                                    Bucket=R2_BUCKET,
                                    Key=object_key,
                                    Body=image_bytes,
                                    ContentType=mime_type,
                                )
                                # Erişim için presigned GET URL üret
                                get_url = client.generate_presigned_url(
                                    "get_object",
                                    Params={"Bucket": R2_BUCKET, "Key": object_key},
                                    ExpiresIn=60 * 60,  # 1 saat
                                )
                                return jsonify({
                                    "success": True,
                                    "url": get_url,
                                    "mimeType": mime_type
                                })
                        except Exception as upload_error:
                            logger.error(f"R2 upload error: {upload_error}")

                        # R2 yoksa geriye base64 döndür (geri uyum)
                        return jsonify({
                            "success": True,
                            "imageData": inline_data_b64,
                            "mimeType": mime_type
                        })
        
        return jsonify({
            "success": False,
            "error": "Görsel oluşturulamadı"
        }), 500

    except Exception as e:
        logger.error(f"Gemini image generation error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ✅ SIGHTENGINE API PROXY ENDPOINT'İ - DÜZELTİLDİ
@app.route("/api/sightengine/check", methods=["POST"])
@limiter(max_requests=40, window_seconds=60)
def sightengine_check():
    """Sightengine AI detection proxy endpoint'i"""
    try:
        if not SIGHTENGINE_API_USER or not SIGHTENGINE_API_SECRET:
            return jsonify({
                "success": False,
                "error": "Sightengine API key'leri yapılandırılmamış"
            }), 500

        logger.info("Sightengine AI detection request received")
        
        if not request.is_json:
            return jsonify({"error": "JSON formatında veri bekleniyor"}), 400
        
        data = request.json
        image_url = data.get('url')
        image_data = data.get('image')
        
        if not image_url and not image_data:
            return jsonify({"error": "URL veya image data gerekli"}), 400

        # Sightengine API isteği hazırla
        payload = {
            'models': 'genai',
            'api_user': SIGHTENGINE_API_USER,
            'api_secret': SIGHTENGINE_API_SECRET
        }

        files = None

        if image_url:
            # ✅ URL güvenlik kontrolü - parametreyi geçir
            if not is_url_safe(image_url, ALLOWED_HOSTS):
                return jsonify({"error": "Forbidden image URL"}), 403
                
            payload['url'] = image_url
            logger.info("Checking via URL")
        elif image_data:
            # Base64 string geldiyse dosyaya dönüştürüp 'media' olarak gönder
            logger.info("Base64 image received, converting to file upload...")
            try:
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                
                # Boyut kontrolü
                if len(image_bytes) > 10 * 1024 * 1024:  # 10MB
                    return jsonify({"error": "Image too large"}), 413
                    
                files = {
                    'media': ('image.jpg', image_bytes, 'image/jpeg')
                }
                logger.info(f"Base64 decode successful, size: {len(image_bytes)} bytes")
            except Exception as e:
                logger.error(f"Base64 decode error: {e}")
                return jsonify({
                    "success": False,
                    "error": f"Base64 decode hatası: {str(e)}"
                }), 400

        # Sightengine API'ye istek gönder
        try:
            if files:
                response = http_session.post(
                    f"{SIGHTENGINE_BASE_URL}/check.json",
                    data=payload,
                    files=files,
                    timeout=30
                )
            else:
                response = http_session.post(
                    f"{SIGHTENGINE_BASE_URL}/check.json",
                    data=payload,
                    timeout=30
                )
        except Exception as e:
            logger.error(f"Sightengine request failed: {e}")
            return jsonify({
                "success": False,
                "error": f"Sightengine isteği gönderilemedi: {str(e)}"
            }), 502

        if not response.ok:
            logger.error(f"Sightengine API error: {response.status_code} - {response.text}")
            return jsonify({
                "success": False,
                "error": f"Sightengine API hatası: {response.status_code}",
                "detail": response.text
            }), 502

        sightengine_data = response.json()
        logger.info(f"Sightengine response received: {sightengine_data}")
        
        return jsonify({
            "success": True,
            "data": sightengine_data
        })

    except Exception as e:
        logger.error(f"Sightengine AI detection error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ... existing code ...

@app.route("/", methods=["GET"])
def home():
    """Ana sayfa"""
    try:
        return jsonify({
            "message": "Deepfake Detection API Server",
            "model": "dima806/deepfake_vs_real_image_detection",
            "status": "active",
            "hf_token_configured": HF_TOKEN != "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "gemini_configured": bool(GEMINI_API_KEY),
            "sightengine_configured": bool(SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET),
            "endpoints": {
                "health": "/health",
                "model_info": "/model-info", 
                "analyze": "/analyze",
                "analyze_url": "/analyze-url",
                "upload_url": "/upload-url",
                "gemini_text": "/api/gemini/text",
                "gemini_image": "/api/gemini/image",
                "sightengine_check": "/api/sightengine/check"
            },
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Ana sayfa hatası: {e}")
        return jsonify({"error": str(e)}), 500

# Error handler'lar
@app.errorhandler(404)
def not_found(error):
    logger.error(f"404 hatası: {error}")
    return jsonify({"error": "Endpoint bulunamadı"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 hatası: {error}")
    return jsonify({"error": "Sunucu hatası"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Genel hata: {e}")
    logger.error(f"Hata detayı: {traceback.format_exc()}")
    return jsonify({"error": "Beklenmeyen hata"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Deepfake Detection API Server starting - Port: {port}")
    logger.info(f"Model: dima806/deepfake_vs_real_image_detection")
    logger.info(f"HF_TOKEN status: {'Configured' if HF_TOKEN != 'hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' else 'NOT CONFIGURED!'}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
