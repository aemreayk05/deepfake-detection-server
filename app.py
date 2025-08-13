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

# Environment variables yükle
load_dotenv()

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

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
    try:
        logger.info("🔍 Deepfake analizi isteği alındı")
        
        # Request kontrolü
        if not request.is_json:
            logger.error("❌ JSON formatında veri bekleniyor")
            return jsonify({"error": "JSON formatında veri bekleniyor"}), 400
        
        if 'image' not in request.json:
            logger.error("❌ 'image' field'ı bulunamadı")
            return jsonify({"error": "Görsel bulunamadı - 'image' field'ı gerekli"}), 400

        image_data = request.json['image']
        logger.info(f"📸 Görsel verisi alındı, uzunluk: {len(image_data)}")
        logger.info(f"📸 Görsel verisi (ilk 100 karakter): {image_data[:100]}")
        
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
            logger.info(f"📸 Base64 kısmı alındı, uzunluk: {len(image_data)}")
        
        try:
            image_bytes = base64.b64decode(image_data)
            logger.info(f"✅ Görsel decode edildi, boyut: {len(image_bytes)} bytes")
            
            # Görsel boyutu kontrolü
            if len(image_bytes) < 100:
                logger.error(f"❌ Görsel çok küçük: {len(image_bytes)} bytes")
                return jsonify({"error": f"Görsel çok küçük: {len(image_bytes)} bytes - geçerli bir görsel değil"}), 400
                
        except Exception as e:
            logger.error(f"❌ Base64 decode hatası: {e}")
            return jsonify({"error": f"Görsel decode hatası: {str(e)}"}), 400

        # ✅ HUGGING FACE API'YE GÖNDER
        # Hugging Face API için doğru format: data:image/jpeg;base64,{base64_data}
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # API'nin beklediği format
        payload = {
            "inputs": f"data:image/jpeg;base64,{base64_image}"
        }
        
        logger.info(f"📤 Hugging Face Deepfake API'ye gönderiliyor...")
        logger.info(f"📤 URL: {HF_API_URL}")
        logger.info(f"🔑 Token: {HF_TOKEN[:10]}..." if len(HF_TOKEN) > 10 else "🔑 Token: Geçersiz")
        
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)

        logger.info(f"📥 Response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"❌ Hugging Face API hatası: {response.status_code}")
            logger.error(f"❌ Response text: {response.text}")
            return jsonify({
                "error": f"Hugging Face API hatası: {response.status_code}",
                "detail": response.text,
                "token_configured": HF_TOKEN != "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            }), 500

        result = response.json()
        logger.info(f"✅ Hugging Face API response: {result}")
        
        if len(result) >= 2:
            result1 = result[0]
            result2 = result[1]

            logger.info(f"📊 Result 1: {result1}")
            logger.info(f"📊 Result 2: {result2}")

            # ✅ DEEPFAKE DETECTION MANTIĞI
            # fake = sahte, real = gerçek
            if 'fake' in result1['label'].lower():
                fake_prob = result1['score'] * 100
                real_prob = result2['score'] * 100
                logger.info(f"🎭 Fake (Sahte) skoru: {fake_prob}%")
                logger.info(f"✅ Real (Gerçek) skoru: {real_prob}%")
            else:
                real_prob = result1['score'] * 100
                fake_prob = result2['score'] * 100
                logger.info(f"✅ Real (Gerçek) skoru: {real_prob}%")
                logger.info(f"🎭 Fake (Sahte) skoru: {fake_prob}%")

            # ✅ DEEPFAKE TAHMİNİ
            prediction = "Sahte" if fake_prob > real_prob else "Gerçek"
            confidence = max(fake_prob, real_prob)
            
            logger.info(f"🎯 Deepfake Tahmini: {prediction} (Güven: {confidence}%)")
            logger.info(f"🎭 Sahte olasılığı: {fake_prob}%")
            logger.info(f"✅ Gerçek olasılığı: {real_prob}%")
            
        else:
            logger.warning(f"⚠️ Beklenmeyen response format: {result}")
            prediction = "Bilinmiyor"
            confidence = 0
            real_prob = 0
            fake_prob = 0

        final_result = {
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": {
                "real": round(real_prob, 2),
                "fake": round(fake_prob, 2)
            },
            "model_used": "dima806/deepfake_vs_real_image_detection",
            "model_info": "ViT-based Deepfake vs Real detection",
            "processing_time": time.time(),
            "raw_scores": {
                "fake": fake_prob,
                "real": real_prob
            }
        }
        
        logger.info(f"✅ Deepfake analizi tamamlandı: {prediction} ({confidence}%)")
        return jsonify(final_result)

    except Exception as e:
        logger.error(f"❌ Deepfake analizi hatası: {e}")
        logger.error(f"❌ Hata detayı: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": "Sunucu hatası", 
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

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

@app.route("/analyze-url", methods=["POST"])
def analyze_url():
    """
    Body: { "url": "https://..." }
    Görseli URL'den indirir, Hugging Face'e aynı /analyze mantığıyla gönderir.
    """
    try:
        data = request.get_json(force=True) or {}
        url = data.get("url")
        if not url:
            return jsonify({"error": "url required"}), 400

        r = requests.get(url, stream=True, timeout=20)
        r.raise_for_status()
        image_bytes = r.content
        if len(image_bytes) < 100:
            return jsonify({"error": f"Görsel çok küçük: {len(image_bytes)} bytes"}), 400

        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        payload = { "inputs": f"data:image/jpeg;base64,{base64_image}" }

        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            logger.error(f"❌ HF API: {response.status_code} - {response.text}")
            return jsonify({ "error": f"Hugging Face API hatası: {response.status_code}", "detail": response.text }), 500

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

        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": {"real": round(real_prob, 2), "fake": round(fake_prob, 2)},
            "model_used": "dima806/deepfake_vs_real_image_detection",
            "model_info": "ViT-based Deepfake vs Real detection",
            "processing_time": time.time()
        })
    except Exception as e:
        logger.error(f"❌ analyze-url hatası: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ✅ GEMINI API PROXY ENDPOINT'LERİ
@app.route("/api/gemini/text", methods=["POST"])
def gemini_text():
    """Gemini metin oluşturma proxy endpoint'i"""
    try:
        if not GEMINI_API_KEY:
            return jsonify({
                "success": False,
                "error": "Gemini API key yapılandırılmamış"
            }), 500

        logger.info("📝 Gemini metin oluşturma isteği alındı")
        
        if not request.is_json:
            return jsonify({"error": "JSON formatında veri bekleniyor"}), 400
        
        data = request.json
        prompt = data.get('prompt')
        image_data = data.get('imageData')
        image_mime = data.get('imageMime', 'image/jpeg')
        
        if not prompt:
            return jsonify({"error": "Prompt gerekli"}), 400

        # Gemini API isteği hazırla
        parts = []
        if image_data:
            parts.append({
                "inline_data": {
                    "mime_type": image_mime,
                    "data": image_data
                }
            })
        
        parts.append({"text": prompt})
        
        payload = {
            "contents": [{"parts": parts}]
        }

        # Gemini API'ye istek gönder
        response = requests.post(
            f"{GEMINI_BASE_URL}/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )

        if not response.ok:
            logger.error(f"❌ Gemini API hatası: {response.status_code} - {response.text}")
            return jsonify({
                "success": False,
                "error": f"Gemini API hatası: {response.status_code}"
            }), 500

        gemini_data = response.json()
        
        if gemini_data.get('candidates') and gemini_data['candidates'][0].get('content'):
            text = gemini_data['candidates'][0]['content']['parts'][0]['text']
            logger.info(f"✅ Gemini metin oluşturuldu: {text[:50]}...")
            return jsonify({
                "success": True,
                "text": text.strip()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Gemini geçersiz yanıt döndü"
            }), 500

    except Exception as e:
        logger.error(f"❌ Gemini metin oluşturma hatası: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/gemini/image", methods=["POST"])
def gemini_image():
    """Gemini görsel oluşturma proxy endpoint'i"""
    try:
        if not GEMINI_API_KEY:
            return jsonify({
                "success": False,
                "error": "Gemini API key yapılandırılmamış"
            }), 500

        logger.info("🎨 Gemini görsel oluşturma isteği alındı")
        
        if not request.is_json:
            return jsonify({"error": "JSON formatında veri bekleniyor"}), 400
        
        data = request.json
        prompt = data.get('prompt')
        input_image_data = data.get('inputImageBase64')
        input_image_mime = data.get('inputImageMime', 'image/jpeg')
        
        if not prompt:
            return jsonify({"error": "Prompt gerekli"}), 400

        # Gemini API isteği hazırla
        parts = []
        if input_image_data:
            parts.append({
                "inline_data": {
                    "mime_type": input_image_mime,
                    "data": input_image_data
                }
            })
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
        response = requests.post(
            f"{GEMINI_BASE_URL}/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )

        if not response.ok:
            logger.error(f"❌ Gemini API hatası: {response.status_code} - {response.text}")
            return jsonify({
                "success": False,
                "error": f"Gemini API hatası: {response.status_code}"
            }), 500

        gemini_data = response.json()
        
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
                for part in candidate['content']['parts']:
                    if part.get('inlineData') and part['inlineData'].get('data'):
                        logger.info("✅ Gemini görsel oluşturuldu")
                        inline_data_b64 = part['inlineData']['data']
                        mime_type = part['inlineData'].get('mimeType', 'image/png')

                        # Eğer R2 yapılandırılmışsa görüntüyü yükleyip URL döndür
                        try:
                            client = r2_client()
                            if client is not None:
                                import uuid, base64 as _b64
                                file_ext = 'png' if 'png' in mime_type else 'jpg'
                                object_key = f"generated/{int(time.time()*1000)}_{uuid.uuid4().hex}.{file_ext}"
                                image_bytes = _b64.b64decode(inline_data_b64)
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
                            logger.error(f"❌ R2 yükleme hatası: {upload_error}")

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
        logger.error(f"❌ Gemini görsel oluşturma hatası: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ✅ SIGHTENGINE API PROXY ENDPOINT'İ
@app.route("/api/sightengine/check", methods=["POST"])
def sightengine_check():
    """Sightengine AI detection proxy endpoint'i"""
    try:
        if not SIGHTENGINE_API_USER or not SIGHTENGINE_API_SECRET:
            return jsonify({
                "success": False,
                "error": "Sightengine API key'leri yapılandırılmamış"
            }), 500

        logger.info("🔍 Sightengine AI detection isteği alındı")
        
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
            payload['url'] = image_url
            logger.info("🌐 URL üzerinden kontrol edilecek")
        elif image_data:
            # Base64 string geldiyse dosyaya dönüştürüp 'media' olarak gönder
            logger.info("🖼️ Base64 image alındı, dosya upload'a dönüştürülüyor...")
            try:
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                files = {
                    'media': ('image.jpg', image_bytes, 'image/jpeg')
                }
                logger.info(f"✅ Base64 decode başarılı, boyut: {len(image_bytes)} bytes")
            except Exception as e:
                logger.error(f"❌ Base64 decode hatası: {e}")
                return jsonify({
                    "success": False,
                    "error": f"Base64 decode hatası: {str(e)}"
                }), 400

        # Sightengine API'ye istek gönder
        try:
            if files:
                response = requests.post(
                    f"{SIGHTENGINE_BASE_URL}/check.json",
                    data=payload,
                    files=files,
                    timeout=30
                )
            else:
                response = requests.post(
                    f"{SIGHTENGINE_BASE_URL}/check.json",
                    data=payload,
                    timeout=30
                )
        except Exception as e:
            logger.error(f"❌ Sightengine isteği gönderilemedi: {e}")
            return jsonify({
                "success": False,
                "error": f"Sightengine isteği gönderilemedi: {str(e)}"
            }), 500

        if not response.ok:
            logger.error(f"❌ Sightengine API hatası: {response.status_code} - {response.text}")
            return jsonify({
                "success": False,
                "error": f"Sightengine API hatası: {response.status_code}",
                "detail": response.text
            }), 500

        sightengine_data = response.json()
        logger.info(f"✅ Sightengine yanıtı alındı: {sightengine_data}")
        
        return jsonify({
            "success": True,
            "data": sightengine_data
        })

    except Exception as e:
        logger.error(f"❌ Sightengine AI detection hatası: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

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
                "gemini_text": "/api/gemini/text",
                "gemini_image": "/api/gemini/image",
                "sightengine_check": "/api/sightengine/check"
            },
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"❌ Ana sayfa hatası: {e}")
        return jsonify({"error": str(e)}), 500

# Error handler'lar
@app.errorhandler(404)
def not_found(error):
    logger.error(f"❌ 404 hatası: {error}")
    return jsonify({"error": "Endpoint bulunamadı"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"❌ 500 hatası: {error}")
    return jsonify({"error": "Sunucu hatası"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"❌ Genel hata: {e}")
    logger.error(f"❌ Hata detayı: {traceback.format_exc()}")
    return jsonify({"error": "Beklenmeyen hata"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Deepfake Detection API Server başlatılıyor - Port: {port}")
    logger.info(f"🤖 Model: dima806/deepfake_vs_real_image_detection")
    logger.info(f"🔑 HF_TOKEN durumu: {'Ayarlandı' if HF_TOKEN != 'hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' else 'AYARLANMADI!'}")
    
    app.run(host='0.0.0.0', port=port, debug=False) 
