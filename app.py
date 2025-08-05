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

headers = {
    "Content-Type": "application/json"  # ✅ JSON formatı için
}

# Token varsa Authorization header'ı ekle
if HF_TOKEN:
    headers["Authorization"] = f"Bearer {HF_TOKEN}"

logger.info(f"✅ Hugging Face API yapılandırıldı: {HF_API_URL}")
logger.info(f"✅ Token durumu: {'Ayarlandı' if HF_TOKEN != 'hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' else 'Fallback'}")

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
            "hf_token_configured": HF_TOKEN != "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
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
        
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        try:
            image_bytes = base64.b64decode(image_data)
            logger.info(f"✅ Görsel decode edildi, boyut: {len(image_bytes)} bytes")
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

@app.route("/", methods=["GET"])
def home():
    """Ana sayfa"""
    try:
        return jsonify({
            "message": "Deepfake Detection API Server",
            "model": "dima806/deepfake_vs_real_image_detection",
            "status": "active",
            "hf_token_configured": HF_TOKEN != "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "endpoints": {
                "health": "/health",
                "model_info": "/model-info", 
                "analyze": "/analyze"
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