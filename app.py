from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import numpy as np
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
import logging
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global değişkenler
model = None
processor = None

def load_model():
    """Deepfake detection modelini yükle"""
    global model, processor
    try:
        logger.info("🤖 Deepfake detection modeli yükleniyor...")
        
        # Hugging Face modelini yükle
        model_name = "dima806/deepfake_vs_real_image_detection"
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(model_name)
        
        logger.info("✅ Model başarıyla yüklendi!")
        return True
    except Exception as e:
        logger.error(f"❌ Model yükleme hatası: {e}")
        return False

def preprocess_image(image_data):
    """Görseli model için hazırla"""
    try:
        # Base64'ten görseli decode et
        if isinstance(image_data, str):
            if image_data.startswith('data:image'):
                # data:image/jpeg;base64, formatından base64 kısmını al
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
        
        # PIL Image'e çevir
        image = Image.open(io.BytesIO(image_bytes))
        
        # RGB'ye çevir (RGBA ise)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Model için hazırla
        inputs = processor(images=image, return_tensors="pt")
        
        return inputs, image
    except Exception as e:
        logger.error(f"❌ Görsel ön işleme hatası: {e}")
        raise e

def predict_deepfake(inputs):
    """Deepfake detection tahmini yap"""
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Sonuçları al
            fake_prob = probabilities[0][1].item()  # Fake olasılığı
            real_prob = probabilities[0][0].item()  # Real olasılığı
            
            # Tahmin
            prediction = "Fake" if fake_prob > real_prob else "Real"
            confidence = max(fake_prob, real_prob) * 100
            
            return {
                "prediction": prediction,
                "confidence": round(confidence, 2),
                "probabilities": {
                    "real": round(real_prob * 100, 2),
                    "fake": round(fake_prob * 100, 2)
                }
            }
    except Exception as e:
        logger.error(f"❌ Tahmin hatası: {e}")
        raise e

@app.route('/health', methods=['GET'])
def health_check():
    """Sağlık kontrolü"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Görsel analizi yap"""
    try:
        # Request verilerini al
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "Görsel verisi bulunamadı"}), 400
        
        image_data = data['image']
        
        logger.info("🔍 Görsel analizi başlıyor...")
        start_time = datetime.now()
        
        # Görseli ön işle
        inputs, image = preprocess_image(image_data)
        
        # Tahmin yap
        result = predict_deepfake(inputs)
        
        # İşlem süresini hesapla
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Sonucu formatla
        response = {
            "success": True,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "processing_time": round(processing_time, 3),
            "model_info": {
                "name": "dima806/deepfake_vs_real_image_detection",
                "type": "ViT (Vision Transformer)",
                "description": "Deepfake vs Real Image Detection"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Analiz tamamlandı: {result['prediction']} ({result['confidence']}%)")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Analiz hatası: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Model bilgilerini döndür"""
    return jsonify({
        "model_name": "dima806/deepfake_vs_real_image_detection",
        "model_type": "ViT (Vision Transformer)",
        "description": "Deepfake vs Real Image Detection Model",
        "accuracy": "99.27%",
        "paper": "https://www.kaggle.com/code/dima806/deepfake-vs-real-faces-detection-vit",
        "huggingface": "https://huggingface.co/dima806/deepfake_vs_real_image_detection",
        "loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def root():
    """Ana sayfa"""
    return jsonify({
        "service": "Deepfake Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST)",
            "model_info": "/model-info"
        },
        "model": "dima806/deepfake_vs_real_image_detection",
        "status": "running"
    })

if __name__ == '__main__':
    # Modeli yükle
    if load_model():
        logger.info("🚀 Deepfake Detection Server başlatılıyor...")
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        logger.error("❌ Model yüklenemedi, server başlatılamıyor!") 