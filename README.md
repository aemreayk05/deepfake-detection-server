# Deepfake Detection API Server

Bu server, [Hugging Face'deki deepfake detection modelini](https://huggingface.co/dima806/deepfake_vs_real_image_detection) kullanarak görsellerin gerçek mi yoksa AI tarafından oluşturulmuş mu olduğunu tespit eder.

## 🚀 Özellikler

- **ViT (Vision Transformer) Modeli**: Yüksek doğruluklu deepfake detection
- **99.27% Doğruluk**: Test edilmiş ve kanıtlanmış performans
- **RESTful API**: Kolay entegrasyon
- **CORS Desteği**: Cross-origin istekler için
- **Health Check**: Server durumu kontrolü

## 📊 Model Bilgileri

- **Model**: `dima806/deepfake_vs_real_image_detection`
- **Tip**: ViT (Vision Transformer)
- **Doğruluk**: 99.27%
- **Boyut**: 85.8M parametre
- **Kağıt**: [Kaggle Notebook](https://www.kaggle.com/code/dima806/deepfake-vs-real-faces-detection-vit)

## 🔧 Kurulum

### Yerel Geliştirme

```bash
# Gereksinimleri yükle
pip install -r requirements.txt

# Server'ı başlat
python app.py
```

### Render.com Deploy

1. Bu repository'yi GitHub'a push edin
2. Render.com'da yeni Web Service oluşturun
3. GitHub repository'sini bağlayın
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `gunicorn app:app`

## 📡 API Endpoints

### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-01-04T16:55:00.000Z"
}
```

### 2. Görsel Analizi
```http
POST /analyze
Content-Type: application/json

{
  "image": "base64_encoded_image_data"
}
```

**Response:**
```json
{
  "success": true,
  "prediction": "Real",
  "confidence": 98.45,
  "probabilities": {
    "real": 98.45,
    "fake": 1.55
  },
  "processing_time": 0.234,
  "model_info": {
    "name": "dima806/deepfake_vs_real_image_detection",
    "type": "ViT (Vision Transformer)",
    "description": "Deepfake vs Real Image Detection"
  },
  "timestamp": "2025-01-04T16:55:00.000Z"
}
```

### 3. Model Bilgileri
```http
GET /model-info
```

**Response:**
```json
{
  "model_name": "dima806/deepfake_vs_real_image_detection",
  "model_type": "ViT (Vision Transformer)",
  "description": "Deepfake vs Real Image Detection Model",
  "accuracy": "99.27%",
  "paper": "https://www.kaggle.com/code/dima806/deepfake-vs-real-faces-detection-vit",
  "huggingface": "https://huggingface.co/dima806/deepfake_vs_real_image_detection",
  "loaded": true,
  "timestamp": "2025-01-04T16:55:00.000Z"
}
```

## 🔗 Entegrasyon

### React Native Örneği

```javascript
const analyzeImage = async (base64Image) => {
  try {
    const response = await fetch('https://your-render-url.onrender.com/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: base64Image
      })
    });
    
    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Analiz hatası:', error);
  }
};
```

## ⚠️ Önemli Notlar

1. **Model Boyutu**: Model 85.8M parametre içerir, ilk yükleme zaman alabilir
2. **Memory Kullanımı**: Render.com'da en az 512MB RAM önerilir
3. **Cold Start**: İlk istek model yüklemesi nedeniyle yavaş olabilir
4. **Threshold**: Model %99.27 doğrulukla çalışır, ancak yeni deepfake teknikleri için güncelleme gerekebilir

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Model [Hugging Face](https://huggingface.co/dima806/deepfake_vs_real_image_detection) üzerinden Apache 2.0 lisansı ile kullanılabilir.

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📞 İletişim

- **Geliştirici**: AEA
- **Email**: a.emreaykut@gmail.com
- **GitHub**: [@aemreayk05](https://github.com/aemreayk05) 