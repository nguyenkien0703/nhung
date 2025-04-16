# Hệ Thống Chẩn Đoán Bệnh Cây Trồng

Hệ thống AI chẩn đoán bệnh cây trồng dựa trên hình ảnh và dữ liệu môi trường.

## Tính năng

- Chẩn đoán bệnh cây trồng dựa trên hình ảnh
- Phân tích dữ liệu môi trường (nhiệt độ, độ ẩm, độ ẩm đất, ánh sáng)
- Kết hợp kết quả từ cả hai nguồn dữ liệu để đưa ra chẩn đoán chính xác
- API RESTful để tích hợp với các ứng dụng khác

## Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

2. Chạy ứng dụng:
```bash
python app.py
```

## Sử dụng API

### Chẩn đoán bệnh

Gửi POST request đến endpoint `/predict` với:
- Hình ảnh cây trồng (file)
- Dữ liệu môi trường (form data):
  - temperature: nhiệt độ (°C)
  - humidity: độ ẩm không khí (%)
  - soil_moisture: độ ẩm đất (%)
  - light: cường độ ánh sáng (lux)

Ví dụ sử dụng curl:
```bash
curl -X POST -F "image=@plant_image.jpg" -F "temperature=25" -F "humidity=60" -F "soil_moisture=70" -F "light=1000" http://localhost:5000/predict
```

### Huấn luyện mô hình

Gửi POST request đến endpoint `/train` với dữ liệu huấn luyện dạng JSON:
```json
{
    "image_data": [...],
    "env_data": [...],
    "labels": [...]
}
```

## Kết quả trả về

API sẽ trả về kết quả dạng JSON:
```json
{
    "disease": "Tên bệnh được chẩn đoán",
    "confidence": 0.95,
    "image_prediction": "Dự đoán từ hình ảnh",
    "environment_prediction": "Dự đoán từ dữ liệu môi trường"
}
```

## Các loại bệnh có thể chẩn đoán

- Khỏe mạnh
- Bệnh đốm lá
- Bệnh thối rễ
- Bệnh nấm mốc
- Bệnh virus 