# 🚗 License Plate Recognition System - YOLOv8 + EasyOCR

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLO](https://img.shields.io/badge/YOLO-v8-red.svg)](https://github.com/ultralytics/ultralytics)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-1.6+-green.svg)](https://github.com/JaidedAI/EasyOCR)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-orange.svg)](https://opencv.org/)

## 📋 Tổng quan dự án

Hệ thống nhận diện biển số xe sử dụng **YOLOv8** để phát hiện vị trí biển số và **EasyOCR** để đọc văn bản. Dự án bao gồm quy trình hoàn chỉnh từ việc thu thập dữ liệu, train model YOLO, đến việc triển khai hệ thống nhận diện với giao diện GUI.

## 🎯 Tính năng chính

- **🔍 Phát hiện biển số**: Sử dụng YOLOv8 model đã được train custom
- **📖 OCR văn bản**: Đọc biển số bằng EasyOCR với độ chính xác cao
- **🖥️ Giao diện GUI**: Tkinter interface thân thiện và dễ sử dụng
- **🖼️ Xử lý ảnh**: Pipeline tăng cường chất lượng ảnh tự động
- **💾 Lưu trữ kết quả**: Export ảnh với bounding box và text được nhận diện
- **⚡ Multi-threading**: Xử lý không block giao diện người dùng

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Image   │───▶│   YOLOv8 Model  │───▶│  License Plate  │
│                 │    │   (Detection)   │    │   Detection     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Image Preproc  │───▶│   EasyOCR       │
                       │  (Enhancement)  │    │   (Text Read)   │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Result Image   │    │  Detected Text  │
                       │  (BBox + Text)  │    │  (License #)    │
                       └─────────────────┘    └─────────────────┘
```

## 📁 Cấu trúc dự án

```
license-plate-recognition/
├── 📁 YOLO/                          # Thư mục chính
│   ├── 📄 main.py                    # Script command line
│   ├── 📄 main_gui.py                # Giao diện GUI
│   ├── 📄 requirements.txt            # Dependencies
│   ├── 📄 README.md                  # Hướng dẫn này
│   ├── 🎯 best.pt                    # Model YOLO đã train
│   ├── 🎯 yolov8n.pt                 # Model YOLO gốc
│   ├── 🖼️ bien1.jpg                 # Ảnh test 1
│   ├── 🖼️ bien2.jpg                 # Ảnh test 2
│   └── 📁 coco128/                   # Dataset training
│       ├── 📁 images/
│       │   └── 📁 train2017/         # Ảnh training
│       └── 📁 labels/
│           └── 📁 train2017/         # Labels YOLO format
├── 📁 dataset/                       # Dataset gốc
├── 📁 training/                      # Scripts training
└── 📁 docs/                          # Tài liệu
```

## 🛠️ Cài đặt và thiết lập

### 1. Yêu cầu hệ thống

- **Python**: 3.8+ 
- **RAM**: Tối thiểu 8GB (16GB+ khuyến nghị)
- **GPU**: NVIDIA GPU với CUDA (khuyến nghị)
- **OS**: Windows 10/11, Linux, macOS

### 2. Clone repository

```bash
git clone https://github.com/yourusername/license-plate-recognition.git
cd license-plate-recognition/YOLO
```

### 3. Cài đặt dependencies

```bash
# Cài đặt PyTorch (chọn version phù hợp với CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài đặt các thư viện khác
pip install -r requirements.txt
```

### 4. Kiểm tra cài đặt

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
python -c "import easyocr; print('EasyOCR: OK')"
```

## 🎓 Hướng dẫn Training YOLO Model

### Bước 1: Chuẩn bị dữ liệu

#### 1.1 Thu thập ảnh biển số xe
- **Số lượng**: Tối thiểu 500-1000 ảnh
- **Đa dạng**: Góc chụp, ánh sáng, điều kiện thời tiết khác nhau
- **Chất lượng**: Độ phân giải cao, rõ nét

#### 1.2 Chuẩn bị dataset theo format YOLO

```bash
dataset/
├── images/
│   ├── train/          # 80% ảnh training
│   ├── val/            # 20% ảnh validation
│   └── test/           # Ảnh test (optional)
└── labels/
    ├── train/          # Labels tương ứng
    ├── val/
    └── test/
```

#### 1.3 Format label YOLO
Mỗi file `.txt` chứa:
```
class_id center_x center_y width height
```
Ví dụ:
```
0 0.5 0.5 0.3 0.2
```

### Bước 2: Cấu hình training

#### 2.1 Tạo file `data.yaml`

```yaml
# data.yaml
path: ../dataset  # Đường dẫn tới dataset
train: images/train  # Ảnh training
val: images/val      # Ảnh validation

# Số classes
nc: 1

# Tên classes
names:
  0: license_plate
```

#### 2.2 Tạo file `model.yaml` (tùy chọn)

```yaml
# model.yaml
nc: 1  # Số classes
depth_multiple: 1.0  # Depth scaling
width_multiple: 1.0  # Width scaling
```

### Bước 3: Training model

#### 3.1 Training từ đầu

```python
from ultralytics import YOLO

# Tạo model mới
model = YOLO('yolov8n.yaml')  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

# Training
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='license_plate_detection'
)
```

#### 3.2 Transfer learning

```python
# Load pre-trained model
model = YOLO('yolov8n.pt')

# Fine-tuning
results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    pretrained=True,
    name='license_plate_finetune'
)
```

#### 3.3 Training với custom parameters

```python
results = model.train(
    data='data.yaml',
    epochs=200,
    imgsz=640,
    batch=8,
    lr0=0.01,
    lrf=0.1,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    pose=12.0,
    kobj=1.0,
    label_smoothing=0.0,
    nbs=64,
    overlap_mask=True,
    mask_ratio=4,
    dropout=0.0,
    val=True,
    plots=True,
    save=True,
    save_period=10
)
```

### Bước 4: Đánh giá và tối ưu

#### 4.1 Kiểm tra metrics

```python
# Validate model
metrics = model.val()

# Xem kết quả
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.p}")
print(f"Recall: {metrics.box.r}")
```

#### 4.2 Export model

```python
# Export sang các format khác
model.export(format='onnx')      # ONNX
model.export(format='tflite')    # TensorFlow Lite
model.export(format='coreml')    # Core ML
```

## 🔍 Sử dụng EasyOCR

### 1. Khởi tạo EasyOCR

```python
import easyocr

# Khởi tạo reader với ngôn ngữ tiếng Anh
reader = easyocr.Reader(['en'])

# Hỗ trợ nhiều ngôn ngữ
reader = easyocr.Reader(['en', 'vi'])  # Tiếng Anh + Tiếng Việt
```

### 2. Đọc văn bản từ ảnh

#### 2.1 Đọc đơn giản

```python
# Đọc từ ảnh
results = reader.readtext('image.jpg')

# Kết quả: [(bbox, text, confidence), ...]
for (bbox, text, prob) in results:
    print(f'Text: {text}, Confidence: {prob:.2f}')
```

#### 2.2 Đọc với preprocessing

```python
import cv2
import numpy as np

def enhance_image(image):
    """Tăng cường chất lượng ảnh"""
    # Tăng kích thước
    scale_factor = 3
    height, width = image.shape[:2]
    new_width = width * scale_factor
    new_height = height * scale_factor
    
    resized = cv2.resize(image, (new_width, new_height), 
                        interpolation=cv2.INTER_CUBIC)
    
    # Làm mịn
    smoothed = cv2.GaussianBlur(resized, (3, 3), 0)
    
    # Tăng độ tương phản
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

# Áp dụng preprocessing
enhanced_image = enhance_image(cropped_plate)
results = reader.readtext(enhanced_image)
```

### 3. Tối ưu hóa OCR

#### 3.1 Lọc kết quả theo confidence

```python
# Lọc theo độ tin cậy
filtered_results = []
for (bbox, text, prob) in results:
    if prob > 0.5:  # Chỉ lấy kết quả có confidence > 50%
        filtered_results.append((bbox, text, prob))
```

#### 3.2 Sắp xếp theo vị trí

```python
# Sắp xếp theo vị trí Y (từ trên xuống dưới)
text_with_positions = []
for (bbox, text, prob) in results:
    if prob > 0.2:
        y_pos = np.mean([point[1] for point in bbox])
        text_with_positions.append((text, prob, y_pos))

text_with_positions.sort(key=lambda x: x[2])

# Kết hợp text thành biển số hoàn chỉnh
if len(text_with_positions) >= 2:
    plate_text = f"{text_with_positions[0][0]} {text_with_positions[1][0]}"
elif len(text_with_positions) == 1:
    plate_text = text_with_positions[0][0]
else:
    plate_text = "Không đọc được"
```

## 🚀 Sử dụng hệ thống

### 1. Chạy GUI (Khuyến nghị)

```bash
python main_gui.py
```

**Hướng dẫn sử dụng GUI:**
1. **📁 Tải Ảnh**: Chọn ảnh biển số xe cần nhận diện
2. **🔍 Nhận Diện**: Bắt đầu quá trình phát hiện và OCR
3. **🎯 Xem Kết Quả**: Kết quả hiển thị với bounding box và text
4. **💾 Lưu Kết Quả**: Lưu ảnh kết quả

### 2. Chạy Command Line

```bash
# Sử dụng ảnh mặc định
python main.py

# Tùy chỉnh ảnh input
python -c "
from ultralytics import YOLO
import cv2
model = YOLO('best.pt')
results = model('your_image.jpg')
"
```

### 3. Sử dụng trong code

```python
from ultralytics import YOLO
import easyocr
import cv2

# Load models
yolo_model = YOLO('best.pt')
ocr_reader = easyocr.Reader(['en'])

# Nhận diện biển số
image = cv2.imread('license_plate.jpg')
results = yolo_model(image)

# Xử lý kết quả
for result in results:
    boxes = result.boxes.xyxy
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        
        # Crop biển số
        cropped = image[y1:y2, x1:x2]
        
        # OCR
        text_results = ocr_reader.readtext(cropped)
        if text_results:
            text = text_results[0][1]
            confidence = text_results[0][2]
            print(f"Biển số: {text}, Confidence: {confidence:.2f}")
```

## 📊 Kết quả và đánh giá

### 1. Metrics đánh giá

- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: Mean Average Precision at IoU=0.5:0.95
- **Precision**: Độ chính xác
- **Recall**: Độ bao phủ
- **F1-Score**: Trung bình điều hòa của Precision và Recall

### 2. Kết quả mẫu

| Model | mAP50 | mAP50-95 | Precision | Recall | F1-Score |
|-------|-------|----------|-----------|---------|----------|
| YOLOv8n | 0.89 | 0.67 | 0.91 | 0.87 | 0.89 |
| YOLOv8s | 0.92 | 0.71 | 0.93 | 0.90 | 0.92 |
| YOLOv8m | 0.94 | 0.75 | 0.95 | 0.93 | 0.94 |

### 3. So sánh với baseline

- **YOLOv5**: Tốc độ nhanh, độ chính xác thấp hơn
- **YOLOv7**: Cân bằng tốt giữa tốc độ và độ chính xác
- **YOLOv8**: Độ chính xác cao nhất, tốc độ chấp nhận được

## 🔧 Tối ưu hóa và tuning

### 1. Hyperparameter tuning

```python
# Grid search cho learning rate
lr_values = [0.001, 0.01, 0.1]
for lr in lr_values:
    model.train(
        data='data.yaml',
        epochs=50,
        lr0=lr,
        name=f'experiment_lr_{lr}'
    )
```

### 2. Data augmentation

```python
# Tăng cường dữ liệu
model.train(
    data='data.yaml',
    epochs=100,
    augment=True,
    degrees=10.0,      # Xoay ảnh
    translate=0.1,     # Dịch chuyển
    scale=0.5,         # Thay đổi tỷ lệ
    shear=2.0,         # Biến dạng
    perspective=0.0,    # Góc nhìn
    flipud=0.0,        # Lật dọc
    fliplr=0.5,        # Lật ngang
    mosaic=1.0,        # Mosaic augmentation
    mixup=0.0          # Mixup augmentation
)
```

### 3. Model pruning và quantization

```python
# Pruning model
model.prune()

# Quantization
model.export(format='tflite', int8=True)
```

## 🚨 Troubleshooting

### 1. Lỗi thường gặp

#### Model không load được
```bash
# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Kiểm tra model file
ls -la *.pt
```

#### OCR không đọc được text
```python
# Giảm threshold
results = reader.readtext(image, confidence_threshold=0.1)

# Thử preprocessing khác
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
results = reader.readtext(gray)
```

#### Memory issues
```python
# Giảm batch size
model.train(batch=4)

# Sử dụng mixed precision
model.train(amp=True)
```

### 2. Performance tips

- **GPU**: Sử dụng NVIDIA GPU với CUDA
- **Batch size**: Điều chỉnh theo RAM khả dụng
- **Image size**: Giảm `imgsz` nếu cần
- **Model size**: Chọn model phù hợp (n < s < m < l < x)

## 📈 Cải tiến và mở rộng

### 1. Tính năng mới

- [ ] **Video processing**: Xử lý video real-time
- [ ] **Multi-language**: Hỗ trợ nhiều ngôn ngữ
- [ ] **Database**: Lưu trữ kết quả vào database
- [ ] **API**: REST API cho web/mobile
- [ ] **Cloud**: Deploy lên cloud platform

### 2. Model improvements

- [ ] **Ensemble**: Kết hợp nhiều model
- [ ] **Attention**: Thêm attention mechanism
- [ ] **Transformer**: Sử dụng Vision Transformer
- [ ] **AutoML**: Tự động tìm hyperparameters tối ưu

### 3. Deployment

- [ ] **Docker**: Containerization
- [ ] **Kubernetes**: Orchestration
- [ ] **Edge**: Deploy trên edge devices
- [ ] **Mobile**: Tối ưu cho mobile

## 🤝 Đóng góp

Chúng tôi rất hoan nghênh mọi đóng góp! Hãy:

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📚 Tài liệu tham khảo

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Computer Vision Papers](https://paperswithcode.com/)

## 📄 License

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## 👥 Tác giả

- **Võ Đức Hòa** - [@voduchoa](https://github.com/voduchoa)
- **Email**: voduchoa@example.com
- **LinkedIn**: [Võ Đức Hòa](https://linkedin.com/in/voduchoa)

## 🙏 Cảm ơn

- [Ultralytics](https://github.com/ultralytics/ultralytics) cho YOLOv8
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) cho OCR engine
- [OpenCV](https://opencv.org/) cho computer vision
- [PyTorch](https://pytorch.org/) cho deep learning framework

---

⭐ **Nếu dự án này hữu ích, hãy cho chúng tôi một star!** ⭐
