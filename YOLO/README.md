# 🚗 Hệ Thống Nhận Diện Biển Số Xe - YOLO v8

## 📋 Mô tả dự án

Hệ thống nhận diện biển số xe sử dụng YOLO v8 để phát hiện biển số và EasyOCR để đọc văn bản. Dự án bao gồm cả giao diện GUI và script command line.

## 🚀 Tính năng chính

- **Phát hiện biển số**: Sử dụng YOLO v8 model đã được train
- **OCR văn bản**: Đọc biển số bằng EasyOCR
- **Giao diện GUI**: Tkinter interface thân thiện
- **Xử lý ảnh**: Tăng cường chất lượng ảnh tự động
- **Lưu kết quả**: Export ảnh với bounding box và text

## 📁 Cấu trúc dự án

```
YOLO/
├── main.py              # Script command line chính
├── main_gui.py          # Giao diện GUI
├── best.pt              # Model YOLO đã train
├── yolov8n.pt           # Model YOLO gốc
├── requirements.txt     # Dependencies
├── README.md           # Hướng dẫn này
├── bien1.jpg           # Ảnh test 1
├── bien2.jpg           # Ảnh test 2
└── coco128/            # Dataset training
```

## 🛠️ Cài đặt

### 1. Cài đặt Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Kiểm tra model files

Đảm bảo các file sau tồn tại:
- `best.pt` - Model YOLO đã train
- `yolov8n.pt` - Model YOLO gốc (backup)

## 🎯 Cách sử dụng

### Chạy GUI (Khuyến nghị)

```bash
python main_gui.py
```

**Hướng dẫn sử dụng GUI:**
1. **Tải ảnh**: Nhấn "📁 Tải Ảnh Lên" để chọn ảnh biển số
2. **Nhận diện**: Nhấn "🔍 Nhận Diện" để bắt đầu xử lý
3. **Xem kết quả**: Kết quả hiển thị trong panel bên phải
4. **Lưu kết quả**: Nhấn "💾 Lưu Kết Quả" để lưu ảnh

### Chạy Command Line

```bash
python main.py
```

**Tùy chỉnh ảnh input:**
Sửa dòng `source = "bien2.jpg"` trong `main.py` để thay đổi ảnh test.

## 🔧 Cấu hình

### Threshold settings

- **YOLO confidence**: 0.25 (trong `main.py` và `main_gui.py`)
- **OCR confidence**: 0.2 (độ tin cậy tối thiểu cho text)

### Model paths

- **YOLO model**: `best.pt` (model đã train)
- **EasyOCR**: `['en']` (tiếng Anh)

## 📊 Kết quả output

### Files được tạo

- `result_detection.jpg` - Ảnh kết quả với bounding box
- `cropped_plate_X_original.jpg` - Ảnh crop gốc
- `cropped_plate_X_enhanced.jpg` - Ảnh đã xử lý

### Thông tin hiển thị

- Vị trí biển số (bounding box)
- Văn bản biển số được đọc
- Độ tin cậy của detection
- Thời gian xử lý

## 🎨 Giao diện GUI

### Layout

- **Header**: Tiêu đề ứng dụng
- **Left Panel**: 
  - Điều khiển (Upload, Detect, Save, Clear)
  - Trạng thái và progress bar
  - Kết quả text
- **Right Panel**: Hiển thị ảnh với scrollbars
- **Footer**: Thông tin bản quyền

### Features

- **Multi-threading**: Không block GUI khi xử lý
- **Real-time status**: Cập nhật trạng thái liên tục
- **Image scaling**: Tự động resize ảnh vừa màn hình
- **Error handling**: Xử lý lỗi gracefully

## 🔍 Xử lý ảnh

### Preprocessing pipeline

1. **Resize**: Tăng kích thước 3x với INTER_CUBIC
2. **Smoothing**: Gaussian blur để giảm nhiễu
3. **Contrast enhancement**: CLAHE để tăng độ tương phản
4. **Color space**: Chuyển đổi LAB để xử lý L channel

### OCR strategy

- **Dual approach**: Thử cả ảnh gốc và ảnh đã xử lý
- **Position sorting**: Sắp xếp text theo vị trí Y
- **Confidence filtering**: Lọc kết quả theo độ tin cậy
- **Text combination**: Kết hợp text từ nhiều dòng

## 🚨 Troubleshooting

### Lỗi thường gặp

1. **Model không load được**
   - Kiểm tra file `best.pt` tồn tại
   - Cài đặt đúng version ultralytics

2. **OCR không đọc được text**
   - Giảm threshold confidence
   - Kiểm tra chất lượng ảnh input
   - Thử với ảnh khác

3. **GUI không hiển thị**
   - Kiểm tra tkinter được cài đặt
   - Chạy với Python 3.x

### Performance tips

- Sử dụng GPU nếu có (CUDA)
- Giảm kích thước ảnh input nếu cần
- Tắt real-time display trong command line mode

## 📈 Cải tiến tương lai

- [ ] Hỗ trợ video stream
- [ ] Multi-language OCR
- [ ] Database lưu trữ kết quả
- [ ] API web interface
- [ ] Mobile app
- [ ] Real-time processing

## 📞 Hỗ trợ

Nếu gặp vấn đề, hãy kiểm tra:
1. Logs trong terminal
2. File requirements.txt
3. Version Python và dependencies
4. Model files tồn tại

## 📄 License

© 2024 Hệ Thống Nhận Diện Biển Số Xe - YOLO v8
