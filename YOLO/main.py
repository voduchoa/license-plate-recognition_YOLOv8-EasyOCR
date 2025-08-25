from ultralytics import YOLO
import easyocr
import cv2
import os
import numpy as np

# Initialize YOLO model
model = YOLO("best.pt")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Source image
source = "bien2.jpg"

# Check if source image exists
if not os.path.exists(source):
    print(f"Không tìm thấy ảnh: {source}")
    print("Các ảnh có sẵn:")
    for file in os.listdir("."):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif', '.webp')):
            print(f"  - {file}")
    exit(1)

print(f"Đang xử lý ảnh: {source}")

# Load the image
image = cv2.imread(source)
if image is None:
    print(f"Không thể đọc ảnh: {source}")
    exit(1)

# Start prediction (chỉ chạy một lần)
print("Đang nhận diện biển số xe...")
results = model.predict(source=image, show=False)  # Không hiển thị trong quá trình predict

# Process results
detected_text = "Không tìm thấy biển số"
for result in results:
    boxes = result.boxes.xyxy  # Bounding box coordinates (x1, y1, x2, y2)
    confidences = result.boxes.conf  # Confidence scores
    
    if len(boxes) == 0:
        print("Không tìm thấy biển số xe trong ảnh")
        continue
        
    for i, (box, confidence) in enumerate(zip(boxes, confidences)):
        if confidence > 0.25:  # Ensure the detection is confident enough
            x1, y1, x2, y2 = map(int, box)
            print(f"Phát hiện biển số #{i+1} với độ tin cậy: {confidence:.2f}")
            
            # Crop the detected license plate area
            cropped_plate = image[y1:y2, x1:x2]
            
            # Lưu ảnh gốc crop để kiểm tra
            cv2.imwrite(f"cropped_plate_{i+1}_original.jpg", cropped_plate)
            
            # Preprocessing để khắc phục dè chữ
            # 1. Giữ nguyên tỷ lệ khung hình
            height, width = cropped_plate.shape[:2]
            
            # 2. Tăng kích thước ảnh để dễ đọc hơn (giữ tỷ lệ)
            scale_factor = 3
            new_width = width * scale_factor
            new_height = height * scale_factor
            
            # 3. Resize với interpolation tốt hơn
            resized_plate = cv2.resize(cropped_plate, (new_width, new_height), 
                                     interpolation=cv2.INTER_CUBIC)
            
            # 4. Làm mịn ảnh để giảm nhiễu
            smoothed_plate = cv2.GaussianBlur(resized_plate, (3, 3), 0)
            
            # 5. Tăng độ tương phản
            lab = cv2.cvtColor(smoothed_plate, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            enhanced_plate = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Lưu ảnh đã xử lý để kiểm tra
            cv2.imwrite(f"cropped_plate_{i+1}_enhanced.jpg", enhanced_plate)
            
            # 6. Thử OCR với ảnh gốc trước
            print("  Đang đọc với ảnh gốc...")
            text_results_original = reader.readtext(cropped_plate)
            
            # 7. Thử OCR với ảnh đã xử lý
            print("  Đang đọc với ảnh đã xử lý...")
            text_results_enhanced = reader.readtext(enhanced_plate)
            
            # Kết hợp kết quả từ cả hai phương pháp
            all_text_results = text_results_original + text_results_enhanced
            
            # Extract and print the detected text
            if all_text_results:
                print(f"  Tìm thấy {len(all_text_results)} kết quả OCR:")
                
                # Tách các phần text và sắp xếp theo vị trí Y (từ trên xuống dưới)
                text_with_positions = []
                for (bbox, text, prob) in all_text_results:
                    if prob > 0.2:  # Giảm threshold để lấy nhiều text hơn
                        # Tính vị trí Y trung bình của text
                        y_pos = np.mean([point[1] for point in bbox])
                        text_with_positions.append((text, prob, y_pos))
                
                # Sắp xếp theo vị trí Y (từ trên xuống dưới)
                text_with_positions.sort(key=lambda x: x[2])
                
                # Kết hợp các text thành biển số hoàn chỉnh
                if len(text_with_positions) >= 2:
                    # Lấy 2 text đầu tiên (dòng trên và dòng dưới)
                    top_text = text_with_positions[0][0]
                    bottom_text = text_with_positions[1][0]
                    
                    # Kết hợp thành biển số hoàn chỉnh
                    full_plate = f"{top_text} {bottom_text}"
                    
                    detected_text = full_plate
                    print(f"  → Biển số hoàn chỉnh: '{full_plate}'")
                    
                    # Draw the bounding box and text on the image
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, full_plate, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                elif len(text_with_positions) == 1:
                    # Chỉ có 1 text
                    single_text = text_with_positions[0][0]
                    detected_text = single_text
                    print(f"  → Chỉ tìm thấy 1 dòng: '{single_text}'")
                    
                    # Draw the bounding box and text on the image
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, single_text, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                else:
                    print("  → Không có kết quả đủ tin cậy")
                    # Still draw the bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(image, "No text", (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                print("  Không thể đọc văn bản từ biển số")
                # Still draw the bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(image, "No text", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Save the result image
output_filename = "result_detection.jpg"
cv2.imwrite(output_filename, image)
print(f"\nKết quả đã được lưu vào: {output_filename}")

# Hiển thị ảnh kết quả (chỉ một lần)
try:
    cv2.imshow("License Plate Detection Result", image)
    print("Nhấn phím 'q' để đóng cửa sổ...")
    
    # Chờ người dùng nhấn phím 'q' để thoát
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("Đã đóng cửa sổ hiển thị")
    
except Exception as e:
    print(f"Không thể hiển thị cửa sổ: {e}")
    print("Ảnh kết quả đã được lưu, bạn có thể mở để xem")

print(f"\n🎯 Kết quả cuối cùng: {detected_text}")
print("Hoàn thành nhận diện biển số xe!")
print("Bạn có thể mở file 'result_detection.jpg' để xem kết quả!")
print("\n📁 Các file ảnh crop đã được lưu:")
print("  - cropped_plate_1_original.jpg: Ảnh crop gốc")
print("  - cropped_plate_1_enhanced.jpg: Ảnh đã xử lý")

