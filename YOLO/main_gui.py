import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import easyocr
import os
from ultralytics import YOLO
from PIL import Image, ImageTk
import threading

class LicensePlateDetectorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🚗 Nhận Diện Biển Số Xe - YOLO v8")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Khởi tạo model và reader
        self.model = None
        self.reader = None
        self.current_image = None
        self.result_image = None
        
        # Tạo giao diện
        self.create_widgets()
        
        # Khởi tạo model trong thread riêng
        self.init_models()
    
    def create_widgets(self):
        # Header
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill='x', padx=10, pady=10)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="🚗 HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE", 
                              font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        
        # Control buttons
        controls_frame = tk.LabelFrame(left_panel, text="🎛️ Điều Khiển", 
                                     font=('Arial', 12, 'bold'), bg='white', fg='#2c3e50')
        controls_frame.pack(fill='x', padx=15, pady=15)
        
        # Upload button
        self.upload_btn = tk.Button(controls_frame, text="📁 Tải Ảnh Lên", 
                                   command=self.upload_image, 
                                   font=('Arial', 11), bg='#3498db', fg='white',
                                   relief='flat', padx=20, pady=10)
        self.upload_btn.pack(fill='x', padx=10, pady=5)
        
        # Detect button
        self.detect_btn = tk.Button(controls_frame, text="🔍 Nhận Diện", 
                                   command=self.detect_license_plate, 
                                   font=('Arial', 11), bg='#27ae60', fg='white',
                                   relief='flat', padx=20, pady=10, state='disabled')
        self.detect_btn.pack(fill='x', padx=10, pady=5)
        
        # Save button
        self.save_btn = tk.Button(controls_frame, text="💾 Lưu Kết Quả", 
                                 command=self.save_result, 
                                 font=('Arial', 11), bg='#f39c12', fg='white',
                                 relief='flat', padx=20, pady=10, state='disabled')
        self.save_btn.pack(fill='x', padx=10, pady=5)
        
        # Clear button
        self.clear_btn = tk.Button(controls_frame, text="🗑️ Xóa", 
                                  command=self.clear_all, 
                                  font=('Arial', 11), bg='#e74c3c', fg='white',
                                  relief='flat', padx=20, pady=10)
        self.clear_btn.pack(fill='x', padx=10, pady=5)
        
        # Status frame
        status_frame = tk.LabelFrame(left_panel, text="📊 Trạng Thái", 
                                   font=('Arial', 12, 'bold'), bg='white', fg='#2c3e50')
        status_frame.pack(fill='x', padx=15, pady=15)
        
        self.status_label = tk.Label(status_frame, text="⏳ Đang khởi tạo...", 
                                    font=('Arial', 10), bg='white', fg='#7f8c8d')
        self.status_label.pack(padx=10, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.pack(fill='x', padx=10, pady=(0, 10))
        
        # Results frame
        results_frame = tk.LabelFrame(left_panel, text="🎯 Kết Quả", 
                                    font=('Arial', 12, 'bold'), bg='white', fg='#2c3e50')
        results_frame.pack(fill='x', padx=15, pady=15)
        
        self.result_text = tk.Text(results_frame, height=8, width=30, 
                                  font=('Arial', 10), bg='#ecf0f1', fg='#2c3e50')
        self.result_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Right panel - Image display
        right_panel = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Image display frame
        image_frame = tk.LabelFrame(right_panel, text="🖼️ Hiển Thị Ảnh", 
                                  font=('Arial', 12, 'bold'), bg='white', fg='#2c3e50')
        image_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Canvas for image display
        self.canvas = tk.Canvas(image_frame, bg='#ecf0f1', relief='sunken', bd=2)
        self.canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Scrollbars for canvas
        h_scrollbar = tk.Scrollbar(image_frame, orient='horizontal', command=self.canvas.xview)
        v_scrollbar = tk.Scrollbar(image_frame, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        h_scrollbar.pack(side='bottom', fill='x')
        v_scrollbar.pack(side='right', fill='y')
        
        # Footer
        footer_frame = tk.Frame(self.root, bg='#34495e', height=40)
        footer_frame.pack(fill='x', side='bottom')
        footer_frame.pack_propagate(False)
        
        footer_label = tk.Label(footer_frame, text="© 2024 Hệ Thống Nhận Diện Biển Số Xe - YOLO v8", 
                               font=('Arial', 9), fg='white', bg='#34495e')
        footer_label.pack(expand=True)
    
    def init_models(self):
        """Khởi tạo model YOLO và EasyOCR trong thread riêng"""
        def load_models():
            try:
                self.status_label.config(text="🔄 Đang tải model YOLO...")
                self.progress.start()
                
                # Load YOLO model
                self.model = YOLO("best.pt")
                
                self.status_label.config(text="🔄 Đang tải EasyOCR...")
                
                # Load EasyOCR
                self.reader = easyocr.Reader(['en'])
                
                self.status_label.config(text="✅ Sẵn sàng nhận diện!")
                self.progress.stop()
                self.upload_btn.config(state='normal')
                
            except Exception as e:
                self.status_label.config(text=f"❌ Lỗi: {str(e)}")
                self.progress.stop()
                messagebox.showerror("Lỗi", f"Không thể khởi tạo model:\n{str(e)}")
        
        # Chạy trong thread riêng để không block GUI
        threading.Thread(target=load_models, daemon=True).start()
    
    def upload_image(self):
        """Upload ảnh từ file"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh biển số xe",
            filetypes=[
                ("Ảnh", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Tất cả", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Đọc ảnh
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    raise Exception("Không thể đọc ảnh")
                
                # Hiển thị ảnh
                self.display_image(self.current_image)
                
                # Cập nhật trạng thái
                self.status_label.config(text=f"📁 Đã tải: {os.path.basename(file_path)}")
                self.detect_btn.config(state='normal')
                
                # Xóa kết quả cũ
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Ảnh đã sẵn sàng để nhận diện!\n")
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải ảnh:\n{str(e)}")
    
    def display_image(self, image):
        """Hiển thị ảnh trên canvas"""
        # Resize ảnh để vừa với canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1:  # Canvas chưa được render
            canvas_width = 600
            canvas_height = 400
        
        # Tính tỷ lệ resize
        img_height, img_width = image.shape[:2]
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        if scale < 1:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            image_resized = cv2.resize(image, (new_width, new_height))
        else:
            image_resized = image
        
        # Chuyển đổi BGR sang RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Chuyển sang PIL Image
        pil_image = Image.fromarray(image_rgb)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Hiển thị trên canvas
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=self.photo, anchor='center'
        )
        
        # Cập nhật scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def detect_license_plate(self):
        """Nhận diện biển số xe"""
        if self.current_image is None or self.model is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh trước!")
            return
        
        # Disable buttons trong quá trình xử lý
        self.detect_btn.config(state='disabled')
        self.upload_btn.config(state='disabled')
        self.status_label.config(text="🔍 Đang nhận diện...")
        self.progress.start()
        
        # Chạy nhận diện trong thread riêng
        threading.Thread(target=self._detect_thread, daemon=True).start()
    
    def _detect_thread(self):
        """Thread xử lý nhận diện"""
        try:
            # Nhận diện với YOLO
            results = self.model.predict(source=self.current_image, show=False)
            
            # Tạo bản sao ảnh để vẽ kết quả
            result_image = self.current_image.copy()
            detected_plates = []
            
            # Xử lý kết quả
            for result in results:
                boxes = result.boxes.xyxy
                confidences = result.boxes.conf
                
                for i, (box, confidence) in enumerate(zip(boxes, confidences)):
                    if confidence > 0.25:
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Crop biển số
                        cropped_plate = self.current_image[y1:y2, x1:x2]
                        
                        # Preprocessing
                        enhanced_plate = self.enhance_image(cropped_plate)
                        
                        # OCR
                        text_results = self.reader.readtext(enhanced_plate)
                        
                        if text_results:
                            # Sắp xếp theo vị trí Y
                            text_with_positions = []
                            for (bbox, text, prob) in text_results:
                                if prob > 0.2:
                                    y_pos = np.mean([point[1] for point in bbox])
                                    text_with_positions.append((text, prob, y_pos))
                            
                            text_with_positions.sort(key=lambda x: x[2])
                            
                            if len(text_with_positions) >= 2:
                                plate_text = f"{text_with_positions[0][0]} {text_with_positions[1][0]}"
                            elif len(text_with_positions) == 1:
                                plate_text = text_with_positions[0][0]
                            else:
                                plate_text = "Không đọc được"
                            
                            detected_plates.append({
                                'text': plate_text,
                                'confidence': confidence,
                                'box': (x1, y1, x2, y2)
                            })
                            
                            # Vẽ bounding box và text
                            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(result_image, plate_text, (x1, y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Cập nhật GUI trong main thread
            self.root.after(0, lambda: self._update_results(result_image, detected_plates))
            
        except Exception as e:
            self.root.after(0, lambda: self._show_error(f"Lỗi nhận diện: {str(e)}"))
    
    def enhance_image(self, image):
        """Tăng cường chất lượng ảnh"""
        # Tăng kích thước
        scale_factor = 3
        height, width = image.shape[:2]
        new_width = width * scale_factor
        new_height = height * scale_factor
        
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Làm mịn
        smoothed = cv2.GaussianBlur(resized, (3, 3), 0)
        
        # Tăng độ tương phản
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def _update_results(self, result_image, detected_plates):
        """Cập nhật kết quả lên GUI"""
        self.result_image = result_image
        self.display_image(result_image)
        
        # Cập nhật text kết quả
        self.result_text.delete(1.0, tk.END)
        
        if detected_plates:
            self.result_text.insert(tk.END, f"🎯 Tìm thấy {len(detected_plates)} biển số:\n\n")
            for i, plate in enumerate(detected_plates, 1):
                self.result_text.insert(tk.END, f"Biển số #{i}:\n")
                self.result_text.insert(tk.END, f"  Văn bản: {plate['text']}\n")
                self.result_text.insert(tk.END, f"  Độ tin cậy: {plate['confidence']:.2f}\n\n")
        else:
            self.result_text.insert(tk.END, "❌ Không tìm thấy biển số xe trong ảnh")
        
        # Cập nhật trạng thái
        self.status_label.config(text="✅ Hoàn thành nhận diện!")
        self.progress.stop()
        
        # Enable buttons
        self.detect_btn.config(state='normal')
        self.upload_btn.config(state='normal')
        self.save_btn.config(state='normal')
    
    def _show_error(self, error_msg):
        """Hiển thị lỗi"""
        self.status_label.config(text="❌ Có lỗi xảy ra!")
        self.progress.stop()
        self.detect_btn.config(state='normal')
        self.upload_btn.config(state='normal')
        messagebox.showerror("Lỗi", error_msg)
    
    def save_result(self):
        """Lưu ảnh kết quả"""
        if self.result_image is None:
            messagebox.showwarning("Cảnh báo", "Không có kết quả để lưu!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Lưu ảnh kết quả",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG", "*.jpg"),
                ("PNG", "*.png"),
                ("Tất cả", "*.*")
            ]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.result_image)
                messagebox.showinfo("Thành công", f"Đã lưu ảnh kết quả vào:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu ảnh:\n{str(e)}")
    
    def clear_all(self):
        """Xóa tất cả"""
        self.current_image = None
        self.result_image = None
        self.canvas.delete("all")
        self.result_text.delete(1.0, tk.END)
        self.status_label.config(text="✅ Sẵn sàng nhận diện!")
        self.detect_btn.config(state='disabled')
        self.save_btn.config(state='disabled')
        self.upload_btn.config(state='normal')
    
    def run(self):
        """Chạy ứng dụng"""
        self.root.mainloop()

if __name__ == "__main__":
    app = LicensePlateDetectorGUI()
    app.run()
