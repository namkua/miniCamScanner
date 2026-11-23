# Mini CamScanner - Hệ thống Quét & Cải Thiện Chất Lượng Tài Liệu

Ứng dụng quét tài liệu thông minh kết hợp mô hình học sâu **YOLO11n-Pose (ONNX Runtime)** để định vị chính xác 4 góc tài liệu và chuỗi thuật toán **Xử lý ảnh số (Digital Image Processing)** để nắn phẳng phối cảnh, triệt tiêu bóng đổ và tăng cường độ sắc nét văn bản (**Magic Color**).

---

## 🌟 Tính năng nổi bật

1. **Định vị 4 góc tài liệu tự động bằng YOLO11n-Pose ONNX (`src/core/detector.py`):**
   - Tiền xử lý **Letterbox `640×640`** giữ nguyên tỷ lệ ảnh gốc và suy luận trên CPU qua `onnxruntime` (`CPUExecutionProvider`, tối ưu đồ thị `ORT_ENABLE_ALL`, cache model bằng `@lru_cache`).
   - Trả về trực tiếp hộp bao và 4 tọa độ đỉnh góc theo thứ tự chiều kim đồng hồ **`[TL, TR, BR, BL]`**.
   - Hỗ trợ điều chỉnh ngưỡng tin cậy (`conf_threshold`, mặc định `0.25`) và tự động kích hoạt khung dự phòng lùi viền `2%` khi không phát hiện tài liệu vượt ngưỡng.
2. **Sắp xếp góc & Nắn phẳng phối cảnh (`src/core/transform.py`):**
   - Sắp xếp 4 điểm chuẩn xác theo tổng/hiệu tọa độ (`order_points`) và tính kích thước mặt phẳng đích tối đa.
   - Chiếu phối cảnh (`cv2.warpPerspective`) với nội suy song khối (**`cv2.INTER_CUBIC`**) giúp giữ độ sắc nét tối đa cho chữ viết và bảng biểu.
3. **Triệt tiêu bóng đổ & Bộ lọc Magic Color (`src/filters/shadow.py`, `src/filters/enhancement.py`):**
   - **Khử bóng trên kênh độ sáng `L` (CIELAB):** Ước lượng bản đồ nền sáng (Background Map) bằng phép đóng hình thái học (**Morphological Closing** nhân Elip kích thước thích ứng `max(shadow_ksize, (w // 18) | 1)`) kết hợp **Gaussian Blur** để xóa quầng viền (halo), sau đó chia chuẩn hóa nền (`cv2.divide`).
   - **Kéo dãn mức sáng (Levels) & Gamma phi tuyến:** Đưa nền giấy về trắng tinh (`black_point=50.0`, `white_point=218.0`) và làm đậm nét mực (`text_boldness`).
   - **Bảo toàn & tăng tươi màu sắc:** Tăng bão hòa màu kênh `S` trong không gian **HSV** (`× 1.3`) để giữ rõ màu đỏ con dấu và chữ ký xanh; làm đanh viền chữ với **Unsharp Masking** (`sigma=1.0`).
4. **Đóng gói Pipeline & Xuất PDF (`src/pipeline.py`):**
   - Lớp `DocumentScanner` tích hợp toàn bộ quy trình `detect_corners` → `warp` → `rotate_image` (`0°/90°/180°/270°`) → `apply_filter` và xuất file PDF chuẩn `150 DPI` (`export_pdf`).
5. **Giao diện Web tương tác Streamlit (`src/ui/app.py`):**
   - Tải ảnh chụp tài liệu (`JPG`, `PNG`, `JPEG`) và tinh chỉnh thời gian thực các tham số: *Confidence Threshold*, *Xoay ảnh*, *Text Sharpness*, *Text Boldness*, *Shadow Kernel Size*.
   - Chế độ xem các bước trung gian (**DIP Stages**): `YOLO Letterbox (640×640)`, `Box & 4 góc YOLO11n-Pose`, và `Background Map`.
   - Tải về trực tiếp ảnh scan (**PNG**) hoặc đóng gói thành tài liệu **PDF**.

---

## 📊 Kết quả huấn luyện & Benchmark mô hình

Mô hình `models/yolo11n_doc_corner.onnx` được fine-tune từ `yolo11n-pose.pt` trên bộ dữ liệu **`mapo80/DocCornerDataset`** (`48,265` ảnh: `32,968` train, `8,645` val, `6,652` test gồm cả mẫu dương và mẫu âm) trong `30 epochs` trên GPU NVIDIA L4 (`notebooks/train_yolo11n_pose_colab copy.ipynb`).

Kết quả benchmark trên **500 ảnh test phân tầng** (`441` positive, `59` negative):

| Phương pháp / Mô hình | mIoU (%) | F1-Score (%) | Corner Error NME (%) | Latency (ms/ảnh) |
| :--- | :---: | :---: | :---: | :---: |
| Classical OpenCV (`Auto-Canny` + `Otsu` + `Contour`) | 63.99 | 64.71 | 15.69 | 74.74 |
| YOLO11n-Pose Pretrained (Trước fine-tune) | 0.12 | 0.00 | 74.06 | 78.55 |
| YOLO11n-Pose Fine-tuned (`.pt`) | 89.86 | 97.59 | 4.79 | 28.54 |
| **YOLO11n-Pose Fine-tuned (`.onnx` CPU)** | **90.34** | **97.59** | **4.32** | **35.14** |

---

## 📁 Cấu trúc thư mục

```text
miniCamscanner/
├── README.md                                   # Tài liệu hướng dẫn dự án
├── requirements.txt                            # Các thư viện Python yêu cầu
├── run.sh                                      # Script khởi chạy nhanh Streamlit UI
├── models/
│   └── yolo11n_doc_corner.onnx                 # Trọng số YOLO11n-Pose ONNX (11.0 MB)
├── notebooks/
│   └── train_yolo11n_pose_colab copy.ipynb     # Notebook train & benchmark trên Colab
└── src/
    ├── __init__.py
    ├── pipeline.py                             # Lớp DocumentScanner tích hợp toàn bộ luồng xử lý
    ├── core/
    │   ├── __init__.py
    │   ├── detector.py                         # YoloCornerDetector & find_document_corners (ONNX)
    │   └── transform.py                        # order_points & four_point_transform (INTER_CUBIC)
    ├── filters/
    │   ├── __init__.py
    │   ├── shadow.py                           # remove_shadow_gray (Morphological Background Division)
    │   └── enhancement.py                      # magic_color_filter & unsharp_mask
    └── ui/
        └── app.py                              # Giao diện Web tương tác Streamlit
```

---

## 🚀 Hướng dẫn cài đặt & Khởi chạy

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy Giao diện Web
Chạy bằng lệnh Streamlit trực tiếp:
```bash
streamlit run src/ui/app.py
```
Hoặc dùng script khởi chạy nhanh có sẵn (tự động ưu tiên `/opt/anaconda3/bin/streamlit` nếu có):
```bash
./run.sh
```

Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501`.
