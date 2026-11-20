# CamScanner Pro - Hệ thống Quét & Cải Thiện Chất Lượng Tài Liệu

Dự án Xử lý ảnh (Digital Image Processing) mô phỏng và cải tiến tính năng quét tài liệu của **CamScanner**: tự động phát hiện biên tờ giấy, nắn thẳng góc nhìn (Perspective Transform), triệt tiêu bóng đổ (Shadow Removal), và áp dụng các bộ lọc chuyên dụng (Magic Color, B&W Sauvola, Adaptive, Gray Crisp).

---

## 🌟 Tính năng nổi bật

1. **Tự động định vị 4 góc tài liệu (Corner Detection):**
   - Áp dụng **Bilateral Filter** khử nhiễu bề mặt nhưng giữ sắc nét biên mép.
   - **Auto-Canny thích ứng:** Tự động tính 2 ngưỡng Canny dựa trên giá trị trung vị (median) của ảnh.
   - **Phép đóng hình thái học (Morphological Close)** nối liền các nét đứt gãy của đường viền.
   - **Lọc tứ giác lồi & Heuristic diện tích** kết hợp cơ chế Fallback thông minh.
2. **Nắn phẳng phối cảnh & Cắt nền thừa (Perspective Warp):**
   - Thuật toán sắp xếp 4 góc theo vector cực tâm, đảm bảo chiều xoay chuẩn xác.
   - Chiếu phối cảnh với nội suy song khối (`cv2.INTER_CUBIC`) cho độ sắc nét tối đa.
3. **Triệt tiêu bóng đổ (Shadow & Non-Uniform Illumination Removal):**
   - Thuật toán **Morphological Background Division**: Tách bản đồ nền bằng phép đóng kernel Elip lớn, sau đó chia ảnh cho nền để xóa sổ 100% bóng tay, bóng đèn loang lổ.
4. **Bộ 4 chế độ lọc scan chuyên nghiệp:**
   - **Magic Color (Khuyên dùng):** Giữ nguyên con dấu mộc đỏ, chữ ký xanh, dạ quang; nền trắng tinh khiết; nét chữ đanh sắc bằng Unsharp Masking.
   - **B&W Clean (Sauvola):** Phân ngưỡng tài liệu chuyên dụng, chữ nét đậm rõ, không bị đứt nét.
   - **B&W Adaptive:** Phân ngưỡng thích nghi Gaussian tiêu chuẩn.
   - **Gray Crisp:** Ảnh xám tối ưu tương phản bằng CLAHE.
5. **Giao diện Web tương tác (Streamlit):**
   - Tải ảnh lên hoặc dùng ảnh mẫu hợp đồng có sẵn.
   - Trực quan hóa 4 góc và cho phép tinh chỉnh thủ công nếu cần.
   - Xem toàn bộ các bước trung gian của thuật toán Xử lý ảnh (Debug Stages).
   - Tải về ảnh Scan (PNG) hoặc đóng gói file PDF chất lượng cao.

---

## 📁 Cấu trúc thư mục

```
XLA/
├── requirements.txt            # Thư viện yêu cầu
├── README.md                   # Tài liệu hướng dẫn
├── src/
│   ├── core/
│   │   ├── detector.py         # Auto-Canny, Contours & Dò 4 góc
│   │   └── transform.py        # Sắp xếp góc & Nắn phối cảnh
│   ├── filters/
│   │   ├── shadow.py           # Triệt tiêu bóng đổ bằng Morphology
│   │   ├── binarization.py     # Phân ngưỡng Sauvola & Adaptive
│   │   └── enhancement.py      # Magic Color & Gray Crisp
│   ├── pipeline.py             # Lớp DocumentScanner tích hợp toàn bộ
│   └── ui/
│       └── app.py              # Giao diện Web tương tác Streamlit
└── data/
    └── samples/                # Ảnh tài liệu mẫu thử nghiệm
```

---

## 🚀 Hướng dẫn cài đặt & Khởi chạy

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy Giao diện Web
```bash
streamlit run src/ui/app.py
```
*(Nếu dùng môi trường Anaconda trên máy của bạn: `/opt/anaconda3/bin/streamlit run src/ui/app.py`)*

Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501`.
