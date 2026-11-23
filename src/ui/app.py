import os
import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image

import sys
# Thêm thư mục gốc vào path để import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.pipeline import DocumentScanner
from src.filters.shadow import remove_shadow_gray
from src.filters.enhancement import magic_color_filter

st.set_page_config(
    page_title="CamScanner Pro - Hệ thống Quét Tài Liệu",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 0.2rem;
    }
    .brand-name {
        color: #00D26A;
        font-weight: 800;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .step-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title"><span class="brand-name">Mini CamScanner - Quét & Cải Thiện Tài Liệu</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Định vị 4 góc bằng YOLO11n-Pose ONNX, nắn phối cảnh, triệt tiêu bóng đổ & bộ lọc Magic Color</div>', unsafe_allow_html=True)


@st.cache_resource
def get_scanner():
    return DocumentScanner()


scanner = get_scanner()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Tuỳ chọn & Cấu hình")
    
    uploaded_file = st.file_uploader("Chọn ảnh chụp tài liệu (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"])
    image = None
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.markdown("---")
    st.subheader("Nhận diện 4 góc (YOLO11n-Pose ONNX)")
    st.caption("Model: `models/yolo11n_doc_corner.onnx`")
    conf_threshold = st.slider(
        "Ngưỡng tin cậy (Confidence):",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )

    rotation = st.selectbox("Xoay tài liệu:", [0, 90, 180, 270], format_func=lambda x: f"{x}°")

    st.markdown("---")
    st.subheader("Cấu hình độ nét & Nâng cao")
    sharpen_strength = st.slider("Độ sắc nét của chữ (Text Sharpness):", min_value=0.5, max_value=2.5, value=1.3, step=0.1)
    text_boldness = st.slider("Độ đậm nét mực (Text Boldness):", min_value=1.0, max_value=2.2, value=1.4, step=0.05)
    shadow_ksize = st.slider("Kích thước Kernel khử bóng:", min_value=15, max_value=85, value=51, step=2)
    show_debug = st.checkbox("Hiển thị các bước trung gian (DIP Stages)", value=True)


# --- MAIN CONTENT ---
if image is not None:
    # 1. Tự động phát hiện 4 góc bằng YOLO11n-Pose ONNX
    orig_h, orig_w = image.shape[:2]
    auto_corners, debug_info = scanner.detect_corners(image, conf_threshold=conf_threshold)

    # Hiển thị thông báo trạng thái nhận diện góc
    if debug_info.get("detected", False):
        st.success(debug_info["status_message"])
    else:
        st.warning(debug_info["status_message"])

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Ảnh gốc & 4 góc định vị")
        preview_corners_img = image.copy()
        pts_int = np.round(auto_corners).astype(int)
        line_thickness = max(2, int(round(min(orig_h, orig_w) / 250)))
        point_radius = max(5, int(round(min(orig_h, orig_w) / 100)))
        font_scale = max(0.6, min(orig_h, orig_w) / 900.0)
        corner_names = ("TL", "TR", "BR", "BL")

        cv2.polylines(preview_corners_img, [pts_int], isClosed=True, color=(0, 255, 0), thickness=line_thickness)
        for label, pt in zip(corner_names, pts_int):
            cv2.circle(preview_corners_img, tuple(pt), point_radius, (0, 0, 255), -1)
            cv2.putText(
                preview_corners_img,
                label,
                (int(pt[0]) + point_radius + 4, int(pt[1]) - point_radius),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 0),
                max(2, line_thickness),
            )
        
        st.image(cv2.cvtColor(preview_corners_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    # 2. Xử lý nắn phối cảnh & áp dụng bộ lọc Magic Color
    warped = scanner.warp(image, auto_corners)
    if rotation != 0:
        warped = scanner.rotate_image(warped, rotation)

    result = magic_color_filter(
        warped, 
        shadow_ksize=shadow_ksize, 
        sharpen_strength=sharpen_strength, 
        text_boldness=text_boldness
    )

    with col2:
        st.subheader("2. Bản Scan Kết Quả")
        if len(result.shape) == 2:
            st.image(result, use_container_width=True, clamp=True)
        else:
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Nút tải ảnh và xuất PDF
        st.markdown("### Tải xuống")
        btn_c1, btn_c2 = st.columns(2)
        
        # 1. Download ảnh
        if len(result.shape) == 2:
            res_pil = Image.fromarray(result)
        else:
            res_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        
        img_buffer = io.BytesIO()
        res_pil.save(img_buffer, format="PNG")
        btn_c1.download_button(
            label="Tải ảnh Scan (PNG)",
            data=img_buffer.getvalue(),
            file_name="camscanner_result.png",
            mime="image/png",
            use_container_width=True
        )

        # 2. Download PDF
        pdf_buffer = io.BytesIO()
        res_pil.convert("RGB").save(pdf_buffer, format="PDF", resolution=150.0)
        btn_c2.download_button(
            label="Xuất file PDF",
            data=pdf_buffer.getvalue(),
            file_name="camscanner_document.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    # 3. Hiển thị các bước trung gian (YOLO11n-Pose & Background Map)
    if show_debug and debug_info:
        st.markdown("---")
        st.subheader("Các bước Xử lý ảnh trung gian (YOLO11n-Pose & DIP Stages)")
        d_col1, d_col2, d_col3 = st.columns(3)

        with d_col1:
            st.caption("1. Ảnh đầu vào YOLO Letterbox (640×640)")
            st.image(cv2.cvtColor(debug_info["letterbox_img"], cv2.COLOR_BGR2RGB), use_container_width=True)

        with d_col2:
            st.caption("2. Định vị Box & 4 góc YOLO11n-Pose")
            st.image(cv2.cvtColor(debug_info["detection_img"], cv2.COLOR_BGR2RGB), use_container_width=True)

        with d_col3:
            st.caption("3. Bản đồ ước lượng nền (Background Map)")
            if len(warped.shape) == 3:
                w_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            else:
                w_gray = warped
            _, bg_map = remove_shadow_gray(w_gray, kernel_size=shadow_ksize)
            st.image(bg_map, use_container_width=True)

else:
    st.info("Vui lòng tải ảnh lên để bắt đầu trải nghiệm!")
