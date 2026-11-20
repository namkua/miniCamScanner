import os
import cv2
import numpy as np


def auto_canny(image, sigma=0.33):
    """
    Tự động xác định ngưỡng Canny dựa trên giá trị trung vị (median) của độ sáng ảnh.
    Giúp phát hiện biên ổn định trong nhiều điều kiện sáng khác nhau.
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def _extract_quad_from_mask(mask, new_w, canonical_height, min_area_ratio=0.15):
    """
    Tìm tứ giác lồi 4 góc từ mask nhị phân (0/255) với đệm viền đen 2px
    để xử lý cả trường hợp tờ giấy chạm sát mép ảnh.
    """
    padded = cv2.copyMakeBorder(mask, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    min_area = (new_w * canonical_height) * min_area_ratio
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        for geom in [c, cv2.convexHull(c)]:
            peri = cv2.arcLength(geom, True)
            for eps_factor in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
                approx = cv2.approxPolyDP(geom, eps_factor * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    pts = np.clip(approx.reshape(4, 2) - 2, [0, 0], [new_w - 1, canonical_height - 1])
                    return pts.reshape(4, 1, 2)
    return None


def _detect_with_ai(resized, new_w, canonical_height):
    """
    Nhận diện vùng tài liệu bằng mô hình AI (ONNX Runtime hoặc Ultralytics YOLOv8-Seg).
    Tìm file weights trong thư mục `models/`:
      - `models/deeplabv3_mbv3_docseg.onnx` hoặc `models/doc_seg.onnx` (chạy qua onnxruntime)
      - `models/doc_seg.pt` (chạy qua ultralytics nếu đã cài)
    Trả về (target_contour, ai_status, ai_message).
    """
    onnx_candidates = [
        "models/deeplabv3_mbv3_docseg.onnx",
        "models/doc_seg.onnx",
    ]
    onnx_path = next((p for p in onnx_candidates if os.path.exists(p)), None)

    if onnx_path is not None:
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            input_meta = session.get_inputs()[0]
            in_shape = input_meta.shape
            in_h = in_shape[2] if isinstance(in_shape[2], int) else 384
            in_w = in_shape[3] if isinstance(in_shape[3], int) else 384

            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            blob = cv2.resize(rgb, (in_w, in_h)).astype(np.float32) / 255.0
            fname = os.path.basename(onnx_path)
            if "u2netp" in fname or "docseg" in fname:
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                blob = (blob - mean) / std
            blob = np.transpose(blob, (2, 0, 1))[None, ...]
            outputs = session.run(None, {input_meta.name: blob})

            pred = np.squeeze(outputs[0])
            if pred.ndim == 3 and pred.shape[0] > 1:
                prob = np.argmax(pred, axis=0).astype(np.float32)
            else:
                if pred.ndim == 3:
                    pred = pred[0]
                prob = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

            mask = (cv2.resize(prob, (new_w, canonical_height)) > 0.5).astype(np.uint8) * 255
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
            quad = _extract_quad_from_mask(mask, new_w, canonical_height)
            if quad is not None:
                return quad, "ok", f"Đang dùng model ONNX chuyên dụng: `{onnx_path}`"
            return None, "no_quad", f"Model `{onnx_path}` không tìm thấy vùng tài liệu đủ lớn."
        except Exception as exc:
            return None, "error", f"Lỗi chạy ONNX `{onnx_path}`: {exc}"

    pt_path = "models/doc_seg.pt"
    if os.path.exists(pt_path):
        try:
            from ultralytics import YOLO
            model = YOLO(pt_path)
            results = model.predict(resized, verbose=False)
            if results and results[0].masks is not None and len(results[0].masks.data) > 0:
                raw_mask = results[0].masks.data[0].cpu().numpy()
                mask = (cv2.resize(raw_mask, (new_w, canonical_height)) > 0.5).astype(np.uint8) * 255
                quad = _extract_quad_from_mask(mask, new_w, canonical_height)
                if quad is not None:
                    return quad, "ok", f"Đang dùng model YOLO: `{pt_path}`"
            return None, "no_quad", f"Model `{pt_path}` không phát hiện được mask tài liệu."
        except Exception as exc:
            return None, "error", f"Lỗi chạy YOLO `{pt_path}`: {exc}"

    return None, "missing_model", (
        "Chưa có file model AI trong thư mục `models/` (`models/deeplabv3_mbv3_docseg.onnx`, `models/doc_seg.onnx` hoặc `models/doc_seg.pt`). "
        "Hệ thống tạm dùng thuật toán cổ điển để giữ ứng dụng hoạt động."
    )


def find_document_corners(image, canonical_height=800, method="classical"):
    """
    Tự động phát hiện 4 góc của tờ giấy tài liệu.

    Args:
        image: Ảnh BGR đầu vào.
        canonical_height: Chiều cao chuẩn hóa khi xử lý.
        method:
            - "classical": Thuật toán Xử lý ảnh cổ điển (Auto-Canny + Otsu Paper Mask).
            - "ai": Mô hình Deep Learning Segmentation (ONNX / YOLOv8-Seg), tự fallback về cổ điển nếu thiếu file weights.

    Returns:
        corners: np.ndarray shape (4, 2) theo toạ độ ảnh gốc.
        debug_info: dict chứa các ảnh trung gian và trạng thái nhận diện.
    """
    orig_h, orig_w = image.shape[:2]
    ratio = orig_h / float(canonical_height)
    new_w = int(orig_w / ratio)
    
    # 1. Resize ảnh để đồng nhất kích thước và tăng tốc độ xử lý
    resized = cv2.resize(image, (new_w, canonical_height), interpolation=cv2.INTER_AREA)

    # 2. Tiền xử lý: Chuyển sang ảnh xám và khử nhiễu
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Bilateral Filter giữ cạnh sắc nét hơn Gaussian thông thường
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 3. Auto-Canny dò cạnh thích ứng
    edged = auto_canny(blurred)

    # Dùng phép đóng hình thái học để nối các đường viền mép giấy bị hở
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # 4. Tìm các đường bao (Contours)
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sắp xếp các contour theo diện tích giảm dần
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    min_area = (new_w * canonical_height) * 0.10  # Ít nhất 10% diện tích ảnh
    target_contour = None
    ai_status = None
    ai_message = None

    # Nếu người dùng chọn hướng AI (ONNX / YOLOv8-Seg), thử chạy AI trước
    if method == "ai":
        target_contour, ai_status, ai_message = _detect_with_ai(resized, new_w, canonical_height)

    # Tìm contour tứ giác lồi hợp lệ (Chiến lược 1: Auto-Canny)
    if target_contour is None:
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            peri = cv2.arcLength(c, True)
            # Thử nghiệm với các mức epsilon từ chặt đến nới lỏng
            for eps_factor in [0.02, 0.03, 0.04]:
                approx = cv2.approxPolyDP(c, eps_factor * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    target_contour = approx
                    break
            if target_contour is not None:
                break

    # 5. Chiến lược 2: Phân vùng nền giấy (Morphological Close -> Otsu -> Open + Border Pad)
    # Giúp nhận diện chính xác khi mép giấy mờ, góc bo tròn, hoặc chạm sát cạnh khung hình
    if target_contour is None:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        paper_bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k_close)
        _, otsu = cv2.threshold(paper_bg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        clean_mask = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, k_open)
        target_contour = _extract_quad_from_mask(clean_mask, new_w, canonical_height)

    # 6. Xử lý trường hợp đặc biệt & Fallback
    is_fallback = False
    is_full_frame = False

    if target_contour is None:
        # Kiểm tra xem ảnh có phải là ảnh chụp cận cảnh toàn màn hình (Full-frame Document) không
        # Nếu mép ảnh (4 cạnh) chủ yếu là nền giấy (> 60% pixel sáng), tức tờ giấy đã chiếm trọn khung ảnh
        borders = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        if np.mean(borders > 160) > 0.6:
            is_full_frame = True
            target_contour = np.array([
                [0, 0],
                [new_w - 1, 0],
                [new_w - 1, canonical_height - 1],
                [0, canonical_height - 1]
            ], dtype=np.int32).reshape(4, 1, 2)
        elif len(contours) > 0 and cv2.contourArea(contours[0]) >= min_area:
            is_fallback = True
            # Dùng hình chữ nhật bao quanh diện tích nhỏ nhất (minAreaRect)
            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            target_contour = np.int32(box).reshape(4, 1, 2)
        else:
            is_fallback = True
            # Khung mặc định lùi vào 2% mỗi mép ảnh
            margin_x = int(new_w * 0.02)
            margin_y = int(canonical_height * 0.02)
            box = np.array([
                [margin_x, margin_y],
                [new_w - margin_x, margin_y],
                [new_w - margin_x, canonical_height - margin_y],
                [margin_x, canonical_height - margin_y]
            ], dtype=np.int32)
            target_contour = box.reshape(4, 1, 2)

    # Rescale toạ độ 4 điểm về kích thước ảnh gốc
    corners = target_contour.reshape(4, 2).astype("float32") * ratio

    # Chuẩn bị ảnh debug để hiển thị lên UI
    contour_img = resized.copy()
    cv2.drawContours(contour_img, [target_contour.reshape(4, 2)], -1, (0, 255, 0), 2)
    for pt in target_contour.reshape(4, 2):
        cv2.circle(contour_img, tuple(pt), 6, (0, 0, 255), -1)

    debug_info = {
        "resized": resized,
        "edged": edged,
        "closed": closed,
        "contour_img": contour_img,
        "is_fallback": is_fallback,
        "is_full_frame": is_full_frame,
        "method": method,
        "ai_status": ai_status,
        "ai_message": ai_message,
    }

    return corners, debug_info
