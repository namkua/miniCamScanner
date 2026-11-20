import cv2
import numpy as np


def unsharp_mask(image, sigma=1.0, strength=1.3):
    """
    Làm sắc nét chữ (Unsharp Masking): Image_sharp = Image + strength * (Image - Blurred)
    Sử dụng bán kính hẹp sigma = 1.0 để tập trung tối đa vào độ đanh của viền nét chữ.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def magic_color_filter(color_image, shadow_ksize=51, sharpen=True, sharpen_strength=1.3, text_boldness=1.4):
    """
    Bộ lọc 'Magic Color' siêu nét chuẩn CamScanner:
    - Triệt tiêu bóng mờ và quầng viền (halo) bằng Smooth Background Map.
    - Làm đậm và đanh nét chữ (Bold & Crisp) bằng đường cong Gamma phi tuyến tính.
    - Tăng độ tương phản cực đại bằng White & Black Levels.
    - Làm sắc nét viền chữ với Unsharp Masking tần số cao.
    """
    h, w = color_image.shape[:2]
    # Tự động tính kích thước kernel thích ứng theo độ phân giải ảnh
    ksize = max(shadow_ksize, (w // 18) | 1)
    if ksize % 2 == 0:
        ksize += 1

    # 1. Khử bóng trên kênh L (không gian LAB)
    lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # 2. Ước lượng nền với Kernel lớn + Gaussian Blur để triệt tiêu hoàn toàn quầng viền (Halo)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bg_l = cv2.morphologyEx(l_chan, cv2.MORPH_CLOSE, kernel)
    blur_k = (ksize // 2) | 1
    bg_l = cv2.GaussianBlur(bg_l, (blur_k, blur_k), 0)

    # 3. Chuẩn hóa độ sáng bằng phép chia nền
    norm_l = cv2.divide(l_chan, bg_l, scale=255).astype(np.float32)

    # 4. Kéo dãn mức sáng (Tone Curve / Levels) - Bí quyết của CamScanner:
    # Nền giấy sáng (> 218) được kéo lên 255 (trắng tinh tuyệt đối)
    # Chữ tối (< 50) được kéo đậm đà, tăng độ sâu tương phản
    black_point = 50.0
    white_point = 218.0
    stretched = np.clip((norm_l - black_point) / (white_point - black_point), 0.0, 1.0)

    # 5. Làm đậm nét chữ bằng Gamma phi tuyến tính:
    # Gamma > 1 ép các nét chữ viết tay, bút bi đậm màu, rõ nét, không bị nhạt
    gamma = max(1.0, float(text_boldness))
    darkened_l = np.power(stretched, gamma)
    norm_l_final = (darkened_l * 255.0).astype(np.uint8)

    # 6. Ghép lại kênh LAB và chuyển sang BGR
    merged_lab = cv2.merge([norm_l_final, a_chan, b_chan])
    enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    # 7. Tăng nhẹ độ bão hòa màu (Saturation) trong HSV để màu dấu mộc và mực bút tươi đẹp
    hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
    enhanced_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 8. Làm sắc nét viền chữ bằng Unsharp Masking tần số cao
    if sharpen and sharpen_strength > 0:
        enhanced_bgr = unsharp_mask(enhanced_bgr, sigma=1.0, strength=sharpen_strength)

    return enhanced_bgr
