import cv2
import numpy as np


def remove_shadow_gray(gray_image, kernel_size=51):
    """
    Triệt tiêu bóng đổ trên ảnh xám bằng phép chia hình thái học (Morphological Division).
    
    Thuật toán:
    1. Dùng phép đóng (Closing) với phần tử cấu trúc Elip kích thước lớn để ước lượng nền sáng (Background Map).
    2. Làm mịn nền bằng Gaussian Blur để triệt tiêu hoàn toàn quầng viền (Halo).
    3. Chia ảnh ban đầu cho Background Map và nhân 255 để đưa về dải [0, 255].
    """
    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Ước lượng độ sáng nền
    background = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    blur_k = (kernel_size // 2) | 1
    background = cv2.GaussianBlur(background, (blur_k, blur_k), 0)
    
    # Chuẩn hóa độ sáng bằng phép chia
    normalized = cv2.divide(gray_image, background, scale=255)
    return normalized, background
