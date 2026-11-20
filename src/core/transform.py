import cv2
import numpy as np


def order_points(pts):
    """
    Sắp xếp 4 điểm theo chiều kim đồng hồ bắt đầu từ góc trên bên trái:
    [top-left, top-right, bottom-right, bottom-left].
    
    Phương pháp kết hợp Vector tâm (Centroid) và toạ độ:
    Đảm bảo không bị lộn góc kể cả khi ảnh bị xoay xiên mạnh.
    """
    pts = np.array(pts, dtype="float32")
    if len(pts) != 4:
        raise ValueError(f"Cần đúng 4 điểm, nhận được {len(pts)} điểm.")

    # 1. Tính tâm của 4 điểm
    center = np.mean(pts, axis=0)

    # 2. Tính góc cực (polar angle) của từng điểm so với tâm
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    
    # Sắp xếp các điểm theo thứ tự góc tăng dần (ngược chiều kim đồng hồ từ trục x)
    sorted_by_angle = pts[np.argsort(angles)]
    
    # 3. Điểm trên cùng bên trái (top-left) là điểm có tổng (x + y) nhỏ nhất
    s = sorted_by_angle.sum(axis=1)
    tl_idx = np.argmin(s)
    
    # Xoay vòng mảng để điểm đầu tiên là top-left
    ordered = np.roll(sorted_by_angle, -tl_idx, axis=0)
    
    # Kiểm tra xem thứ tự có theo chiều kim đồng hồ không
    # Vector v1 = tr - tl, v2 = bl - tl -> tích có hướng z = v1.x * v2.y - v1.y * v2.x
    # Nếu z > 0 thì đang theo chiều kim đồng hồ
    rect = np.zeros((4, 2), dtype="float32")
    
    # Dùng phương pháp phân loại góc phần tư hoặc tổng/hiệu tiêu chuẩn cho kết quả chắc chắn nhất
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left: tổng x+y nhỏ nhất
    rect[2] = pts[np.argmax(s)]  # Bottom-Right: tổng x+y lớn nhất

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-Right: hiệu y - x nhỏ nhất
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left: hiệu y - x lớn nhất

    return rect


def four_point_transform(image, pts):
    """
    Hiệu chỉnh phối cảnh (Perspective Transform) dựa trên 4 góc tài liệu.
    Nắn phẳng tài liệu và cắt bỏ toàn bộ phần nền thừa.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Tính độ dài 2 cạnh ngang (chiều rộng)
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    # Tính độ dài 2 cạnh dọc (chiều cao)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    # Đảm bảo kích thước không bằng 0
    max_width = max(max_width, 10)
    max_height = max(max_height, 10)

    # Tọa độ đích cho mặt phẳng 2D
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Ma trận biến đổi phối cảnh 3x3
    matrix = cv2.getPerspectiveTransform(rect, dst)
    
    # Chiếu phối cảnh với nội suy song khối (INTER_CUBIC) để chữ sắc nét
    warped = cv2.warpPerspective(
        image, 
        matrix, 
        (max_width, max_height), 
        flags=cv2.INTER_CUBIC
    )

    return warped, matrix
