import cv2
import numpy as np
from PIL import Image
from src.core.detector import find_document_corners
from src.core.transform import four_point_transform
from src.filters.enhancement import magic_color_filter


class DocumentScanner:
    """
    Pipeline hoàn chỉnh của hệ thống quét và cải thiện chất lượng tài liệu:
    1. Phát hiện 4 góc tài liệu bằng mô hình YOLO11n-Pose (ONNX Runtime).
    2. Nắn phẳng phối cảnh (Perspective Rectification).
    3. Bộ lọc cải thiện Magic Color (Khử bóng, Gamma đậm nét, Levels, Unsharp Masking).
    4. Xoay ảnh theo nhu cầu người dùng.
    5. Xuất ảnh và đóng gói PDF.
    """

    def __init__(self, model_path=None, conf_threshold: float = 0.25):
        self.model_path = model_path
        self.conf_threshold = conf_threshold

    def detect_corners(self, image, conf_threshold=None):
        """Tự động phát hiện 4 góc tài liệu bằng YOLO11n-Pose ONNX và trả về toạ độ cùng thông tin debug."""
        thr = self.conf_threshold if conf_threshold is None else conf_threshold
        return find_document_corners(image, model_path=self.model_path, conf_threshold=thr)

    def warp(self, image, corners):
        """Nắn thẳng tài liệu theo 4 góc chỉ định."""
        warped, _ = four_point_transform(image, corners)
        return warped

    def apply_filter(self, warped_image, shadow_ksize=51, sharpen_strength=1.3, text_boldness=1.4):
        """Áp dụng bộ lọc Magic Color lên ảnh đã nắn thẳng."""
        return magic_color_filter(
            warped_image, 
            shadow_ksize=shadow_ksize, 
            sharpen_strength=sharpen_strength, 
            text_boldness=text_boldness
        )

    def rotate_image(self, image, angle):
        """
        Xoay ảnh theo các góc 90, 180, 270 độ.
        """
        angle = angle % 360
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def process(self, image, corners=None, rotation_angle=0, shadow_ksize=51, sharpen_strength=1.3, text_boldness=1.4, conf_threshold=None):
        """
        Quy trình xử lý hoàn chỉnh từ ảnh chụp đến bản scan cuối cùng.
        """
        debug_info = None
        if corners is None:
            corners, debug_info = self.detect_corners(image, conf_threshold=conf_threshold)

        warped = self.warp(image, corners)
        
        if rotation_angle != 0:
            warped = self.rotate_image(warped, rotation_angle)

        enhanced = self.apply_filter(
            warped, 
            shadow_ksize=shadow_ksize, 
            sharpen_strength=sharpen_strength, 
            text_boldness=text_boldness
        )
        return enhanced, corners, debug_info

    @staticmethod
    def export_pdf(images, output_path):
        """
        Gom một hoặc nhiều trang scan thành một file PDF chất lượng cao.
        """
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                if len(img.shape) == 2:
                    pil_images.append(Image.fromarray(img).convert("RGB"))
                else:
                    pil_images.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            elif isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))

        if not pil_images:
            raise ValueError("Danh sách ảnh trống, không thể xuất PDF.")

        pil_images[0].save(
            output_path,
            "PDF",
            resolution=150.0,
            save_all=True,
            append_images=pil_images[1:]
        )
        return output_path
