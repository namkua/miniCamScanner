import cv2
import numpy as np
from PIL import Image
from src.core.detector import find_document_corners
from src.core.transform import four_point_transform
from src.filters.enhancement import magic_color_filter


class DocumentScanner:
    """
    Pipeline hoàn chỉnh của hệ thống quét và cải thiện chất lượng tài liệu:
    1. Phát hiện 4 góc tài liệu (OpenCV Cổ điển / AI Segmentation).
    2. Nắn phẳng phối cảnh (Perspective Rectification).
    3. Bộ lọc cải thiện Magic Color (Khử bóng, Gamma đậm nét, Levels, Unsharp Masking).
    4. Xoay ảnh theo nhu cầu người dùng.
    5. Xuất ảnh và đóng gói PDF.
    """

    def __init__(self):
        pass

    def detect_corners(self, image, method="classical"):
        """Tự động phát hiện 4 góc tài liệu và trả về toạ độ cùng ảnh debug."""
        return find_document_corners(image, method=method)

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

    def process(self, image, corners=None, rotation_angle=0, shadow_ksize=51, sharpen_strength=1.3, text_boldness=1.4, detect_method="classical"):
        """
        Quy trình xử lý hoàn chỉnh từ ảnh chụp đến bản scan cuối cùng.
        """
        debug_info = None
        if corners is None:
            corners, debug_info = self.detect_corners(image, method=detect_method)

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
