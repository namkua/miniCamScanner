import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "yolo11n_doc_corner.onnx"
CORNER_LABELS = ("TL", "TR", "BR", "BL")


class YoloCornerDetector:
    """
    Bộ nhận diện 4 góc tài liệu bằng mô hình YOLO11n-Pose (ONNX Runtime).
    Trả về trực tiếp 4 tọa độ keypoint theo chiều kim đồng hồ: [TL, TR, BR, BL].
    """

    def __init__(self, model_path=None, num_threads: int = 4):
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file model YOLO ONNX tại: {self.model_path}"
            )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = max(1, min(num_threads, os.cpu_count() or 4))

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        in_shape = input_meta.shape
        self.img_size = in_shape[2] if len(in_shape) >= 3 and isinstance(in_shape[2], int) else 640

    def _letterbox(self, img_bgr: np.ndarray):
        h, w = img_bgr.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

        pad_w = (self.img_size - nw) / 2.0
        pad_h = (self.img_size - nh) / 2.0
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

        canvas = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return canvas, scale, left, top

    def detect(self, image: np.ndarray, conf_threshold: float = 0.25):
        orig_h, orig_w = image.shape[:2]
        canvas, scale, pad_x, pad_y = self._letterbox(image)
        blob = cv2.dnn.blobFromImage(canvas, scalefactor=1.0 / 255.0, swapRB=True, crop=False)

        outputs = self.session.run(None, {self.input_name: blob})
        out = np.squeeze(outputs[0], axis=0)
        if out.shape[0] < out.shape[1]:
            out = out.T  # Shape: (N_anchors, 4 + 1 + 4*3 = 17)

        scores = out[:, 4]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        bbox_px = None
        if best_score >= conf_threshold:
            cx, cy, bw, bh = out[best_idx, :4]
            x1 = float(np.clip((cx - bw / 2.0 - pad_x) / scale, 0.0, orig_w - 1))
            y1 = float(np.clip((cy - bh / 2.0 - pad_y) / scale, 0.0, orig_h - 1))
            x2 = float(np.clip((cx + bw / 2.0 - pad_x) / scale, 0.0, orig_w - 1))
            y2 = float(np.clip((cy + bh / 2.0 - pad_y) / scale, 0.0, orig_h - 1))
            bbox_px = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))

            kpts_raw = out[best_idx, 5:17].reshape(4, 3)[:, :2]
            kpts_px = (kpts_raw - np.array([pad_x, pad_y], dtype=np.float32)) / scale
            kpts_px[:, 0] = np.clip(kpts_px[:, 0], 0.0, orig_w - 1)
            kpts_px[:, 1] = np.clip(kpts_px[:, 1], 0.0, orig_h - 1)

            corners = kpts_px.astype(np.float32)
            detected = True
            is_fallback = False
            status_message = (
                f"YOLO11n-Pose (`{self.model_path.name}`) đã phát hiện 4 góc tài liệu "
                f"(độ tin cậy: {best_score * 100:.1f}%)."
            )
        else:
            mx = max(1, int(orig_w * 0.02))
            my = max(1, int(orig_h * 0.02))
            corners = np.array(
                [
                    [mx, my],
                    [orig_w - 1 - mx, my],
                    [orig_w - 1 - mx, orig_h - 1 - my],
                    [mx, orig_h - 1 - my],
                ],
                dtype=np.float32,
            )
            detected = False
            is_fallback = True
            status_message = (
                f"YOLO11n-Pose không phát hiện tài liệu vượt ngưỡng {conf_threshold:.2f} "
                f"(max conf: {best_score * 100:.1f}%). Đang tạm dùng khung mặc định."
            )

        detection_img = image.copy()
        line_thickness = max(2, int(round(min(orig_h, orig_w) / 250)))
        point_radius = max(5, int(round(min(orig_h, orig_w) / 100)))
        font_scale = max(0.6, min(orig_h, orig_w) / 900.0)

        if bbox_px is not None:
            cv2.rectangle(
                detection_img,
                (bbox_px[0], bbox_px[1]),
                (bbox_px[2], bbox_px[3]),
                (255, 200, 0),
                line_thickness,
            )

        pts_int = np.round(corners).astype(np.int32)
        cv2.polylines(detection_img, [pts_int], isClosed=True, color=(0, 255, 0), thickness=line_thickness)
        for label, pt in zip(CORNER_LABELS, pts_int):
            cv2.circle(detection_img, tuple(pt), point_radius, (0, 0, 255), -1)
            cv2.putText(
                detection_img,
                label,
                (int(pt[0]) + point_radius + 4, int(pt[1]) - point_radius),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 255),
                max(2, line_thickness),
            )

        debug_info = {
            "detected": detected,
            "confidence": best_score,
            "is_fallback": is_fallback,
            "model_path": str(self.model_path),
            "bbox": bbox_px,
            "letterbox_img": canvas,
            "detection_img": detection_img,
            "status_message": status_message,
        }
        return corners, debug_info


@lru_cache(maxsize=2)
def _get_cached_detector(model_path_str: str) -> YoloCornerDetector:
    return YoloCornerDetector(model_path=model_path_str)


def find_document_corners(image: np.ndarray, model_path=None, conf_threshold: float = 0.25):
    """
    Tự động phát hiện 4 góc tài liệu bằng mô hình YOLO11n-Pose ONNX.

    Args:
        image: Ảnh BGR đầu vào.
        model_path: Đường dẫn tới file .onnx (mặc định `models/yolo11n_doc_corner.onnx`).
        conf_threshold: Ngưỡng độ tin cậy để xác nhận có tài liệu trong ảnh.

    Returns:
        corners: np.ndarray shape (4, 2) theo tọa độ ảnh gốc [TL, TR, BR, BL].
        debug_info: dict chứa thông tin dự đoán và ảnh trung gian của YOLO11n-Pose.
    """
    resolved_path = str(Path(model_path).resolve() if model_path is not None else DEFAULT_MODEL_PATH.resolve())
    detector = _get_cached_detector(resolved_path)
    return detector.detect(image, conf_threshold=conf_threshold)
