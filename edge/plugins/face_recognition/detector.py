"""
Smart Cabin - Face Detector Wrapper

Unified interface for face detection. Model priority:
1. SCRFD-500M (det_500m.onnx from InsightFace buffalo_s) — High accuracy, 5-point landmarks
2. YuNet via OpenCV FaceDetectorYN (fallback) — Built-in, no extra dependencies

Both produce: list of FaceInfo with bbox, score, 5-point landmarks.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from edge.core.logging_setup import get_logger

logger = get_logger("plugin")

# Default model paths (priority order)
SCRFD_MODEL = Path("edge/plugins/face_recognition/models/det_500m.onnx")
YUNET_MODEL = Path("edge/plugins/face_recognition/models/yunet.onnx")


@dataclass
class FaceInfo:
    """Detected face result."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    landmarks: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float32))

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


class FaceDetector:
    """
    Unified face detector with SCRFD (primary) and YuNet (fallback).

    Usage:
        detector = FaceDetector()
        detector.load()  # Auto-detects available models
        faces = detector.detect(frame)
    """

    def __init__(self):
        self._backend: str | None = None
        self._net: cv2.dnn.Net | None = None
        self._opencv_detector = None  # For YuNet fallback
        self._inference_time_ms: float = 0.0
        self._input_size: int = 640
        self._model_name: str = ""

    @property
    def backend(self) -> str | None:
        """Current backend: 'scrfd', 'yunet', or None."""
        return self._backend

    @property
    def model_name(self) -> str:
        """Loaded model name."""
        return self._model_name

    @property
    def inference_time_ms(self) -> float:
        """Last inference time in milliseconds."""
        return self._inference_time_ms

    def load(self,
             model_path: str | Path | None = None,
             input_size: int = 640,
             conf_threshold: float = 0.5) -> bool:
        """
        Load face detection model.

        Priority:
        1. If model_path specified → load that model
        2. Try SCRFD det_500m.onnx (primary)
        3. Fallback to YuNet (OpenCV built-in)

        Args:
            model_path: Explicit model path (ONNX). If None, auto-detect.
            input_size: Model input size (640 for SCRFD, 320 for YuNet)
            conf_threshold: Default confidence threshold

        Returns:
            True if loaded successfully
        """
        self._input_size = input_size

        # Explicit path
        if model_path is not None:
            model_path = Path(model_path)
            if "yunet" in model_path.name.lower():
                return self._load_yunet(model_path, 320, 320, conf_threshold)
            else:
                return self._load_scrfd(model_path)

        # Auto-detect: try SCRFD first, then YuNet
        if SCRFD_MODEL.exists():
            if self._load_scrfd(SCRFD_MODEL):
                return True

        if YUNET_MODEL.exists():
            if self._load_yunet(YUNET_MODEL, 320, 320, conf_threshold):
                return True

        logger.error("event=detector_load_failed | reason=no model found")
        return False

    def detect(self, frame: np.ndarray,
               conf_threshold: float = 0.5,
               nms_threshold: float = 0.4) -> list[FaceInfo]:
        """
        Detect faces in a BGR frame.

        Args:
            frame: BGR image (H, W, 3) uint8
            conf_threshold: Minimum confidence
            nms_threshold: NMS IoU threshold

        Returns:
            List of FaceInfo objects
        """
        if self._backend == "scrfd":
            return self._detect_scrfd(frame, conf_threshold, nms_threshold)
        elif self._backend == "yunet":
            return self._detect_yunet(frame, conf_threshold)
        return []

    # --- SCRFD Backend ---

    def _load_scrfd(self, model_path: Path) -> bool:
        """Load SCRFD via OpenCV DNN."""
        try:
            self._net = cv2.dnn.readNet(str(model_path))
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._backend = "scrfd"
            self._model_name = model_path.name

            # Check output count to determine if model has landmarks
            output_names = self._net.getUnconnectedOutLayersNames()
            has_kps = len(output_names) == 9

            logger.info(
                "event=detector_loaded | backend=scrfd | model={m} | input={s}x{s} | has_kps={kps}",
                m=model_path.name, s=self._input_size, kps=has_kps,
            )
            return True
        except Exception as e:
            logger.warning("event=scrfd_load_failed | error={err}", err=str(e))
            return False

    def _detect_scrfd(self, frame: np.ndarray,
                      conf_threshold: float,
                      nms_threshold: float) -> list[FaceInfo]:
        """Detect faces using SCRFD via OpenCV DNN."""
        h, w = frame.shape[:2]
        input_size = self._input_size

        # Resize with aspect ratio + padding
        scale = min(input_size / w, input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        pad_x = (input_size - new_w) // 2
        pad_y = (input_size - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # Preprocess
        blob = cv2.dnn.blobFromImage(padded, 1.0 / 128.0, (input_size, input_size),
                                     (127.5, 127.5, 127.5), swapRB=True)

        # Inference
        t_start = time.time()
        self._net.setInput(blob)
        output_names = self._net.getUnconnectedOutLayersNames()
        outputs = self._net.forward(output_names)
        self._inference_time_ms = (time.time() - t_start) * 1000

        # Decode
        has_kps = len(outputs) == 9
        faces = self._decode_scrfd(outputs, has_kps, w, h, scale, pad_x, pad_y,
                                   conf_threshold, nms_threshold)
        return faces

    def _decode_scrfd(self, outputs, has_kps: bool,
                      img_w: int, img_h: int,
                      scale: float, pad_x: int, pad_y: int,
                      conf_threshold: float, nms_threshold: float) -> list[FaceInfo]:
        """Decode SCRFD multi-stride outputs."""
        feat_strides = [8, 16, 32]
        num_anchors = 2
        input_size = self._input_size
        faces = []

        for idx, stride in enumerate(feat_strides):
            # Output layout: [scores_s8, scores_s16, scores_s32, bbox_s8, bbox_s16, bbox_s32, kps_s8, kps_s16, kps_s32]
            if has_kps:
                scores_raw = outputs[idx]
                bboxes_raw = outputs[idx + 3]
                kps_raw = outputs[idx + 6]
            else:
                scores_raw = outputs[idx]
                bboxes_raw = outputs[idx + 3]
                kps_raw = None

            scores = scores_raw.flatten()
            num_dets = len(scores)
            bboxes = bboxes_raw.reshape(num_dets, 4)

            if kps_raw is not None:
                kps = kps_raw.reshape(num_dets, 10)
            else:
                kps = None

            feat_w = input_size // stride

            for i in range(num_dets):
                score = scores[i]
                if score < conf_threshold:
                    continue

                anchor_idx = i // num_anchors
                row = anchor_idx // feat_w
                col = anchor_idx % feat_w

                cx = (col + 0.5) * stride
                cy = (row + 0.5) * stride

                dx, dy, dw, dh = bboxes[i]
                x1 = (cx - dx * stride - pad_x) / scale
                y1 = (cy - dy * stride - pad_y) / scale
                x2 = (cx + dw * stride - pad_x) / scale
                y2 = (cy + dh * stride - pad_y) / scale

                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(0, min(x2, img_w - 1))
                y2 = max(0, min(y2, img_h - 1))

                landmarks = np.zeros(10, dtype=np.float32)
                if kps is not None:
                    for j in range(5):
                        landmarks[j * 2] = (cx + kps[i][j * 2] * stride - pad_x) / scale
                        landmarks[j * 2 + 1] = (cy + kps[i][j * 2 + 1] * stride - pad_y) / scale

                faces.append(FaceInfo(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    score=score, landmarks=landmarks,
                ))

        # NMS
        if not faces:
            return []

        boxes = np.array([[f.x1, f.y1, f.x2 - f.x1, f.y2 - f.y1] for f in faces])
        scores_arr = np.array([f.score for f in faces])
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores_arr.tolist(),
                                   conf_threshold, nms_threshold)

        if len(indices) == 0:
            return []

        return [faces[i] for i in indices.flatten()]

    # --- YuNet Backend (Fallback) ---

    def _load_yunet(self, model_path: Path, w: int, h: int, conf: float) -> bool:
        """Load YuNet via OpenCV FaceDetectorYN."""
        try:
            if not model_path.exists():
                return False
            self._opencv_detector = cv2.FaceDetectorYN.create(
                str(model_path), "", (w, h), conf, 0.3, 5000
            )
            self._backend = "yunet"
            self._model_name = model_path.name
            self._input_size = w
            logger.info(
                "event=detector_loaded | backend=yunet | model={m} | input={w}x{h}",
                m=model_path.name, w=w, h=h,
            )
            return True
        except Exception as e:
            logger.warning("event=yunet_load_failed | error={err}", err=str(e))
            return False

    def _detect_yunet(self, frame: np.ndarray, conf_threshold: float) -> list[FaceInfo]:
        """Detect using OpenCV FaceDetectorYN."""
        h, w = frame.shape[:2]
        self._opencv_detector.setInputSize((w, h))
        self._opencv_detector.setScoreThreshold(conf_threshold)

        t_start = time.time()
        _, raw_detections = self._opencv_detector.detect(frame)
        self._inference_time_ms = (time.time() - t_start) * 1000

        faces = []
        if raw_detections is not None:
            for det in raw_detections:
                x, y, fw, fh = det[0], det[1], det[2], det[3]
                score = det[14]
                landmarks = np.array([
                    det[4], det[5], det[6], det[7], det[8], det[9],
                    det[10], det[11], det[12], det[13],
                ], dtype=np.float32)
                faces.append(FaceInfo(
                    x1=x, y1=y, x2=x + fw, y2=y + fh,
                    score=score, landmarks=landmarks,
                ))

        return faces
