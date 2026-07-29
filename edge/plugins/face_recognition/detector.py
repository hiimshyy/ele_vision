"""
Smart Cabin - Face Detector Wrapper

Provides a unified Python interface for face detection.
Strategy:
1. Try to use C++ NCNN module (cabin_inference_py) for max performance
2. Fallback to OpenCV FaceDetectorYN (cv2.FaceDetectorYN) with ONNX model

Both produce same output: list of FaceInfo dicts with bbox, score, landmarks.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from edge.core.logging_setup import get_logger

logger = get_logger("plugin")


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
    Unified face detector: C++ NCNN (fast) or OpenCV DNN (fallback).

    Usage:
        detector = FaceDetector()
        detector.load(model_path="path/to/yunet.onnx")
        faces = detector.detect(frame)
    """

    def __init__(self):
        self._backend = None  # "ncnn" or "opencv"
        self._ncnn_detector = None
        self._opencv_detector = None
        self._inference_time_ms: float = 0.0
        self._input_size: tuple[int, int] = (320, 320)

    @property
    def backend(self) -> str | None:
        """Current backend: 'ncnn', 'opencv', or None."""
        return self._backend

    @property
    def inference_time_ms(self) -> float:
        """Last inference time in milliseconds."""
        return self._inference_time_ms

    def load(self,
             model_path: str | Path,
             input_width: int = 320,
             input_height: int = 320,
             num_threads: int = 2,
             conf_threshold: float = 0.7) -> bool:
        """
        Load face detection model.

        Args:
            model_path: Path to .onnx (OpenCV) or directory containing .param/.bin (NCNN)
            input_width: Model input width
            input_height: Model input height
            num_threads: Threads for inference
            conf_threshold: Default confidence threshold

        Returns:
            True if loaded successfully
        """
        self._input_size = (input_width, input_height)
        model_path = Path(model_path)

        # Strategy 1: Try C++ NCNN
        if self._try_load_ncnn(model_path, input_width, input_height, num_threads):
            self._backend = "ncnn"
            logger.info(
                "event=detector_loaded | backend=ncnn | input={w}x{h} | threads={t}",
                w=input_width, h=input_height, t=num_threads,
            )
            return True

        # Strategy 2: Fallback to OpenCV FaceDetectorYN
        if self._try_load_opencv(model_path, input_width, input_height, conf_threshold):
            self._backend = "opencv"
            logger.info(
                "event=detector_loaded | backend=opencv | input={w}x{h} | model={m}",
                w=input_width, h=input_height, m=model_path.name,
            )
            return True

        logger.error("event=detector_load_failed | path={p}", p=str(model_path))
        return False

    def detect(self, frame: np.ndarray,
               conf_threshold: float = 0.7,
               nms_threshold: float = 0.3) -> list[FaceInfo]:
        """
        Detect faces in a BGR frame.

        Args:
            frame: BGR image (H, W, 3) uint8
            conf_threshold: Minimum confidence
            nms_threshold: NMS IoU threshold

        Returns:
            List of FaceInfo objects
        """
        if self._backend == "ncnn":
            return self._detect_ncnn(frame, conf_threshold, nms_threshold)
        elif self._backend == "opencv":
            return self._detect_opencv(frame, conf_threshold)
        else:
            return []

    # --- NCNN Backend ---

    def _try_load_ncnn(self, model_path: Path, w: int, h: int, threads: int) -> bool:
        """Try to load C++ NCNN module."""
        try:
            import cabin_inference_py as ci
            detector = ci.FaceDetector()

            # Look for .param/.bin files
            param_path = model_path.with_suffix(".param")
            bin_path = model_path.with_suffix(".bin")

            if not param_path.exists() or not bin_path.exists():
                # Try in same directory
                parent = model_path.parent if model_path.is_file() else model_path
                param_path = parent / "yunet.param"
                bin_path = parent / "yunet.bin"

            if not param_path.exists() or not bin_path.exists():
                return False

            ok = detector.load(str(param_path), str(bin_path), w, h, threads)
            if ok:
                self._ncnn_detector = detector
            return ok

        except ImportError:
            return False
        except Exception:
            return False

    def _detect_ncnn(self, frame: np.ndarray, conf: float, nms: float) -> list[FaceInfo]:
        """Detect using C++ NCNN backend."""
        t_start = time.time()
        raw_faces = self._ncnn_detector.detect(frame, conf, nms)
        self._inference_time_ms = (time.time() - t_start) * 1000

        faces = []
        for f in raw_faces:
            landmarks = np.array(f.landmarks, dtype=np.float32)
            faces.append(FaceInfo(
                x1=f.x1, y1=f.y1, x2=f.x2, y2=f.y2,
                score=f.score, landmarks=landmarks,
            ))
        return faces

    # --- OpenCV Backend ---

    def _try_load_opencv(self, model_path: Path, w: int, h: int, conf: float) -> bool:
        """Try to load with OpenCV FaceDetectorYN."""
        try:
            # Find .onnx file
            onnx_path = None
            if model_path.suffix == ".onnx" and model_path.exists():
                onnx_path = model_path
            else:
                # Search in directory
                parent = model_path.parent if model_path.is_file() else model_path
                candidates = list(parent.glob("*.onnx"))
                if candidates:
                    onnx_path = candidates[0]

            if onnx_path is None or not onnx_path.exists():
                return False

            self._opencv_detector = cv2.FaceDetectorYN.create(
                str(onnx_path), "", (w, h), conf, 0.3, 5000
            )
            return True

        except Exception as e:
            logger.warning("event=opencv_load_failed | error={err}", err=str(e))
            return False

    def _detect_opencv(self, frame: np.ndarray, conf_threshold: float) -> list[FaceInfo]:
        """Detect using OpenCV FaceDetectorYN backend."""
        h, w = frame.shape[:2]
        self._opencv_detector.setInputSize((w, h))
        self._opencv_detector.setScoreThreshold(conf_threshold)

        t_start = time.time()
        _, raw_detections = self._opencv_detector.detect(frame)
        self._inference_time_ms = (time.time() - t_start) * 1000

        faces = []
        if raw_detections is not None:
            for det in raw_detections:
                # OpenCV FaceDetectorYN output:
                # [x, y, w, h, right_eye_x, right_eye_y, left_eye_x, left_eye_y,
                #  nose_x, nose_y, right_mouth_x, right_mouth_y, left_mouth_x, left_mouth_y, score]
                x, y, fw, fh = det[0], det[1], det[2], det[3]
                score = det[14]

                landmarks = np.array([
                    det[4], det[5],    # right eye
                    det[6], det[7],    # left eye
                    det[8], det[9],    # nose
                    det[10], det[11],  # right mouth
                    det[12], det[13],  # left mouth
                ], dtype=np.float32)

                faces.append(FaceInfo(
                    x1=x, y1=y, x2=x + fw, y2=y + fh,
                    score=score, landmarks=landmarks,
                ))

        return faces
