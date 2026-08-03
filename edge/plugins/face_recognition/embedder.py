"""
Smart Cabin - Face Embedder (MobileFaceNet / w600k_mbf)

Extracts 512-dimensional face embeddings using OpenCV DNN backend.
Model: w600k_mbf.onnx from InsightFace buffalo_s pack.

Usage:
    embedder = FaceEmbedder()
    embedder.load()  # Load ONNX model
    embedding = embedder.extract(aligned_face_112x112)
    similarity = cosine_similarity(emb_a, emb_b)
"""

import time
import threading
from pathlib import Path

import cv2
import numpy as np

from edge.core.logging_setup import get_logger

logger = get_logger("plugin")

# Default model path
EMBEDDING_MODEL = Path("edge/plugins/face_recognition/models/w600k_mbf.onnx")


def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Args:
        emb_a: First embedding (N,)
        emb_b: Second embedding (N,)

    Returns:
        Similarity score in [-1.0, 1.0]. Higher = more similar.
        Typically >0.4 = same person, <0.3 = different person.
    """
    emb_a = emb_a.flatten().astype(np.float64)
    emb_b = emb_b.flatten().astype(np.float64)

    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))


def cosine_similarity_batch(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query and multiple gallery embeddings.

    Args:
        query: Query embedding (D,)
        gallery: Gallery embeddings (N, D)

    Returns:
        Array of similarity scores (N,)
    """
    query = query.flatten().astype(np.float64)
    gallery = gallery.astype(np.float64)

    norm_q = np.linalg.norm(query)
    if norm_q < 1e-10:
        return np.zeros(gallery.shape[0])

    norms_g = np.linalg.norm(gallery, axis=1)
    # Avoid division by zero
    norms_g = np.maximum(norms_g, 1e-10)

    return (gallery @ query) / (norms_g * norm_q)


class FaceEmbedder:
    """
    Face embedding extractor using MobileFaceNet (w600k_mbf.onnx).

    Extracts 512-dim L2-normalized embeddings from aligned 112x112 faces.
    Uses OpenCV DNN backend for CPU inference (compatible with ARM64).

    Usage:
        embedder = FaceEmbedder()
        embedder.load()
        embedding = embedder.extract(aligned_face)
    """

    def __init__(self):
        self._net: cv2.dnn.Net | None = None
        self._model_name: str = ""
        self._inference_time_ms: float = 0.0
        self._embedding_dim: int = 512
        self._input_size: int = 112
        self._lock = threading.Lock()  # OpenCV DNN is NOT thread-safe

    @property
    def model_name(self) -> str:
        """Loaded model name."""
        return self._model_name

    @property
    def inference_time_ms(self) -> float:
        """Last inference time in milliseconds."""
        return self._inference_time_ms

    @property
    def embedding_dim(self) -> int:
        """Embedding vector dimension."""
        return self._embedding_dim

    @property
    def input_size(self) -> int:
        """Required input face size (112x112)."""
        return self._input_size

    @property
    def is_loaded(self) -> bool:
        """Whether model is loaded."""
        return self._net is not None

    def load(self, model_path: str | Path | None = None) -> bool:
        """
        Load embedding model (ONNX via OpenCV DNN).

        Args:
            model_path: Path to .onnx model. If None, uses default w600k_mbf.onnx.

        Returns:
            True if loaded successfully
        """
        if model_path is None:
            model_path = EMBEDDING_MODEL
        model_path = Path(model_path)

        if not model_path.exists():
            logger.error(
                "event=embedder_load_failed | reason=model not found | path={p}",
                p=str(model_path),
            )
            return False

        try:
            self._net = cv2.dnn.readNet(str(model_path))
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._model_name = model_path.name

            # Probe output dimension with dummy input
            dummy = np.zeros((1, 3, self._input_size, self._input_size), dtype=np.float32)
            self._net.setInput(dummy)
            out = self._net.forward()
            self._embedding_dim = out.shape[-1]

            logger.info(
                "event=embedder_loaded | model={m} | input={s}x{s} | output_dim={d}",
                m=model_path.name, s=self._input_size, d=self._embedding_dim,
            )
            return True
        except Exception as e:
            logger.error(
                "event=embedder_load_failed | error={err}", err=str(e),
            )
            self._net = None
            return False

    def extract(self, aligned_face: np.ndarray) -> np.ndarray | None:
        """
        Extract embedding from an aligned face image.

        Args:
            aligned_face: BGR image, 112x112x3 uint8 (output of align_face)

        Returns:
            L2-normalized embedding vector (embedding_dim,) or None on failure
        """
        if self._net is None:
            logger.warning("event=embedder_not_loaded")
            return None

        if aligned_face is None:
            return None

        # Validate input
        if aligned_face.ndim != 3 or aligned_face.shape[2] != 3:
            logger.warning("event=embedder_invalid_input | shape={s}", s=aligned_face.shape)
            return None

        # Resize if not 112x112
        h, w = aligned_face.shape[:2]
        if h != self._input_size or w != self._input_size:
            aligned_face = cv2.resize(aligned_face, (self._input_size, self._input_size))

        # Preprocess: BGR->RGB, normalize to [-1, 1], create blob
        # Formula: (pixel - 127.5) / 127.5
        blob = cv2.dnn.blobFromImage(
            aligned_face,
            scalefactor=1.0 / 127.5,
            size=(self._input_size, self._input_size),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,  # BGR -> RGB
        )

        # Inference (locked - cv2.dnn.Net is NOT thread-safe)
        with self._lock:
            t_start = time.time()
            self._net.setInput(blob)
            output = self._net.forward()
            self._inference_time_ms = (time.time() - t_start) * 1000

        # L2 normalize
        embedding = output.flatten()
        norm = np.linalg.norm(embedding)
        if norm < 1e-10:
            return None
        embedding = embedding / norm

        return embedding.astype(np.float32)

    def extract_batch(self, aligned_faces: list[np.ndarray]) -> list[np.ndarray | None]:
        """
        Extract embeddings from multiple aligned faces.

        Args:
            aligned_faces: List of aligned BGR images (112x112x3 uint8)

        Returns:
            List of embedding vectors (or None for failed extractions)
        """
        return [self.extract(face) for face in aligned_faces]
