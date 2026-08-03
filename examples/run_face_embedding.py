"""
Example: Face Detection → Alignment → Embedding Pipeline

Demonstrates the full face embedding pipeline:
1. Detect faces with SCRFD (5-point landmarks)
2. Align faces to 112x112 canonical pose
3. Extract 512-dim embeddings with MobileFaceNet
4. Compare embeddings with cosine similarity

Usage:
    # Single image:
    python -m examples.run_face_embedding --image path/to/face.jpg

    # Compare two images:
    python -m examples.run_face_embedding --compare img1.jpg img2.jpg

    # From camera (one frame):
    python -m examples.run_face_embedding --camera

Prerequisites:
    - det_500m.onnx in edge/plugins/face_recognition/models/
    - w600k_mbf.onnx in edge/plugins/face_recognition/models/
"""

import argparse
import sys
import time

import cv2
import numpy as np

from edge.core.logging_setup import setup_logging, get_logger
from edge.plugins.face_recognition.detector import FaceDetector
from edge.plugins.face_recognition.alignment import align_face
from edge.plugins.face_recognition.embedder import FaceEmbedder, cosine_similarity

logger = get_logger("system")


def extract_face_embedding(frame: np.ndarray,
                           detector: FaceDetector,
                           embedder: FaceEmbedder) -> tuple[np.ndarray | None, float, float]:
    """
    Full pipeline: detect → align → embed.

    Returns:
        (embedding, detection_ms, embedding_ms) or (None, det_ms, 0)
    """
    # Detect
    faces = detector.detect(frame, conf_threshold=0.5)
    det_ms = detector.inference_time_ms

    if not faces:
        return None, det_ms, 0.0

    # Take the largest face
    largest = max(faces, key=lambda f: f.area)
    logger.info(
        "event=face_detected | score={s:.3f} | bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})",
        s=largest.score, x1=largest.x1, y1=largest.y1, x2=largest.x2, y2=largest.y2,
    )

    # Align
    aligned = align_face(frame, largest.landmarks)
    if aligned is None:
        logger.warning("event=alignment_failed | landmarks_zero=True")
        return None, det_ms, 0.0

    # Embed
    embedding = embedder.extract(aligned)
    emb_ms = embedder.inference_time_ms

    return embedding, det_ms, emb_ms


def run_single_image(image_path: str, detector: FaceDetector, embedder: FaceEmbedder):
    """Process single image and display embedding info."""
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error("event=image_load_failed | path={p}", p=image_path)
        sys.exit(1)

    logger.info("event=processing | image={p} | size={h}x{w}",
                p=image_path, h=frame.shape[0], w=frame.shape[1])

    embedding, det_ms, emb_ms = extract_face_embedding(frame, detector, embedder)

    if embedding is None:
        logger.warning("event=no_face_found | image={p}", p=image_path)
        print(f"\n  No face detected in: {image_path}")
        return None

    print(f"\n  Image: {image_path}")
    print(f"  Detection: {det_ms:.1f}ms")
    print(f"  Embedding: {emb_ms:.1f}ms")
    print(f"  Total:     {det_ms + emb_ms:.1f}ms")
    print(f"  Dimension: {embedding.shape[0]}")
    print(f"  L2 norm:   {np.linalg.norm(embedding):.6f}")
    print(f"  Vector[:5]: {embedding[:5]}")

    return embedding


def run_compare(image1: str, image2: str, detector: FaceDetector, embedder: FaceEmbedder):
    """Compare embeddings from two images."""
    print("\n" + "=" * 60)
    print("  FACE EMBEDDING COMPARISON")
    print("=" * 60)

    emb1 = run_single_image(image1, detector, embedder)
    emb2 = run_single_image(image2, detector, embedder)

    if emb1 is None or emb2 is None:
        print("\n  Cannot compare: face not detected in one or both images.")
        return

    sim = cosine_similarity(emb1, emb2)

    print("\n" + "-" * 60)
    print(f"  Cosine Similarity: {sim:.4f}")
    if sim > 0.4:
        print(f"  Verdict: SAME PERSON (threshold: 0.4)")
    elif sim > 0.3:
        print(f"  Verdict: UNCERTAIN (between 0.3 and 0.4)")
    else:
        print(f"  Verdict: DIFFERENT PERSON (threshold: 0.3)")
    print("-" * 60)


def run_camera(detector: FaceDetector, embedder: FaceEmbedder):
    """Capture one frame from camera and extract embedding."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("event=camera_open_failed")
        print("\n  Failed to open camera.")
        sys.exit(1)

    # Warm up camera
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        logger.error("event=camera_read_failed")
        print("\n  Failed to read frame from camera.")
        sys.exit(1)

    logger.info("event=camera_capture | size={h}x{w}", h=frame.shape[0], w=frame.shape[1])

    embedding, det_ms, emb_ms = extract_face_embedding(frame, detector, embedder)
    if embedding is None:
        print("\n  No face detected from camera.")
        return

    print(f"\n  Camera capture:")
    print(f"  Detection: {det_ms:.1f}ms")
    print(f"  Embedding: {emb_ms:.1f}ms")
    print(f"  Total:     {det_ms + emb_ms:.1f}ms")
    print(f"  Dimension: {embedding.shape[0]}")
    print(f"  L2 norm:   {np.linalg.norm(embedding):.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Face embedding extraction demo"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Single image path")
    group.add_argument("--compare", nargs=2, metavar=("IMG1", "IMG2"),
                       help="Compare two face images")
    group.add_argument("--camera", action="store_true",
                       help="Capture from camera")
    args = parser.parse_args()

    # Setup
    setup_logging("INFO")

    print("\n  Loading models...")
    t0 = time.time()

    detector = FaceDetector()
    if not detector.load():
        print("  ERROR: Failed to load detection model (det_500m.onnx)")
        sys.exit(1)
    print(f"  Detector loaded: {detector.model_name} ({(time.time()-t0)*1000:.0f}ms)")

    t1 = time.time()
    embedder = FaceEmbedder()
    if not embedder.load():
        print("  ERROR: Failed to load embedding model (w600k_mbf.onnx)")
        sys.exit(1)
    print(f"  Embedder loaded: {embedder.model_name} "
          f"(dim={embedder.embedding_dim}, {(time.time()-t1)*1000:.0f}ms)")

    # Run
    if args.image:
        run_single_image(args.image, detector, embedder)
    elif args.compare:
        run_compare(args.compare[0], args.compare[1], detector, embedder)
    elif args.camera:
        run_camera(detector, embedder)


if __name__ == "__main__":
    main()
