"""
Example: Full Face Recognition Pipeline

Demonstrates the complete face recognition system:
- Enroll faces into the database
- Run realtime recognition from camera/video
- Test recognition on a single image

Usage:
    # Enroll a face from image
    python examples/run_recognition.py enroll --image photo.jpg --name "Nguyen Van A" --id person_001

    # Enroll multiple images for same person (better accuracy)
    python examples/run_recognition.py enroll --image photo1.jpg --name "Nguyen Van A" --id person_001
    python examples/run_recognition.py enroll --image photo2.jpg --name "Nguyen Van A" --id person_001

    # List enrolled faces
    python examples/run_recognition.py list

    # Test recognition on single image
    python examples/run_recognition.py test --image test.jpg

    # Run realtime recognition from camera
    python examples/run_recognition.py run --url "rtsp://user:pass@ip:554/stream" --fps 5

    # Run from webcam
    python examples/run_recognition.py run --url 0 --fps 5

    # Remove enrolled face
    python examples/run_recognition.py remove --id person_001

Prerequisites:
    - det_500m.onnx in edge/plugins/face_recognition/models/
    - w600k_mbf.onnx in edge/plugins/face_recognition/models/
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from edge.core.logging_setup import setup_logging, get_logger
from edge.core.event_bus import EventBus
from edge.plugins.face_recognition.detector import FaceDetector
from edge.plugins.face_recognition.alignment import align_face
from edge.plugins.face_recognition.embedder import FaceEmbedder, cosine_similarity
from edge.plugins.face_recognition.tracker import FaceTracker, TrackState
from edge.plugins.face_recognition.database import FaceDatabase

logger = get_logger("system")

# Default paths
MODEL_DIR = Path("edge/plugins/face_recognition/models")
DEFAULT_DB = Path("faces.db")


def load_models():
    """Load detector and embedder. Exit on failure."""
    print("\n  Loading models...")

    detector = FaceDetector()
    if not detector.load():
        print("  ERROR: Failed to load detection model (det_500m.onnx)")
        sys.exit(1)
    print(f"  Detector: {detector.model_name} (backend={detector.backend})")

    embedder = FaceEmbedder()
    if not embedder.load(MODEL_DIR / "w600k_mbf.onnx"):
        print("  ERROR: Failed to load embedding model (w600k_mbf.onnx)")
        sys.exit(1)
    print(f"  Embedder: {embedder.model_name} (dim={embedder.embedding_dim})")

    return detector, embedder


# --- ENROLL Command ---


def cmd_enroll(args):
    """Enroll a face from an image into the database."""
    detector, embedder = load_models()

    # Load image
    frame = cv2.imread(args.image)
    if frame is None:
        logger.error("event=enroll_failed | reason=cannot read image | path={p}", p=args.image)
        print(f"\n  ERROR: Cannot read image: {args.image}")
        sys.exit(1)

    print(f"\n  Image: {args.image} ({frame.shape[1]}x{frame.shape[0]})")

    # Detect faces
    faces = detector.detect(frame, conf_threshold=0.5)
    if not faces:
        logger.error("event=enroll_failed | reason=no face detected | path={p}", p=args.image)
        print("  ERROR: No face detected in image.")
        sys.exit(1)

    if len(faces) > 1:
        logger.warning("event=enroll_multiple_faces | count={n} | using=largest", n=len(faces))
        print(f"  WARNING: {len(faces)} faces detected, using largest.")

    # Use largest face
    face = max(faces, key=lambda f: f.area)
    print(f"  Face: score={face.score:.3f}, size={face.width:.0f}x{face.height:.0f}")

    # Check minimum size
    if face.width < 60 or face.height < 60:
        logger.error("event=enroll_failed | reason=face too small | size={w}x{h}",
                     w=int(face.width), h=int(face.height))
        print("  ERROR: Face too small (min 60x60 px)")
        sys.exit(1)

    # Align
    aligned = align_face(frame, face.landmarks)
    if aligned is None:
        logger.error("event=enroll_failed | reason=alignment failed")
        print("  ERROR: Face alignment failed (bad landmarks)")
        sys.exit(1)

    # Extract embedding
    embedding = embedder.extract(aligned)
    if embedding is None:
        logger.error("event=enroll_failed | reason=embedding extraction failed")
        print("  ERROR: Embedding extraction failed")
        sys.exit(1)

    print(f"  Embedding: dim={embedding.shape[0]}, norm={np.linalg.norm(embedding):.4f}")

    # Save to database
    db = FaceDatabase(args.db)
    db.initialize()

    row_id = db.add_face(args.id, args.name, embedding)
    total = db.count()
    persons = db.count_persons()
    db.close()

    logger.info(
        "event=enroll_success | person_id={pid} | name={name} | "
        "db_total={total} | db_persons={persons} | image={img}",
        pid=args.id, name=args.name, total=total, persons=persons, img=args.image,
    )
    print(f"\n  Enrolled: {args.name} (id={args.id})")
    print(f"  Database: {total} embeddings, {persons} persons")
    print(f"  DB path:  {args.db}")

    # Save aligned face preview
    preview_path = Path(args.db).parent / f"enrolled_{args.id}.jpg"
    cv2.imwrite(str(preview_path), aligned)
    print(f"  Preview:  {preview_path}")


# --- LIST Command ---


def cmd_list(args):
    """List all enrolled faces."""
    db = FaceDatabase(args.db)
    if not db.initialize():
        print(f"\n  ERROR: Cannot open database: {args.db}")
        sys.exit(1)

    records = db.get_all()
    db.close()

    if not records:
        print(f"\n  Database is empty: {args.db}")
        return

    print(f"\n  Face Database: {args.db}")
    print(f"  Total: {len(records)} embeddings")
    print(f"  {'ID':<15} {'Name':<20} {'Dim':<5} {'Created'}")
    print(f"  {'-'*15} {'-'*20} {'-'*5} {'-'*20}")

    seen_persons = set()
    for r in records:
        created = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.created_at))
        dup = "" if r.person_id not in seen_persons else "(+)"
        print(f"  {r.person_id:<15} {r.name:<20} {r.embedding_dim:<5} {created} {dup}")
        seen_persons.add(r.person_id)

    print(f"\n  Unique persons: {len(seen_persons)}")


# --- REMOVE Command ---


def cmd_remove(args):
    """Remove a person from the database."""
    db = FaceDatabase(args.db)
    db.initialize()
    removed = db.remove_face(args.id)
    db.close()

    if removed > 0:
        logger.info("event=face_unenrolled | person_id={pid} | rows_removed={n}", pid=args.id, n=removed)
        print(f"\n  Removed {removed} embedding(s) for person: {args.id}")
    else:
        logger.warning("event=unenroll_not_found | person_id={pid}", pid=args.id)
        print(f"\n  Person not found: {args.id}")


# --- TEST Command ---


def cmd_test(args):
    """Test recognition on a single image."""
    detector, embedder = load_models()

    # Load database
    db = FaceDatabase(args.db)
    if not db.initialize():
        print(f"\n  ERROR: Cannot open database: {args.db}")
        sys.exit(1)

    if db.count() == 0:
        print("\n  WARNING: Database is empty. Enroll faces first.")
        print("  Run: python examples/run_recognition.py enroll --image face.jpg --name 'Name' --id person_001")

    # Load image
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"\n  ERROR: Cannot read image: {args.image}")
        sys.exit(1)

    print(f"\n  Image: {args.image} ({frame.shape[1]}x{frame.shape[0]})")
    print(f"  Database: {db.count()} embeddings, {db.count_persons()} persons")

    # Detect faces
    faces = detector.detect(frame, conf_threshold=0.5)
    print(f"  Faces detected: {len(faces)}")

    if not faces:
        print("  No faces found.")
        db.close()
        return

    # Process each face
    print(f"\n  {'#':<3} {'Score':<7} {'Size':<10} {'Match':<15} {'Similarity':<12} {'Name'}")
    print(f"  {'-'*3} {'-'*7} {'-'*10} {'-'*15} {'-'*12} {'-'*15}")

    for i, face in enumerate(faces):
        # Align
        aligned = align_face(frame, face.landmarks)
        if aligned is None:
            print(f"  {i+1:<3} {face.score:<7.3f} {face.width:.0f}x{face.height:.0f}{'':4} {'align_fail':<15}")
            continue

        # Embed
        embedding = embedder.extract(aligned)
        if embedding is None:
            print(f"  {i+1:<3} {face.score:<7.3f} {face.width:.0f}x{face.height:.0f}{'':4} {'embed_fail':<15}")
            continue

        # Match
        threshold = args.threshold
        match = db.find_match(embedding, threshold=threshold)

        if match:
            print(f"  {i+1:<3} {face.score:<7.3f} {face.width:.0f}x{face.height:.0f}{'':4} "
                  f"{match.person_id:<15} {match.similarity:<12.4f} {match.name}")
        else:
            print(f"  {i+1:<3} {face.score:<7.3f} {face.width:.0f}x{face.height:.0f}{'':4} "
                  f"{'UNKNOWN':<15} {'< ' + str(threshold):<12}")

    db.close()


# --- RUN Command (Realtime) ---


def cmd_run(args):
    """Run realtime face recognition from camera/video."""
    detector, embedder = load_models()

    # Load database
    db = FaceDatabase(args.db)
    if not db.initialize():
        print(f"\n  ERROR: Cannot open database: {args.db}")
        sys.exit(1)

    print(f"  Database: {db.count()} embeddings, {db.count_persons()} persons")

    if db.count() == 0:
        print("  WARNING: Database empty. Faces will show as UNKNOWN.")

    # Initialize tracker
    tracker = FaceTracker(
        iou_threshold=0.4,
        max_lost=int(args.fps * 3),  # 3 seconds
        max_tracks=10,
        reverify_interval=int(args.fps * 3),  # re-verify every 3s
    )

    # Open camera
    url = args.url
    if url.isdigit():
        url = int(url)
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        logger.error("event=camera_open_failed | url={u}", u=args.url)
        print(f"\n  ERROR: Cannot open: {args.url}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(
        "event=recognition_started | url={u} | resolution={w}x{h} | "
        "fps={fps} | db_persons={db} | threshold={t}",
        u=args.url, w=w, h=h, fps=args.fps,
        db=db.count_persons(), t=args.threshold,
    )
    print(f"\n  Camera opened: {w}x{h}")
    print(f"  Processing at {args.fps} fps (Ctrl+C to stop)\n")

    frame_interval = 1.0 / args.fps
    frame_id = 0
    last_process_time = 0
    stats = {"frames": 0, "detections": 0, "embeddings": 0, "recognized": 0, "unknown": 0}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("event=stream_read_failed | frame_id={fid}", fid=frame_id)
                print("  Stream ended or error. Reconnecting...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(url if not isinstance(url, int) else url)
                continue

            # FPS throttle
            now = time.time()
            if now - last_process_time < frame_interval:
                continue
            last_process_time = now
            frame_id += 1
            stats["frames"] += 1

            # Detect
            faces = detector.detect(frame, conf_threshold=0.5)

            # Filter small faces
            detections = []
            for face in faces:
                if face.width < args.min_size or face.height < args.min_size:
                    continue
                detections.append({
                    "bbox": (face.x1, face.y1, face.x2, face.y2),
                    "landmarks": face.landmarks,
                    "confidence": face.score,
                })
                stats["detections"] += 1

            # Update tracker
            tracks = tracker.update(detections, frame_id)

            # Process tracks needing embedding
            for track in tracker.get_tracks_needing_embedding(frame_id):
                aligned = align_face(frame, track.landmarks)
                if aligned is None:
                    continue

                embedding = embedder.extract(aligned)
                if embedding is None:
                    continue

                stats["embeddings"] += 1
                match = db.find_match(embedding, threshold=args.threshold)

                if match:
                    track.set_embedding(embedding, identity=match.person_id,
                                        identity_name=match.name,
                                        confidence=match.similarity, frame_id=frame_id)
                else:
                    track.set_embedding(embedding, identity=None,
                                        confidence=0.0, frame_id=frame_id)

            # Print events for newly-identified tracks
            for track in tracks:
                if track.event_published:
                    continue
                if track.state != TrackState.ACTIVE:
                    continue
                if track.embedding is None:
                    continue

                if track.identity:
                    stats["recognized"] += 1
                    logger.info(
                        "event=face_recognized | track_id={tid} | person={pid} | "
                        "name={name} | confidence={conf:.3f} | frame={fid}",
                        tid=track.track_id, pid=track.identity,
                        name=track.identity_name, conf=track.identity_confidence,
                        fid=frame_id,
                    )
                    print(f"  [{_timestamp()}] RECOGNIZED: {track.identity_name} "
                          f"(id={track.identity}, sim={track.identity_confidence:.3f}, "
                          f"track={track.track_id})")
                else:
                    stats["unknown"] += 1
                    logger.info(
                        "event=face_unknown | track_id={tid} | det_confidence={conf:.3f} | frame={fid}",
                        tid=track.track_id, conf=track.confidence, fid=frame_id,
                    )
                    print(f"  [{_timestamp()}] UNKNOWN FACE "
                          f"(det_score={track.confidence:.3f}, track={track.track_id})")

                track.event_published = True

            # Periodic stats (every 5 seconds)
            if frame_id % (int(args.fps) * 5) == 0:
                active = len([t for t in tracks if t.state == TrackState.ACTIVE])
                logger.info(
                    "event=recognition_stats | frames={fp} | detections={fd} | "
                    "embeddings={ee} | recognized={rec} | unknown={unk} | "
                    "active_tracks={at} | det_ms={dms:.1f} | emb_ms={ems:.1f}",
                    fp=stats["frames"], fd=stats["detections"], ee=stats["embeddings"],
                    rec=stats["recognized"], unk=stats["unknown"], at=active,
                    dms=detector.inference_time_ms, ems=embedder.inference_time_ms,
                )
                print(f"  [{_timestamp()}] stats: frames={stats['frames']} "
                      f"det={stats['detections']} emb={stats['embeddings']} "
                      f"rec={stats['recognized']} unk={stats['unknown']} "
                      f"active_tracks={active}")

    except KeyboardInterrupt:
        print(f"\n\n  Stopped. Stats:")
        print(f"    Frames processed: {stats['frames']}")
        print(f"    Face detections:  {stats['detections']}")
        print(f"    Embeddings extracted: {stats['embeddings']}")
        print(f"    Recognized: {stats['recognized']}")
        print(f"    Unknown:    {stats['unknown']}")
        if stats['frames'] > 0:
            ratio = stats['embeddings'] / max(stats['detections'], 1) * 100
            print(f"    Embedding ratio: {ratio:.1f}% (lower = tracker saving more CPU)")
        logger.info(
            "event=recognition_stopped | frames={fp} | detections={fd} | "
            "embeddings={ee} | recognized={rec} | unknown={unk}",
            fp=stats["frames"], fd=stats["detections"], ee=stats["embeddings"],
            rec=stats["recognized"], unk=stats["unknown"],
        )
    finally:
        cap.release()
        db.close()


# --- Helpers ---


def _timestamp():
    return time.strftime("%H:%M:%S")


# --- Main ---


def main():
    parser = argparse.ArgumentParser(
        description="Smart Cabin - Face Recognition Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enroll a face
  python examples/run_recognition.py enroll --image face.jpg --name "Alice" --id p001

  # List enrolled faces
  python examples/run_recognition.py list

  # Test on single image
  python examples/run_recognition.py test --image test.jpg

  # Run realtime from webcam
  python examples/run_recognition.py run --url 0

  # Run realtime from RTSP
  python examples/run_recognition.py run --url "rtsp://admin:pass@192.168.1.100:554/stream"
""",
    )
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB),
                        help="Database path (default: faces.db)")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Recognition threshold (default: 0.4)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Enroll
    p_enroll = subparsers.add_parser("enroll", help="Enroll a face from image")
    p_enroll.add_argument("--image", required=True, help="Image file path")
    p_enroll.add_argument("--name", required=True, help="Person display name")
    p_enroll.add_argument("--id", required=True, help="Person unique ID")

    # List
    subparsers.add_parser("list", help="List enrolled faces")

    # Remove
    p_remove = subparsers.add_parser("remove", help="Remove a person")
    p_remove.add_argument("--id", required=True, help="Person ID to remove")

    # Test
    p_test = subparsers.add_parser("test", help="Test recognition on image")
    p_test.add_argument("--image", required=True, help="Image file to test")

    # Run
    p_run = subparsers.add_parser("run", help="Run realtime recognition")
    p_run.add_argument("--url", required=True, help="Camera URL or device index (0)")
    p_run.add_argument("--fps", type=float, default=5.0, help="Processing FPS (default: 5)")
    p_run.add_argument("--min-size", type=int, default=80, help="Min face size in px")

    args = parser.parse_args()

    # Setup logging
    setup_logging("INFO")

    # Dispatch
    if args.command == "enroll":
        cmd_enroll(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "remove":
        cmd_remove(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
