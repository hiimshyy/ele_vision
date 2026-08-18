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
DATA_DIR = Path("data/faces")        # Enrolled face images
DB_DIR = Path("data/db")             # Database files
DEFAULT_DB = DB_DIR / "faces.db"


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

    # Quality check (blur detection)
    x1, y1, x2, y2 = int(face.x1), int(face.y1), int(face.x2), int(face.y2)
    h, w = frame.shape[:2]
    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size > 0:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        quality = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"  Quality: {quality:.1f} (Laplacian variance)")
        if quality < 30.0:
            logger.error("event=enroll_failed | reason=face too blurry | quality={q:.1f}", q=quality)
            print("  ERROR: Face too blurry. Use a clearer image.")
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

    # Duplicate check
    db = FaceDatabase(args.db)
    db.initialize()

    existing_match = db.find_match(embedding, threshold=0.7)
    if existing_match:
        print(f"  WARNING: Very similar to existing person: {existing_match.name} "
              f"(id={existing_match.person_id}, sim={existing_match.similarity:.3f})")

    # Save to database
    floor = getattr(args, "floor", None)
    row_id = db.add_face(args.id, args.name, embedding, default_floor=floor)
    total = db.count()
    persons = db.count_persons()
    db.close()

    logger.info(
        "event=enroll_success | person_id={pid} | name={name} | floor={f} | "
        "db_total={total} | db_persons={persons} | image={img}",
        pid=args.id, name=args.name, f=floor, total=total, persons=persons, img=args.image,
    )
    print(f"\n  Enrolled: {args.name} (id={args.id}){f', floor={floor}' if floor else ''}")
    print(f"  Database: {total} embeddings, {persons} persons")
    print(f"  DB path:  {args.db}")

    # Save aligned face to enrolled faces directory
    enrolled_dir = DATA_DIR / args.id
    enrolled_dir.mkdir(parents=True, exist_ok=True)
    face_filename = f"{int(time.time())}_{Path(args.image).stem}.jpg"
    face_path = enrolled_dir / face_filename
    cv2.imwrite(str(face_path), aligned)
    print(f"  Face saved: {face_path}")


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
    print(f"  {'ID':<15} {'Name':<20} {'Floor':<7} {'Emb':<5} {'Created'}")
    print(f"  {'-'*15} {'-'*20} {'-'*7} {'-'*5} {'-'*20}")

    # Group by person
    persons = {}
    for r in records:
        if r.person_id not in persons:
            persons[r.person_id] = {
                "name": r.name, "count": 0,
                "created": r.created_at, "floor": r.default_floor,
            }
        persons[r.person_id]["count"] += 1

    for pid, info in sorted(persons.items()):
        created = time.strftime("%Y-%m-%d %H:%M", time.localtime(info["created"]))
        floor_str = str(info["floor"]) if info["floor"] is not None else "-"
        print(f"  {pid:<15} {info['name']:<20} {floor_str:<7} {info['count']:<5} {created}")

    print(f"\n  Unique persons: {len(persons)}")


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

    # Initialize auto-snapshot
    snapshot = None
    if args.snapshot:
        from edge.tools.face_snapshot import FaceSnapshot
        snapshot = FaceSnapshot(
            snapshot_dir=args.snapshot_dir,
            max_per_person_per_day=args.snapshot_max,
            save_full_frame=True,
        )
        print(f"  Snapshot: enabled (dir={args.snapshot_dir}, max={args.snapshot_max}/person/day)")

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
    print(f"  Processing at {args.fps} fps")
    print(f"  Controls: [q] quit | [s] screenshot | [ESC] quit\n")

    frame_interval = 1.0 / args.fps  # Processing interval (detect+embed)
    frame_id = 0
    last_process_time = 0
    display_scale = args.scale
    show_display = not args.no_display
    stats = {"frames": 0, "detections": 0, "embeddings": 0, "recognized": 0, "unknown": 0}
    fps_timer = time.time()
    fps_frame_count = 0
    display_fps = 0.0

    # Colors (BGR)
    COLOR_RECOGNIZED = (0, 200, 0)     # Green
    COLOR_UNKNOWN = (0, 0, 220)        # Red
    COLOR_NEW = (200, 200, 0)          # Cyan (track not yet embedded)
    COLOR_STATS_BG = (40, 40, 40)      # Dark gray
    COLOR_WHITE = (255, 255, 255)

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

            # FPS throttle for processing (display runs at full camera speed)
            now = time.time()
            should_process = (now - last_process_time) >= frame_interval

            # Count display FPS
            fps_frame_count += 1
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                display_fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_timer = now

            if should_process:
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

                # Log events for newly-identified tracks
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
                    else:
                        stats["unknown"] += 1
                        logger.info(
                            "event=face_unknown | track_id={tid} | det_confidence={conf:.3f} | frame={fid}",
                            tid=track.track_id, conf=track.confidence, fid=frame_id,
                        )

                    track.event_published = True

                    # Auto-snapshot
                    if snapshot is not None:
                        aligned_snap = align_face(frame, track.landmarks)
                        snapshot.save_snapshot(
                            aligned_face=aligned_snap,
                            full_frame=frame,
                            person_id=track.identity,
                            person_name=track.identity_name,
                            confidence=track.identity_confidence if track.identity else track.confidence,
                            bbox=track.bbox,
                        )

                # Periodic stats log (every 5 seconds)
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

            # --- Draw UI overlay ---
            if show_display:
                display = frame.copy()

                # Draw tracked faces (only ACTIVE tracks with valid state)
                for track in tracker.active_tracks:
                    x1, y1, x2, y2 = [int(v) for v in track.bbox]
                    face_w = x2 - x1
                    face_h = y2 - y1

                    # Choose color based on state
                    if track.identity:
                        color = COLOR_RECOGNIZED
                        label = f"{track.identity_name} ({track.identity_confidence:.2f})"
                    elif track.embedding is not None:
                        color = COLOR_UNKNOWN
                        label = f"Unknown ({track.confidence:.2f})"
                    else:
                        color = COLOR_NEW
                        label = f"Track #{track.track_id}"

                    # Add face size to label
                    label += f" [{face_w}x{face_h}]"

                    # Draw bounding box
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                    # Draw label background
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(display, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
                    cv2.putText(display, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1, cv2.LINE_AA)

                    # Draw landmarks (small circles)
                    if track.landmarks is not None and len(track.landmarks) == 10:
                        for j in range(5):
                            lx = int(track.landmarks[j * 2])
                            ly = int(track.landmarks[j * 2 + 1])
                            if lx > 0 and ly > 0:
                                cv2.circle(display, (lx, ly), 2, (0, 255, 255), -1)

                # Draw stats bar at top
                bar_h = 32
                cv2.rectangle(display, (0, 0), (display.shape[1], bar_h), COLOR_STATS_BG, -1)
                active_count = len(tracker.active_tracks)
                stats_text = (
                    f"Display: {display_fps:.0f}fps | "
                    f"Process: {args.fps:.0f}fps | "
                    f"Tracks: {active_count} | "
                    f"Det: {detector.inference_time_ms:.0f}ms | "
                    f"Emb: {embedder.inference_time_ms:.0f}ms | "
                    f"Rec: {stats['recognized']} | "
                    f"Unk: {stats['unknown']}"
                )
                cv2.putText(display, stats_text, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

                # Scale for display if needed
                if display_scale != 1.0:
                    new_w = int(display.shape[1] * display_scale)
                    new_h = int(display.shape[0] * display_scale)
                    display = cv2.resize(display, (new_w, new_h))

                # Show
                cv2.imshow("Smart Cabin - Face Recognition", display)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # q or ESC
                    break
                elif key == ord("s"):  # screenshot
                    screenshot_path = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, display)
                    print(f"  Screenshot saved: {screenshot_path}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        db.close()

        print(f"\n  Stopped. Stats:")
        print(f"    Frames processed: {stats['frames']}")
        print(f"    Face detections:  {stats['detections']}")
        print(f"    Embeddings extracted: {stats['embeddings']}")
        print(f"    Recognized: {stats['recognized']}")
        print(f"    Unknown:    {stats['unknown']}")
        if stats["detections"] > 0:
            ratio = stats["embeddings"] / stats["detections"] * 100
            print(f"    Embedding ratio: {ratio:.1f}% (lower = tracker saving more CPU)")
        logger.info(
            "event=recognition_stopped | frames={fp} | detections={fd} | "
            "embeddings={ee} | recognized={rec} | unknown={unk}",
            fp=stats["frames"], fd=stats["detections"], ee=stats["embeddings"],
            rec=stats["recognized"], unk=stats["unknown"],
        )


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
    p_enroll.add_argument("--floor", type=int, default=None, help="Default floor (elevator)")

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
    p_run.add_argument("--min-size", type=int, default=60, help="Min face size in px (default: 60)")
    p_run.add_argument("--scale", type=float, default=1.0, help="Display scale (e.g. 0.5 for half)")
    p_run.add_argument("--no-display", action="store_true", help="Run headless (no window)")
    p_run.add_argument("--snapshot", action="store_true", help="Enable auto face snapshot")
    p_run.add_argument("--snapshot-dir", type=str, default="data/snapshots",
                        help="Snapshot output directory (default: data/snapshots)")
    p_run.add_argument("--snapshot-max", type=int, default=10,
                        help="Max snapshots per person per day (default: 10)")

    args = parser.parse_args()

    # Setup logging
    setup_logging("INFO")

    # Ensure data directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)

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
