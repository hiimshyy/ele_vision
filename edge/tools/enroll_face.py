"""
Smart Cabin - Face Enrollment Tool

CLI tool to register faces directly on the Edge device.
Supports image file, camera capture, and batch folder enrollment.

Modes:
    image   - Enroll from image file(s)
    camera  - Live preview, press SPACE to capture
    batch   - Enroll all images in a folder (filename = person_id)

Validations:
    - Exactly 1 face detected (reject 0 or multiple)
    - Face size >= min_size (reject too small)
    - Face quality >= min_quality (reject blurry via Laplacian variance)
    - Duplicate detection (warn if embedding already close to existing)

Usage:
    # From image
    python -m edge.tools.enroll_face image --path face.jpg --id p001 --name "Alice"

    # Multiple images for same person
    python -m edge.tools.enroll_face image --path a.jpg b.jpg c.jpg --id p001 --name "Alice"

    # From camera (live preview)
    python -m edge.tools.enroll_face camera --id p001 --name "Alice" --url 0

    # Batch folder (filename as person_id: p001_alice.jpg → id=p001, name=alice)
    python -m edge.tools.enroll_face batch --folder ./faces/ --name-from-file

    # List enrolled faces
    python -m edge.tools.enroll_face list

    # Remove person
    python -m edge.tools.enroll_face remove --id p001
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from edge.core.logging_setup import setup_logging, get_logger
from edge.plugins.face_recognition.detector import FaceDetector
from edge.plugins.face_recognition.alignment import align_face
from edge.plugins.face_recognition.embedder import FaceEmbedder, cosine_similarity
from edge.plugins.face_recognition.database import FaceDatabase

logger = get_logger("system")

# Defaults
MODEL_DIR = Path("edge/plugins/face_recognition/models")
DEFAULT_DB = Path("data/db/faces.db")
DATA_FACES_DIR = Path("data/faces")


class EnrollmentValidator:
    """Validates face images before enrollment."""

    def __init__(self, min_face_size: int = 60, min_quality: float = 30.0):
        self.min_face_size = min_face_size
        self.min_quality = min_quality

    def validate(self, frame: np.ndarray, faces: list, detector_name: str = "") -> tuple[bool, str]:
        """
        Validate frame for enrollment.

        Returns:
            (is_valid, error_message)
        """
        if len(faces) == 0:
            return False, "No face detected"
        if len(faces) > 1:
            return False, f"Multiple faces detected ({len(faces)}). Use image with only 1 face."

        face = faces[0]

        # Size check
        if face.width < self.min_face_size or face.height < self.min_face_size:
            return False, (f"Face too small ({int(face.width)}x{int(face.height)}px). "
                           f"Minimum: {self.min_face_size}x{self.min_face_size}px")

        # Quality check (Laplacian variance = blur detection)
        x1, y1, x2, y2 = int(face.x1), int(face.y1), int(face.x2), int(face.y2)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return False, "Invalid face region"

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        quality = cv2.Laplacian(gray, cv2.CV_64F).var()

        if quality < self.min_quality:
            return False, (f"Face too blurry (quality={quality:.1f}, "
                           f"minimum={self.min_quality}). Hold still or improve lighting.")

        return True, ""

    def check_duplicate(self, embedding: np.ndarray, database: FaceDatabase,
                        threshold: float = 0.7) -> tuple[bool, str]:
        """
        Check if embedding is very similar to an existing person.

        Returns:
            (is_duplicate, warning_message)
        """
        match = database.find_match(embedding, threshold=threshold)
        if match is not None:
            return True, (f"Very similar to existing person: {match.name} "
                          f"(id={match.person_id}, similarity={match.similarity:.3f})")
        return False, ""


class FaceEnroller:
    """Handles the enrollment workflow."""

    def __init__(self, db_path: Path = DEFAULT_DB, min_face_size: int = 60,
                 min_quality: float = 30.0):
        self.db_path = db_path
        self.validator = EnrollmentValidator(min_face_size=min_face_size,
                                            min_quality=min_quality)
        self._detector: FaceDetector | None = None
        self._embedder: FaceEmbedder | None = None
        self._database: FaceDatabase | None = None

    def initialize(self) -> bool:
        """Load models and database."""
        print("  Loading models...")

        self._detector = FaceDetector()
        if not self._detector.load():
            print("  ERROR: Failed to load detection model")
            return False
        print(f"  Detector: {self._detector.model_name}")

        self._embedder = FaceEmbedder()
        if not self._embedder.load(MODEL_DIR / "w600k_mbf.onnx"):
            print("  ERROR: Failed to load embedding model")
            return False
        print(f"  Embedder: {self._embedder.model_name} (dim={self._embedder.embedding_dim})")

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._database = FaceDatabase(self.db_path)
        if not self._database.initialize():
            print("  ERROR: Failed to initialize database")
            return False
        print(f"  Database: {self._database.count()} embeddings, "
              f"{self._database.count_persons()} persons")

        return True

    def enroll_image(self, image_path: str, person_id: str, name: str,
                     skip_duplicate_check: bool = False) -> bool:
        """
        Enroll a face from an image file.

        Returns:
            True if enrollment succeeded
        """
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"    ERROR: Cannot read image: {image_path}")
            return False

        return self._enroll_frame(frame, person_id, name, source=image_path,
                                  skip_duplicate_check=skip_duplicate_check)

    def enroll_from_camera(self, person_id: str, name: str, url: str | int = 0,
                           num_captures: int = 3) -> bool:
        """
        Enroll face from camera with live preview.

        Shows preview window. User presses SPACE to capture (multiple times).
        Press 'q' or ESC to finish.

        Returns:
            True if at least one capture was enrolled
        """
        if str(url).isdigit():
            url = int(url)
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"    ERROR: Cannot open camera: {url}")
            return False

        print(f"\n  Camera preview opened.")
        print(f"  Press SPACE to capture ({num_captures} recommended)")
        print(f"  Press 'q' or ESC when done.\n")

        enrolled = 0
        capture_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Detect for preview
            faces = self._detector.detect(frame, conf_threshold=0.5)
            display = frame.copy()

            # Draw detected faces
            for face in faces:
                x1, y1, x2, y2 = int(face.x1), int(face.y1), int(face.x2), int(face.y2)
                # Validation color
                valid, _ = self.validator.validate(frame, [face])
                color = (0, 255, 0) if valid else (0, 0, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                label = f"{int(face.width)}x{int(face.height)} q={self._get_quality(frame, face):.0f}"
                cv2.putText(display, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Status bar
            status = (f"Enrolled: {enrolled}/{num_captures} | "
                      f"Faces: {len(faces)} | "
                      f"[SPACE] capture | [q] done")
            cv2.rectangle(display, (0, 0), (display.shape[1], 30), (40, 40, 40), -1)
            cv2.putText(display, status, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Face Enrollment", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):  # SPACE to capture
                capture_num += 1
                print(f"    Capture #{capture_num}...", end=" ")
                success = self._enroll_frame(frame, person_id, name,
                                             source=f"camera_capture_{capture_num}",
                                             skip_duplicate_check=(enrolled > 0))
                if success:
                    enrolled += 1
                if enrolled >= num_captures:
                    print(f"\n  Reached {num_captures} captures. Press 'q' to finish or continue.")

            elif key == ord("q") or key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        if enrolled > 0:
            print(f"\n  Enrolled {enrolled} image(s) for: {name} (id={person_id})")
            logger.info(
                "event=camera_enroll_done | person_id={pid} | name={n} | captures={c}",
                pid=person_id, n=name, c=enrolled,
            )
        else:
            print(f"\n  No images enrolled.")

        return enrolled > 0

    def enroll_batch(self, folder: str, name_from_file: bool = False,
                     default_id: str = "", default_name: str = "") -> int:
        """
        Enroll all images from a folder.

        If name_from_file=True: filename format is '{person_id}_{name}.jpg'
        Otherwise: all images enrolled under default_id/default_name.

        Returns:
            Number of successfully enrolled images
        """
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"    ERROR: Folder not found: {folder}")
            return 0

        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [f for f in sorted(folder_path.iterdir())
                  if f.suffix.lower() in image_exts]

        if not images:
            print(f"    ERROR: No images found in: {folder}")
            return 0

        print(f"  Found {len(images)} image(s) in {folder}")
        enrolled = 0

        for img_path in images:
            # Parse person_id and name from filename
            if name_from_file:
                stem = img_path.stem
                parts = stem.split("_", 1)
                person_id = parts[0]
                name = parts[1].replace("_", " ") if len(parts) > 1 else person_id
            else:
                person_id = default_id
                name = default_name

            if not person_id:
                print(f"    SKIP: Cannot determine person_id for {img_path.name}")
                continue

            print(f"    {img_path.name} → id={person_id}, name={name}...", end=" ")
            success = self.enroll_image(str(img_path), person_id, name,
                                        skip_duplicate_check=True)
            if success:
                enrolled += 1

        print(f"\n  Batch complete: {enrolled}/{len(images)} enrolled")
        return enrolled

    def _enroll_frame(self, frame: np.ndarray, person_id: str, name: str,
                      source: str = "", skip_duplicate_check: bool = False) -> bool:
        """Internal: validate, embed, and store a single frame."""
        # Detect
        faces = self._detector.detect(frame, conf_threshold=0.5)

        # Validate
        valid, error = self.validator.validate(frame, faces)
        if not valid:
            print(f"REJECTED: {error}")
            return False

        face = faces[0]

        # Align
        aligned = align_face(frame, face.landmarks)
        if aligned is None:
            print("REJECTED: Alignment failed")
            return False

        # Embed
        embedding = self._embedder.extract(aligned)
        if embedding is None:
            print("REJECTED: Embedding extraction failed")
            return False

        # Duplicate check
        if not skip_duplicate_check:
            is_dup, dup_msg = self.validator.check_duplicate(embedding, self._database)
            if is_dup:
                print(f"WARNING: {dup_msg}")
                # Still enroll but warn

        # Save to database
        self._database.add_face(person_id, name, embedding)

        # Save aligned face to data/faces/{person_id}/
        face_dir = DATA_FACES_DIR / person_id
        face_dir.mkdir(parents=True, exist_ok=True)
        face_filename = f"{int(time.time())}_{Path(source).stem if source else 'capture'}.png"
        cv2.imwrite(str(face_dir / face_filename), aligned)

        print(f"OK (size={int(face.width)}x{int(face.height)}, "
              f"quality={self._get_quality(frame, face):.0f})")

        logger.info(
            "event=face_enrolled | person_id={pid} | name={n} | "
            "source={src} | face_size={w}x{h}",
            pid=person_id, n=name, src=source,
            w=int(face.width), h=int(face.height),
        )
        return True

    def _get_quality(self, frame: np.ndarray, face) -> float:
        """Get Laplacian variance (blur metric) for a face."""
        x1, y1, x2, y2 = int(face.x1), int(face.y1), int(face.x2), int(face.y2)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def list_faces(self) -> None:
        """List all enrolled faces."""
        records = self._database.get_all()
        if not records:
            print("\n  Database is empty.")
            return

        print(f"\n  Face Database: {self.db_path}")
        print(f"  Total: {len(records)} embeddings, {self._database.count_persons()} persons\n")
        print(f"  {'ID':<15} {'Name':<20} {'Embeddings':<12} {'Created'}")
        print(f"  {'-'*15} {'-'*20} {'-'*12} {'-'*20}")

        # Group by person
        persons = {}
        for r in records:
            if r.person_id not in persons:
                persons[r.person_id] = {"name": r.name, "count": 0, "created": r.created_at}
            persons[r.person_id]["count"] += 1

        for pid, info in sorted(persons.items()):
            created = time.strftime("%Y-%m-%d %H:%M", time.localtime(info["created"]))
            print(f"  {pid:<15} {info['name']:<20} {info['count']:<12} {created}")

    def remove_face(self, person_id: str) -> None:
        """Remove a person from database."""
        removed = self._database.remove_face(person_id)
        if removed > 0:
            print(f"\n  Removed {removed} embedding(s) for: {person_id}")
            # Also remove saved face images
            face_dir = DATA_FACES_DIR / person_id
            if face_dir.exists():
                import shutil
                shutil.rmtree(face_dir)
                print(f"  Removed face images: {face_dir}")
            logger.info("event=face_unenrolled | person_id={pid} | removed={n}",
                        pid=person_id, n=removed)
        else:
            print(f"\n  Person not found: {person_id}")

    def close(self) -> None:
        """Close database."""
        if self._database:
            self._database.close()


# --- Main ---


def main():
    parser = argparse.ArgumentParser(
        description="Smart Cabin - Face Enrollment Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m edge.tools.enroll_face image --path face.jpg --id p001 --name "Alice"
  python -m edge.tools.enroll_face image --path a.jpg b.jpg --id p001 --name "Alice"
  python -m edge.tools.enroll_face camera --id p001 --name "Alice" --url 0
  python -m edge.tools.enroll_face batch --folder ./faces/ --name-from-file
  python -m edge.tools.enroll_face list
  python -m edge.tools.enroll_face remove --id p001
""",
    )
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB),
                        help="Database path")
    parser.add_argument("--min-size", type=int, default=60,
                        help="Min face size (default: 60)")
    parser.add_argument("--min-quality", type=float, default=30.0,
                        help="Min quality / Laplacian variance (default: 30)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Image mode
    p_image = subparsers.add_parser("image", help="Enroll from image file(s)")
    p_image.add_argument("--path", nargs="+", required=True, help="Image path(s)")
    p_image.add_argument("--id", required=True, help="Person ID")
    p_image.add_argument("--name", required=True, help="Person name")

    # Camera mode
    p_camera = subparsers.add_parser("camera", help="Enroll from camera (live preview)")
    p_camera.add_argument("--id", required=True, help="Person ID")
    p_camera.add_argument("--name", required=True, help="Person name")
    p_camera.add_argument("--url", default="0", help="Camera URL/index (default: 0)")
    p_camera.add_argument("--captures", type=int, default=3,
                          help="Number of captures recommended (default: 3)")

    # Batch mode
    p_batch = subparsers.add_parser("batch", help="Enroll all images in folder")
    p_batch.add_argument("--folder", required=True, help="Folder with images")
    p_batch.add_argument("--name-from-file", action="store_true",
                         help="Parse person_id_name from filename")
    p_batch.add_argument("--id", default="", help="Default person ID (if not from file)")
    p_batch.add_argument("--name", default="", help="Default name (if not from file)")

    # List
    subparsers.add_parser("list", help="List enrolled faces")

    # Remove
    p_remove = subparsers.add_parser("remove", help="Remove person")
    p_remove.add_argument("--id", required=True, help="Person ID to remove")

    args = parser.parse_args()
    setup_logging("INFO")

    print("\n  Smart Cabin - Face Enrollment Tool")
    print("  " + "=" * 40)

    try:
        if args.command == "list":
            # List and remove don't need models - only database
            enroller = FaceEnroller(db_path=Path(args.db), min_face_size=args.min_size,
                                    min_quality=args.min_quality)
            enroller._database = FaceDatabase(Path(args.db))
            if not enroller._database.initialize():
                print("  ERROR: Cannot open database")
                sys.exit(1)
            enroller.list_faces()
            enroller.close()
            return

        elif args.command == "remove":
            enroller = FaceEnroller(db_path=Path(args.db), min_face_size=args.min_size,
                                    min_quality=args.min_quality)
            enroller._database = FaceDatabase(Path(args.db))
            if not enroller._database.initialize():
                print("  ERROR: Cannot open database")
                sys.exit(1)
            enroller.remove_face(args.id)
            enroller.close()
            return

        # Other commands need full initialization (models + db)
        enroller = FaceEnroller(
            db_path=Path(args.db),
            min_face_size=args.min_size,
            min_quality=args.min_quality,
        )

        if not enroller.initialize():
            sys.exit(1)

        if args.command == "image":
            success_count = 0
            for img_path in args.path:
                print(f"\n    [{img_path}]...", end=" ")
                if enroller.enroll_image(img_path, args.id, args.name):
                    success_count += 1
            print(f"\n  Result: {success_count}/{len(args.path)} enrolled for {args.name}")

        elif args.command == "camera":
            enroller.enroll_from_camera(args.id, args.name, url=args.url,
                                        num_captures=args.captures)

        elif args.command == "batch":
            enroller.enroll_batch(args.folder, name_from_file=args.name_from_file,
                                  default_id=args.id, default_name=args.name)

        enroller.close()

    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
