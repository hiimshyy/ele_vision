"""
Smart Cabin - Face Alignment

Aligns detected face to canonical 112x112 using 5-point landmarks.
Uses similarity transform (affine) based on ArcFace reference points.

Landmark order from SCRFD/InsightFace:
    0: left_eye, 1: right_eye, 2: nose, 3: mouth_left, 4: mouth_right
"""

import numpy as np
import cv2


# ArcFace standard reference landmarks for 112x112 aligned face
# Source: InsightFace alignment code (same across all ArcFace models)
ARCFACE_REF_LANDMARKS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # mouth left
    [70.7299, 92.2041],   # mouth right
], dtype=np.float32)


def estimate_similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Estimate 2D similarity transform (rotation + scale + translation).

    Uses Umeyama algorithm to find the best fit transform.

    Args:
        src: Source points (N, 2)
        dst: Destination points (N, 2)

    Returns:
        2x3 affine matrix
    """
    num = src.shape[0]
    dim = 2

    # Compute mean
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Center the points
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Compute covariance
    A = dst_demean.T @ src_demean / num

    # SVD
    U, S, Vt = np.linalg.svd(A)

    # Handle reflection
    d = np.ones(dim, dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    # Rotation
    D = np.diag(d)
    R = U @ D @ Vt

    # Scale
    src_var = src_demean.var(axis=0).sum()
    if src_var < 1e-10:
        scale = 1.0
    else:
        scale = (S * d).sum() / src_var

    # Build transform matrix
    T = np.eye(dim + 1, dtype=np.float64)
    T[:dim, :dim] = scale * R
    T[:dim, dim] = dst_mean - scale * R @ src_mean

    return T[:dim, :]  # Return 2x3 matrix


def align_face(frame: np.ndarray,
               landmarks: np.ndarray,
               output_size: int = 112) -> np.ndarray | None:
    """
    Align a face crop to canonical pose using 5-point landmarks.

    Args:
        frame: BGR image (H, W, 3) containing the face
        landmarks: Flat array of 10 floats [x0,y0,x1,y1,...,x4,y4]
                   or (5, 2) array of landmark coordinates
        output_size: Output image size (default 112x112)

    Returns:
        Aligned face image (output_size x output_size x 3) or None if invalid
    """
    # Reshape landmarks to (5, 2)
    if landmarks is None:
        return None

    lmk = np.array(landmarks, dtype=np.float32)
    if lmk.ndim == 1:
        if lmk.shape[0] != 10:
            return None
        lmk = lmk.reshape(5, 2)
    elif lmk.shape != (5, 2):
        return None

    # Check for zero landmarks (not detected)
    if np.all(lmk == 0):
        return None

    # Scale reference landmarks if output_size != 112
    ref = ARCFACE_REF_LANDMARKS.copy()
    if output_size != 112:
        ref = ref * (output_size / 112.0)

    # Estimate similarity transform
    M = estimate_similarity_transform(lmk.astype(np.float64), ref.astype(np.float64))

    # Warp
    aligned = cv2.warpAffine(
        frame, M.astype(np.float32), (output_size, output_size),
        borderValue=(0, 0, 0),
    )

    return aligned
