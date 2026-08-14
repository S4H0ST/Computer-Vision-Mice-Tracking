"""
Offline data augmentation for YOLO Pose dataset (kpt_shape=[3,3]).

Strategy
--------
  aug1 — horizontal flip          : ALL classes
  aug2 — rotation -12°            : ALL classes
  aug3 — rotation +12°            : ALL classes
  aug4 — flip + rotation -12°     : grooming (1) + rearing (4) only

Result: ×4 for all classes, ×5 for grooming and rearing.

  climbing    42 → 168
  grooming    23 → 115
  head_dip    48 → 192
  horizontal  38 → 152
  rearing     25 → 125
  ──────────────────────
  TOTAL      176 → 752

Why geometric-only?
  The recording environment is always the same box with same lighting and
  same white mice — color/brightness changes do not reflect any real variation
  and would teach the model spurious invariances. Rotations and flips do reflect
  genuine variation (camera angle, rat direction of travel).

YOLO Pose label format (one line per object):
  class_id  cx  cy  w  h  kp0x kp0y kp0v  kp1x kp1y kp1v  kp2x kp2y kp2v
  All coords normalized [0, 1].

Horizontal flip rule:
  - bbox:      cx_new = 1 − cx   (cy, w, h unchanged)
  - keypoint:  kpx_new = 1 − kpx (kpy, kpv unchanged)
  - flip_idx=[0,1,2]: snout/spine/tail are on the body centerline — order unchanged.

Rotation rule:
  - Image rotated with cv2.warpAffine (black border fill).
  - Each bbox corner rotated → new axis-aligned bbox from min/max of corners.
  - Each keypoint rotated; if it lands outside [0,1] → visibility set to 0.
  - Object skipped (line omitted) if bbox shrinks below 5 px.

Idempotent: skips images whose stem already ends with _aug1/_aug2/_aug3/_aug4.
"""

import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR   = PROJECT_ROOT / "datasets" / "train" / "images"
LABELS_DIR   = PROJECT_ROOT / "datasets" / "train" / "labels"

MINORITY_CLASSES = {1, 4}           # rat_grooming, rat_rearing
_AUG_SUFFIXES    = ("_aug1", "_aug2", "_aug3", "_aug4")
ROTATION_ANGLE   = 12               # degrees


# ──────────────────────────────────────────────────────────────────────────────
#  Label helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_label(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text("utf-8").splitlines() if l.strip()]


def _classes_in(lines: list[str]) -> set[int]:
    return {int(l.split()[0]) for l in lines}


def _flip_line(line: str) -> str:
    """Horizontal flip of a single label line."""
    p = line.split()
    cx_f = f"{1.0 - float(p[1]):.8f}"
    out  = [p[0], cx_f, p[2], p[3], p[4]]
    kpts = p[5:]
    for i in range(0, len(kpts), 3):
        kx_f = f"{1.0 - float(kpts[i]):.8f}"
        out += [kx_f, kpts[i + 1], kpts[i + 2]]
    return " ".join(out)


def _rotation_matrix(img_w: int, img_h: int, angle_deg: float):
    cx, cy = img_w / 2.0, img_h / 2.0
    return cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)


def _rotate_point(x_n: float, y_n: float, M, img_w: int, img_h: int):
    """Rotate a normalised (x, y) through affine M; return normalised result."""
    px = x_n * img_w
    py = y_n * img_h
    px_r = M[0, 0] * px + M[0, 1] * py + M[0, 2]
    py_r = M[1, 0] * px + M[1, 1] * py + M[1, 2]
    return px_r / img_w, py_r / img_h


def _rotate_line(line: str, M, img_w: int, img_h: int) -> str | None:
    """Rotate a single label line; returns None if object left the frame."""
    p   = line.split()
    cls = p[0]
    cx_n, cy_n = float(p[1]), float(p[2])
    w_n,  h_n  = float(p[3]), float(p[4])

    # Rotate bbox corners
    cx_px, cy_px = cx_n * img_w, cy_n * img_h
    hw, hh = w_n * img_w / 2, h_n * img_h / 2
    corners = np.array([
        [cx_px - hw, cy_px - hh, 1],
        [cx_px + hw, cy_px - hh, 1],
        [cx_px + hw, cy_px + hh, 1],
        [cx_px - hw, cy_px + hh, 1],
    ], dtype=np.float32)
    rot = (M @ corners.T).T   # shape (4, 2)

    x_min = np.clip(rot[:, 0].min(), 0, img_w)
    x_max = np.clip(rot[:, 0].max(), 0, img_w)
    y_min = np.clip(rot[:, 1].min(), 0, img_h)
    y_max = np.clip(rot[:, 1].max(), 0, img_h)

    new_w_px = x_max - x_min
    new_h_px = y_max - y_min
    if new_w_px < 5 or new_h_px < 5:
        return None

    new_cx_n = ((x_min + x_max) / 2) / img_w
    new_cy_n = ((y_min + y_max) / 2) / img_h
    new_w_n  = new_w_px / img_w
    new_h_n  = new_h_px / img_h

    out = [cls,
           f"{new_cx_n:.8f}", f"{new_cy_n:.8f}",
           f"{new_w_n:.8f}",  f"{new_h_n:.8f}"]

    kpts = p[5:]
    for i in range(0, len(kpts), 3):
        kx_r, ky_r = _rotate_point(float(kpts[i]), float(kpts[i + 1]), M, img_w, img_h)
        kv = int(kpts[i + 2])
        if not (0.0 <= kx_r <= 1.0 and 0.0 <= ky_r <= 1.0):
            kv    = 0
            kx_r  = float(np.clip(kx_r, 0.0, 1.0))
            ky_r  = float(np.clip(ky_r, 0.0, 1.0))
        out += [f"{kx_r:.8f}", f"{ky_r:.8f}", str(kv)]

    return " ".join(out)


def _apply_rotation(img: np.ndarray, lines: list[str], angle_deg: float):
    """Return (rotated_img, rotated_lines) or (None, None) if no object survives."""
    h, w = img.shape[:2]
    M    = _rotation_matrix(w, h, angle_deg)
    rotated = cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
    new_lines = [r for l in lines if (r := _rotate_line(l, M, w, h)) is not None]
    if not new_lines:
        return None, None
    return rotated, new_lines


def _save(img: np.ndarray, lines: list[str], stem: str, suffix: str) -> bool:
    """Save image + label pair. Returns True if written, False if already existed."""
    img_out = IMAGES_DIR / f"{stem}{suffix}.jpg"
    lbl_out = LABELS_DIR / f"{stem}{suffix}.txt"
    if img_out.exists():
        return False
    cv2.imwrite(str(img_out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lbl_out.write_text("\n".join(lines), encoding="utf-8")
    return True


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    originals = [
        p for p in sorted(IMAGES_DIR.glob("*.jpg"))
        if not any(p.stem.endswith(s) for s in _AUG_SUFFIXES)
    ]
    print(f"[INFO] {len(originals)} imágenes originales en train/images/")
    print(f"[INFO] Ángulo de rotación: ±{ROTATION_ANGLE}°\n")

    counts = {s: 0 for s in _AUG_SUFFIXES}

    for img_path in originals:
        lbl_path = LABELS_DIR / (img_path.stem + ".txt")
        if not lbl_path.exists():
            print(f"  [SKIP] sin label: {img_path.name}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] no se puede leer: {img_path.name}")
            continue

        lines   = _read_label(lbl_path)
        classes = _classes_in(lines)
        stem    = img_path.stem

        # aug1 — horizontal flip
        flipped_img    = cv2.flip(img, 1)
        flipped_labels = [_flip_line(l) for l in lines]
        if _save(flipped_img, flipped_labels, stem, "_aug1"):
            counts["_aug1"] += 1

        # aug2 — rotation -12°
        rot_img, rot_lines = _apply_rotation(img, lines, -ROTATION_ANGLE)
        if rot_img is not None and _save(rot_img, rot_lines, stem, "_aug2"):
            counts["_aug2"] += 1

        # aug3 — rotation +12°
        rot_img, rot_lines = _apply_rotation(img, lines, +ROTATION_ANGLE)
        if rot_img is not None and _save(rot_img, rot_lines, stem, "_aug3"):
            counts["_aug3"] += 1

        # aug4 — flip + rotation -12° (minority only)
        if classes & MINORITY_CLASSES:
            rot_img, rot_lines = _apply_rotation(flipped_img, flipped_labels, -ROTATION_ANGLE)
            if rot_img is not None and _save(rot_img, rot_lines, stem, "_aug4"):
                counts["_aug4"] += 1

    total_added = sum(counts.values())
    total_now   = len(list(IMAGES_DIR.glob("*.jpg")))

    print("[OK] Augmentación completada:")
    print(f"  aug1 (flip):              {counts['_aug1']:>4} imágenes  (todas las clases)")
    print(f"  aug2 (rotación  -12°):    {counts['_aug2']:>4} imágenes  (todas las clases)")
    print(f"  aug3 (rotación  +12°):    {counts['_aug3']:>4} imágenes  (todas las clases)")
    print(f"  aug4 (flip + rot -12°):   {counts['_aug4']:>4} imágenes  (grooming + rearing)")
    print(f"  -----------------------------------------")
    print(f"  Total anadidas:           {total_added:>4}")
    print(f"  Total en train/ ahora:    {total_now:>4}")


if __name__ == "__main__":
    main()
