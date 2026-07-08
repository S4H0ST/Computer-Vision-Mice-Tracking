"""
augment_dataset.py — Offline data augmentation for YOLO-Pose.

Applies controlled transformations to the training images already placed in
datasets/train/, preserving full YOLO-Pose label coherence (bounding box +
3 keypoints: snout, spine, tail).

Augmentation strategy (see README, Phase 5c):
  All classes   →  horizontal flip  +  brightness +30 %     (×3 total)
  grooming (1)  →  additionally     +  brightness −25 %     (×4 total)
  rearing  (4)  →  additionally     +  brightness −25 %     (×4 total)

The asymmetric ×4 compensates for the 2:1 class imbalance between
head_dipping (majority) and grooming/rearing (minority).

Generated files carry the suffixes _aug1 / _aug2 / _aug3 so they are
distinguishable from originals. The script is idempotent: re-running it
will not create duplicates.

Usage:
    cd scripts
    python dataset_tools/augment_dataset.py
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers.configuracion import paths

# ── Configuration ──────────────────────────────────────────────────────────────
CLASS_NAMES   = ["climbing", "grooming", "head_dipping", "horizontal", "rearing"]
EXTRA_CLASSES = {1, 4}      # grooming and rearing receive one extra augmentation

ALPHA_BRIGHT  = 1.30        # +30 % brightness
ALPHA_DARK    = 0.75        # −25 % brightness (minority classes only)

TRAIN_IMGS = paths.root / "datasets" / "train" / "images"
TRAIN_LBLS = paths.root / "datasets" / "train" / "labels"


# ── Transformations ────────────────────────────────────────────────────────────

def _flip_label_line(line: str) -> str:
    """
    Reflects one YOLO-Pose label line horizontally.

    Formula: x_new = 1.0 − x_old
    Applied to: bbox center-x (token 1) and each keypoint-x (tokens 5, 8, 11).
    Keypoint-y and visibility flags are unchanged.
    """
    t = line.strip().split()
    t[1] = f"{1.0 - float(t[1]):.6f}"          # bbox cx
    for i in range(5, len(t), 3):               # kp0_x, kp1_x, kp2_x
        t[i] = f"{1.0 - float(t[i]):.6f}"
    return " ".join(t)


def apply_flip(img: np.ndarray, label_text: str) -> tuple[np.ndarray, str]:
    """Horizontal flip on image and all label lines."""
    img_f     = cv2.flip(img, 1)
    new_label = "\n".join(_flip_label_line(l) for l in label_text.splitlines())
    return img_f, new_label


def apply_brightness(img: np.ndarray, alpha: float) -> np.ndarray:
    """Scale pixel values by alpha. Labels are unchanged."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


# ── Core logic ─────────────────────────────────────────────────────────────────

def augment_pair(img_path: Path, lbl_path: Path) -> int:
    """
    Generates up to 3 augmented variants of one image+label pair.
    Returns the number of new image files written.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    label_text = lbl_path.read_text().strip()
    if not label_text:
        return 0

    cls_id = int(label_text.splitlines()[0].split()[0])
    stem   = img_path.stem
    ext    = img_path.suffix
    saved  = 0

    def write(suffix: str, aug_img: np.ndarray, aug_lbl: str) -> None:
        nonlocal saved
        out_img = TRAIN_IMGS / (stem + suffix + ext)
        out_lbl = TRAIN_LBLS / (stem + suffix + ".txt")
        if not out_img.exists():        # idempotent: never overwrite
            cv2.imwrite(str(out_img), aug_img)
            out_lbl.write_text(aug_lbl)
            saved += 1

    # AUG 1 — horizontal flip (all classes)
    f_img, f_lbl = apply_flip(img, label_text)
    write("_aug1", f_img, f_lbl)

    # AUG 2 — brightness +30 % (all classes)
    write("_aug2", apply_brightness(img, ALPHA_BRIGHT), label_text)

    # AUG 3 — brightness −25 % (grooming and rearing only)
    if cls_id in EXTRA_CLASSES:
        write("_aug3", apply_brightness(img, ALPHA_DARK), label_text)

    return saved


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    if not TRAIN_IMGS.exists():
        print(f"[X] Directory not found: {TRAIN_IMGS}")
        print("    Place the Roboflow export in datasets/train/ first.")
        sys.exit(1)

    # Only process originals (no _aug in filename)
    originals = sorted(
        p for p in TRAIN_IMGS.iterdir()
        if p.suffix in {".jpg", ".png"} and "_aug" not in p.stem
    )
    if not originals:
        print("[X] No original images found in datasets/train/images/")
        sys.exit(1)

    print(f"[+] {len(originals)} original images found")
    print(f"[+] Strategy: ×3 all classes | ×4 grooming and rearing\n")

    stats: dict = defaultdict(lambda: {"orig": 0, "new": 0})

    for img_path in originals:
        lbl_path = TRAIN_LBLS / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        first_line = lbl_path.read_text().strip().splitlines()[0]
        cls_id     = int(first_line.split()[0])
        cls_name   = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
        stats[cls_name]["orig"] += 1
        stats[cls_name]["new"]  += augment_pair(img_path, lbl_path)

    # ── Summary ────────────────────────────────────────────────────────────────
    total_orig = total_new = 0
    print(f"{'Class':<15}  {'Orig':>5}  {'New':>5}  {'Total':>6}  {'Factor':>7}")
    print("─" * 44)
    for name in CLASS_NAMES:
        o = stats[name]["orig"]
        n = stats[name]["new"]
        total_orig += o
        total_new  += n
        factor = f"×{(o + n) / o:.1f}" if o else "—"
        print(f"{name:<15}  {o:>5}  {n:>5}  {o+n:>6}  {factor:>7}")
    print("─" * 44)
    print(f"{'TOTAL':<15}  {total_orig:>5}  {total_new:>5}  {total_orig + total_new:>6}")
    print(f"\n[OK] {total_new} new images added.")
    print(f"     Training set now contains {total_orig + total_new} images.")


if __name__ == "__main__":
    main()
