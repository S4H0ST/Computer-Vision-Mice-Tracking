"""
Reconstruye el dataset completo combinando:
  - 301 imagenes de media_original/roboflow_testRata1 (auto-etiquetadas)
  - 113 imagenes de Roboflow ya existentes en datasets/train+valid
Total esperado: ~414 pares -> 80% train / 20% valid
"""
import sys
import shutil
import random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).parent.parent
SRC_DIR  = ROOT / "media_original" / "roboflow_testRata1"
DATASET  = ROOT / "datasets"

# ── 1. Auto-etiquetar las 301 imágenes ──────────────────────────────────────
print("[1/4] Auto-etiquetando imágenes en", SRC_DIR.name, "...")
from dataset_tools.auto_labeler import run as auto_label
auto_label(SRC_DIR, debug=False)

# ── 2. Recopilar pares (imagen, label) ──────────────────────────────────────
print("[2/4] Recopilando pares imagen+label...")
pairs = []

# a) Imágenes auto-etiquetadas (raíz de SRC_DIR)
labels_dir = SRC_DIR / "labels"
auto_ok = auto_fail = 0
for img in sorted(SRC_DIR.glob("*.jpg")):
    lbl = labels_dir / (img.stem + ".txt")
    if lbl.exists():
        pairs.append((img, lbl))
        auto_ok += 1
    else:
        auto_fail += 1

print(f"   Auto-etiquetadas : {auto_ok} OK  /  {auto_fail} sin blob")

# b) Imágenes Roboflow ya en datasets/
rflow_count = 0
for split in ["train", "valid"]:
    for img in sorted((DATASET / split / "images").glob("*.jpg")):
        lbl = DATASET / split / "labels" / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
            rflow_count += 1

print(f"   Roboflow         : {rflow_count}")
print(f"   TOTAL            : {len(pairs)} pares")

# Distribución por clase
by_class = defaultdict(int)
for img, _ in pairs:
    cls = img.name.split("_")[0].replace("rat_", "").replace("head", "head_dipping")
    by_class[cls] += 1
print("\n   Distribución:")
for c, n in sorted(by_class.items()):
    print(f"     {c:<18} {n}")

# ── 3. Shuffle y split 80/20 ────────────────────────────────────────────────
print("\n[3/4] Dividiendo train/valid (80/20)...")
random.seed(42)
random.shuffle(pairs)
n_train = int(len(pairs) * 0.8)
splits = {"train": pairs[:n_train], "valid": pairs[n_train:]}

# ── 4. Limpiar y copiar ──────────────────────────────────────────────────────
print("[4/4] Escribiendo dataset...")
import stat, os

def _force_remove(path):
    """Borra recursivo con permisos forzados (Windows)."""
    def _on_error(func, p, _):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if path.exists():
        shutil.rmtree(path, onerror=_on_error)
    path.mkdir(parents=True)

for split in ["train", "valid"]:
    for sub in ["images", "labels"]:
        _force_remove(DATASET / split / sub)

for split, items in splits.items():
    for i, (img, lbl) in enumerate(items):
        shutil.copy(img, DATASET / split / "images" / f"rat_{i:05d}{img.suffix}")
        shutil.copy(lbl, DATASET / split / "labels" / f"rat_{i:05d}.txt")
    print(f"   {split}: {len(items)} imágenes")

print("\n[OK] Dataset reconstruido. Listo para entrenar.")
