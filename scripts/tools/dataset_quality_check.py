"""
Herramienta de evaluación de calidad del dataset YOLOv8-Pose.

Analiza datasets/ y genera un reporte con:
  1. Distribución de clases (train / valid)
  2. Integridad de anotaciones
  3. Nitidez de imágenes (varianza del Laplaciano)
  4. Tabla resumen y recomendaciones

Uso:
    python scripts/tools/dataset_quality_check.py
"""

import sys
import io
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# Forzar UTF-8 en stdout para que los caracteres especiales no fallen en Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Rutas ─────────────────────────────────────────────────────────────────── #
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

DATASETS_DIR   = PROJECT_ROOT / "datasets"
OUTPUT_DIR     = PROJECT_ROOT / "outputs"
REPORT_PATH    = OUTPUT_DIR / "dataset_quality_report.txt"
DATA_YAML      = DATASETS_DIR / "data.yaml"

# Umbrales
SHARPNESS_THRESHOLD  = 100.0   # varianza del Laplaciano
BLUR_ALERT_PCT       = 20.0    # alerta si más del 20 % de imágenes son borrosas
MIN_TRAIN_IMGS       = 30      # alerta si clase < 30 imágenes en train
IMBALANCE_RATIO      = 3.0     # alerta si clase_max / clase_min > 3
BBOX_AREA_MIN        = 0.005   # bbox menor al 0.5 % del área → sospechosa


# ── Helpers ────────────────────────────────────────────────────────────────── #
def _load_class_names() -> list[str]:
    if not DATA_YAML.exists():
        return []
    with open(DATA_YAML) as f:
        for line in f:
            if line.strip().startswith("names:"):
                raw = line.split(":", 1)[1].strip()
                raw = raw.strip("[]").replace("'", "").replace('"', "")
                return [n.strip() for n in raw.split(",")]
    return []


def _laplacian_variance(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _parse_label_file(txt_path: Path):
    """
    Devuelve lista de dicts con los campos de cada línea de anotación YOLO-Pose.
    kpt_shape [3, 3] → 14 valores por línea.
    """
    annotations = []
    lines = txt_path.read_text().strip().splitlines()
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        ann = {"cls": int(parts[0]), "raw": parts}
        if len(parts) >= 5:
            ann["cx"]  = float(parts[1])
            ann["cy"]  = float(parts[2])
            ann["bw"]  = float(parts[3])
            ann["bh"]  = float(parts[4])
        # keypoints: cols 5..13 → (x,y,v) × 3
        kpts = []
        for k in range(3):
            base = 5 + k * 3
            if base + 2 < len(parts):
                kx, ky, kv = float(parts[base]), float(parts[base+1]), float(parts[base+2])
                kpts.append((kx, ky, kv))
        ann["kpts"] = kpts
        annotations.append(ann)
    return annotations


def _report_section(title: str) -> str:
    bar = "═" * 60
    return f"\n{bar}\n  {title}\n{bar}"


# ── Análisis principal ─────────────────────────────────────────────────────── #
def analyze(splits: list[str] = ("train", "valid")):
    class_names = _load_class_names()
    n_classes   = len(class_names)

    lines_out: list[str] = []
    issues: list[tuple[str, str]] = []   # (severidad, descripción)

    def log(msg=""):
        print(msg)
        lines_out.append(msg)

    # ── 1. DISTRIBUCIÓN DE CLASES ─────────────────────────────────────────── #
    log(_report_section("1. DISTRIBUCIÓN DE CLASES"))

    counts_by_split: dict[str, dict[int, int]] = {}

    for split in splits:
        img_dir = DATASETS_DIR / split / "images"
        lbl_dir = DATASETS_DIR / split / "labels"
        if not img_dir.exists():
            log(f"  ⚠  {split}/images no encontrado — omitiendo")
            continue

        cls_count: dict[int, int] = defaultdict(int)
        for lbl_file in lbl_dir.glob("*.txt"):
            try:
                anns = _parse_label_file(lbl_file)
                for ann in anns:
                    cls_count[ann["cls"]] += 1
            except Exception:
                pass
        counts_by_split[split] = cls_count

    # Cabecera tabla
    col_w = 22
    header = f"  {'Clase':<{col_w}}" + "".join(f"  {s:>8}" for s in splits)
    log(header)
    log("  " + "-" * (col_w + 12 * len(splits)))

    for cls_id in range(n_classes):
        name = class_names[cls_id] if cls_id < n_classes else f"cls_{cls_id}"
        row  = f"  {name:<{col_w}}"
        for split in splits:
            cnt = counts_by_split.get(split, {}).get(cls_id, 0)
            row += f"  {cnt:>8}"
        log(row)

    # Alertas distribución
    train_counts = counts_by_split.get("train", {})
    if train_counts:
        for cls_id, cnt in train_counts.items():
            name = class_names[cls_id] if cls_id < n_classes else f"cls_{cls_id}"
            if cnt < MIN_TRAIN_IMGS:
                msg = f"Clase '{name}' tiene solo {cnt} imágenes en train (mínimo recomendado: {MIN_TRAIN_IMGS})"
                issues.append(("🔴 CRÍTICO", msg))
                log(f"\n  ⚠  ALERTA: {msg}")

        vals = list(train_counts.values())
        if vals and max(vals) / max(min(vals), 1) > IMBALANCE_RATIO:
            msg = f"Desbalance de clases en train — ratio max/min = {max(vals)/max(min(vals),1):.1f}:1 (umbral: {IMBALANCE_RATIO:.0f}:1)"
            issues.append(("🟡 ADVERTENCIA", msg))
            log(f"\n  ⚠  ALERTA: {msg}")

    # Ratio train/valid
    log("\n  Ratios train/valid por clase:")
    for cls_id in range(n_classes):
        name  = class_names[cls_id] if cls_id < n_classes else f"cls_{cls_id}"
        tr    = counts_by_split.get("train", {}).get(cls_id, 0)
        val   = counts_by_split.get("valid", {}).get(cls_id, 0)
        ratio = f"{tr}/{val}" if val > 0 else f"{tr}/0 (sin válid)"
        log(f"    {name:<{col_w}}  {ratio}")

    # ── 2. INTEGRIDAD DE ANOTACIONES ──────────────────────────────────────── #
    log(_report_section("2. INTEGRIDAD DE ANOTACIONES"))

    for split in splits:
        img_dir = DATASETS_DIR / split / "images"
        lbl_dir = DATASETS_DIR / split / "labels"
        if not img_dir.exists():
            continue

        log(f"\n  [{split}]")

        img_files = set(p.stem for p in img_dir.glob("*.jpg")) | \
                    set(p.stem for p in img_dir.glob("*.png"))
        lbl_files = set(p.stem for p in lbl_dir.glob("*.txt"))

        # Imágenes sin .txt
        missing_lbl = img_files - lbl_files
        if missing_lbl:
            msg = f"{split}: {len(missing_lbl)} imágenes sin .txt"
            issues.append(("🔴 CRÍTICO", msg))
            log(f"    ❌ Imágenes sin anotación: {len(missing_lbl)}")
            for s in sorted(missing_lbl)[:5]:
                log(f"       · {s}")
            if len(missing_lbl) > 5:
                log(f"       ... y {len(missing_lbl)-5} más")
        else:
            log(f"    ✅ Todas las imágenes tienen .txt")

        # .txt vacíos
        empty_txts = [p for p in lbl_dir.glob("*.txt") if p.stat().st_size == 0]
        if empty_txts:
            msg = f"{split}: {len(empty_txts)} archivos .txt vacíos"
            issues.append(("🟡 ADVERTENCIA", msg))
            log(f"    ⚠  .txt vacíos: {len(empty_txts)}")
        else:
            log(f"    ✅ Ningún .txt vacío")

        # Keypoints en (0,0) con visibilidad > 0
        zero_kp_files = []
        oob_kp_files  = []
        small_bbox    = []

        for lbl_file in lbl_dir.glob("*.txt"):
            try:
                anns = _parse_label_file(lbl_file)
                for ann in anns:
                    # Bbox pequeño
                    area = ann.get("bw", 0) * ann.get("bh", 0)
                    if 0 < area < BBOX_AREA_MIN:
                        small_bbox.append(lbl_file.name)

                    for kx, ky, kv in ann.get("kpts", []):
                        # Keypoint visible pero en (0,0)
                        if kv > 0 and kx == 0.0 and ky == 0.0:
                            zero_kp_files.append(lbl_file.name)
                            break
                        # Keypoint fuera de [0,1] (excluye los no anotados)
                        if kv > 0 and (kx < 0 or kx > 1 or ky < 0 or ky > 1):
                            oob_kp_files.append(lbl_file.name)
                            break
            except Exception as e:
                log(f"    ⚠  Error leyendo {lbl_file.name}: {e}")

        if zero_kp_files:
            msg = f"{split}: {len(zero_kp_files)} archivos con keypoint en (0,0) visible"
            issues.append(("🔴 CRÍTICO", msg))
            log(f"    ❌ Keypoints en (0,0) con vis>0: {len(zero_kp_files)}")
            for s in sorted(set(zero_kp_files))[:5]:
                log(f"       · {s}")
        else:
            log(f"    ✅ Sin keypoints en (0,0) con visibilidad")

        if oob_kp_files:
            msg = f"{split}: {len(oob_kp_files)} archivos con keypoints fuera de [0,1]"
            issues.append(("🔴 CRÍTICO", msg))
            log(f"    ❌ Keypoints fuera de [0,1]: {len(oob_kp_files)}")
        else:
            log(f"    ✅ Todos los keypoints en rango [0,1]")

        if small_bbox:
            msg = f"{split}: {len(set(small_bbox))} anotaciones con bbox < {BBOX_AREA_MIN*100:.1f}% del área"
            issues.append(("🟡 ADVERTENCIA", msg))
            log(f"    ⚠  Bboxes muy pequeños: {len(set(small_bbox))}")
        else:
            log(f"    ✅ Todos los bboxes tienen tamaño razonable")

    # ── 3. ANÁLISIS DE NITIDEZ ────────────────────────────────────────────── #
    log(_report_section("3. ANÁLISIS DE NITIDEZ (varianza del Laplaciano)"))

    for split in ("train",):   # El análisis de nitidez es especialmente relevante en train
        img_dir = DATASETS_DIR / split / "images"
        lbl_dir = DATASETS_DIR / split / "labels"
        if not img_dir.exists():
            continue

        log(f"\n  [{split}]  (umbral nítida: Laplacian var >= {SHARPNESS_THRESHOLD:.0f})")

        # Agrupar por clase
        sharp_by_cls:  dict[int, int] = defaultdict(int)
        blurry_by_cls: dict[int, int] = defaultdict(int)

        for img_path in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")):
            lbl_file = lbl_dir / (img_path.stem + ".txt")
            cls_ids  = set()
            if lbl_file.exists():
                try:
                    anns = _parse_label_file(lbl_file)
                    cls_ids = {ann["cls"] for ann in anns}
                except Exception:
                    pass

            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Calcular nitidez sobre el bounding box de la rata, no sobre la
                # imagen completa. El fondo negro del box test desinfla la varianza
                # global aunque la rata esté bien enfocada.
                roi = img
                if lbl_file.exists():
                    try:
                        first_ann = _parse_label_file(lbl_file)[0] if _parse_label_file(lbl_file) else None
                        if first_ann:
                            h_im, w_im = img.shape[:2]
                            cx, cy = first_ann["cx"], first_ann["cy"]
                            bw, bh = first_ann["bw"], first_ann["bh"]
                            x1 = max(0, int((cx - bw / 2) * w_im))
                            y1 = max(0, int((cy - bh / 2) * h_im))
                            x2 = min(w_im, int((cx + bw / 2) * w_im))
                            y2 = min(h_im, int((cy + bh / 2) * h_im))
                            if x2 > x1 and y2 > y1:
                                roi = img[y1:y2, x1:x2]
                    except Exception:
                        pass

                var = _laplacian_variance(roi)
                is_sharp = var >= SHARPNESS_THRESHOLD

                for c in (cls_ids if cls_ids else {-1}):
                    if is_sharp:
                        sharp_by_cls[c] += 1
                    else:
                        blurry_by_cls[c] += 1
            except Exception:
                pass

        # Tabla
        log(f"  {'Clase':<22}  {'Nítidas':>8}  {'Borrosas':>8}  {'% Borrosas':>11}")
        log("  " + "-" * 58)
        for cls_id in range(n_classes):
            name    = class_names[cls_id] if cls_id < n_classes else f"cls_{cls_id}"
            sharp   = sharp_by_cls.get(cls_id, 0)
            blurry  = blurry_by_cls.get(cls_id, 0)
            total   = sharp + blurry
            pct_bl  = 100.0 * blurry / total if total > 0 else 0.0
            alert   = " ⚠" if pct_bl > BLUR_ALERT_PCT else "   "
            log(f"  {name:<22}  {sharp:>8}  {blurry:>8}  {pct_bl:>10.1f}%{alert}")

            if pct_bl > BLUR_ALERT_PCT:
                msg = f"Clase '{name}' tiene {pct_bl:.1f}% de imágenes borrosas en train (umbral: {BLUR_ALERT_PCT:.0f}%)"
                issues.append(("🟡 ADVERTENCIA", msg))

    # ── 4. RESUMEN FINAL ──────────────────────────────────────────────────── #
    log(_report_section("4. RESUMEN — PROBLEMAS ENCONTRADOS"))

    if not issues:
        log("\n  ✅ No se detectaron problemas significativos.")
    else:
        criticos    = [i for i in issues if i[0].startswith("🔴")]
        advertencias = [i for i in issues if i[0].startswith("🟡")]

        if criticos:
            log(f"\n  {len(criticos)} problema(s) CRÍTICO(s):")
            for sev, msg in criticos:
                log(f"    {sev}  {msg}")

        if advertencias:
            log(f"\n  {len(advertencias)} advertencia(s):")
            for sev, msg in advertencias:
                log(f"    {sev}  {msg}")

    # Recomendaciones
    log("\n  RECOMENDACIONES:")
    if any("Clase" in m and "menos de" in m.lower() or "solo" in m for _, m in issues if "CRÍTICO" in _):
        log("    · Recolecta más imágenes para las clases con pocos ejemplos.")
    if any("Desbalance" in m for _, m in issues):
        log("    · Aplica augmentation agresivo a las clases minoritarias.")
    if any("keypoint en (0,0)" in m for _, m in issues):
        log("    · Revisa y corrige las anotaciones con keypoints en (0,0) visible.")
    if any("fuera de [0,1]" in m for _, m in issues):
        log("    · Regenera las anotaciones con valores normalizados incorrectos.")
    if any("borrosas" in m for _, m in issues):
        log("    · Considera capturar más imágenes con mayor nitidez (más luz o ISO más bajo).")
    if not issues:
        log("    · El dataset está en buen estado. Continúa con el entrenamiento.")

    log("\n" + "═" * 60)

    # ── Guardar reporte ───────────────────────────────────────────────────── #
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines_out), encoding="utf-8")
    print(f"\n  Reporte guardado en: {REPORT_PATH}")


if __name__ == "__main__":
    analyze()
