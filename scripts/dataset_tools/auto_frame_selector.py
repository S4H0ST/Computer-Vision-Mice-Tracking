"""
auto_frame_selector.py — Selección automática de frames para Roboflow.

Procesa:
  - testRata1: frames ya extraídos en media_original/frames_original/
  - testRata2: directamente desde el vídeo (dos pasadas: scan → guardar)
  - testRata3: directamente desde el vídeo (dos pasadas: scan → guardar)

Usa yolo_ratas.pt para predecir la clase de cada frame (referencia para el
usuario — se corregirá manualmente en Roboflow). Selecciona los frames más
nítidos garantizando diversidad temporal (MIN_GAP fotogramas entre elegidos
de la misma clase en el mismo vídeo).

Salida: media_original/roboflow_testRata1/seleccion/
Naming: {clase}_{video}_f{frame:05d}_blur{nitidez}.jpg

Uso:
    cd scripts
    python dataset_tools/auto_frame_selector.py
"""
import sys
import shutil
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent.parent
FRAMES1_DIR = ROOT / "media_original" / "frames_original"
VIDEOS_DIR  = ROOT / "media_original" / "videos"
OUT_DIR     = ROOT / "media_original" / "roboflow_testRata1" / "seleccion"
MODEL_PATH  = ROOT / "models" / "yolo_ratas.pt"

# ── Parámetros ────────────────────────────────────────────────────────────────
CLASS_NAMES = ["climbing", "grooming", "head_dipping", "horizontal", "rearing"]

CONF_MIN = 0.25   # umbral de confianza (bajo porque el modelo aún mejorará)
MIN_GAP  = 90     # fotogramas mínimos entre seleccionados del mismo clase+vídeo

# Cuántos frames NUEVOS añadir por (vídeo, clase)
TARGETS = {
    "testRata1": {"grooming": 15, "head_dipping": 15, "rearing": 10,
                  "climbing": 10, "horizontal":  5},
    "testRata2": {"grooming": 30, "head_dipping": 30, "rearing": 25,
                  "climbing": 20, "horizontal": 15},
    "testRata3": {"grooming": 15, "head_dipping": 15, "rearing": 15,
                  "climbing": 10, "horizontal": 10},
}

# Cada cuántos frames muestrear (evita procesar el vídeo entero)
STEP = {
    "testRata1": 10,   # ~4422 archivos → ~442 muestreados
    "testRata2": 45,   # vídeo largo    → ~400 muestreados
    "testRata3": 15,   # vídeo corto    → ~200 muestreados
}


# ── Utilidades ────────────────────────────────────────────────────────────────

def blur_score(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def run_yolo(model, img_bgr):
    """Devuelve (class_idx, conf) de la detección de mayor confianza, o None."""
    results = model(img_bgr, verbose=False, conf=CONF_MIN)
    if not results or len(results[0].boxes) == 0:
        return None
    boxes = results[0].boxes
    best  = int(boxes.conf.argmax().item())
    return int(boxes.cls[best].item()), float(boxes.conf[best].item())


def select_diverse(candidates, target, min_gap):
    """
    candidates: list of (frame_idx, blur, extra)
    Devuelve hasta `target` ítems ordenados por blur DESC con gap >= min_gap
    entre cualquier par de frame_idx seleccionados.
    """
    sorted_c = sorted(candidates, key=lambda x: x[1], reverse=True)
    selected  = []
    sel_idxs  = []

    for frame_idx, blur, extra in sorted_c:
        if len(selected) >= target:
            break
        if all(abs(frame_idx - f) >= min_gap for f in sel_idxs):
            selected.append((frame_idx, blur, extra))
            sel_idxs.append(frame_idx)

    return selected


# ── FUENTE 1: archivos de imagen ya extraídos (testRata1) ─────────────────────

def scan_images(model, files, step):
    """Escanea archivos de imagen muestreados → candidatos por clase."""
    candidates = defaultdict(list)   # cls → [(frame_idx, blur, Path)]
    sorted_files = sorted(files)
    total = len(sorted_files)

    for i, img_path in enumerate(sorted_files):
        if i % step != 0:
            continue
        if i % (step * 20) == 0:
            print(f"   ... {i}/{total} archivos", end="\r")

        try:
            frame_idx = int(img_path.stem.split("_")[-1])
        except ValueError:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        score  = blur_score(img)
        result = run_yolo(model, img)
        if result is None:
            continue
        cls_idx, _ = result
        candidates[CLASS_NAMES[cls_idx]].append((frame_idx, score, img_path))

    print()
    return candidates


def save_from_files(selected_by_class, video_stem, out_dir):
    """Copia los archivos seleccionados al destino con el naming estándar."""
    saved = defaultdict(int)
    for cls_name, items in selected_by_class.items():
        for frame_idx, blur, src_path in items:
            fname = f"{cls_name}_{video_stem}_f{frame_idx:05d}_blur{int(blur)}.jpg"
            dst   = out_dir / fname
            if not dst.exists():
                shutil.copy(src_path, dst)
                saved[cls_name] += 1
    return saved


# ── FUENTE 2: vídeo directo (testRata2, testRata3) ───────────────────────────

def scan_video(model, video_path, step):
    """
    Primera pasada: recorre el vídeo muestreando cada `step` fotogramas.
    No almacena imágenes — sólo (frame_idx, blur) por clase.
    """
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    candidates = defaultdict(list)   # cls → [(frame_idx, blur)]
    fi = 0
    processed = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break

        if fi % step == 0:
            score  = blur_score(img)
            result = run_yolo(model, img)
            if result is not None:
                cls_idx, _ = result
                candidates[CLASS_NAMES[cls_idx]].append((fi, score, None))
            processed += 1
            if processed % 30 == 0:
                pct = fi / max(total, 1) * 100
                print(f"   ... frame {fi}/{total} ({pct:.0f}%)", end="\r")

        fi += 1

    cap.release()
    print()
    return candidates


def save_from_video(video_path, selected_by_class, out_dir):
    """
    Segunda pasada (seek): lee sólo los fotogramas elegidos y los guarda.
    Usar seek es eficiente para MP4 con pocos frames a recuperar.
    """
    items_to_save = []
    for cls_name, items in selected_by_class.items():
        for frame_idx, blur in items:
            items_to_save.append((frame_idx, blur, cls_name))

    if not items_to_save:
        return defaultdict(int)

    video_stem = video_path.stem
    saved = defaultdict(int)
    cap   = cv2.VideoCapture(str(video_path))

    for frame_idx, blur, cls_name in items_to_save:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = cap.read()
        if not ret:
            continue
        fname = f"{cls_name}_{video_stem}_f{frame_idx:05d}_blur{int(blur)}.jpg"
        dst   = out_dir / fname
        if not dst.exists():
            cv2.imwrite(str(dst), img)
            saved[cls_name] += 1

    cap.release()
    return saved


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    from ultralytics import YOLO

    if not MODEL_PATH.exists():
        print(f"[X] Modelo no encontrado: {MODEL_PATH}")
        print("    Entrena primero con train_yolo.py")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[+] Modelo : {MODEL_PATH.name}")
    print(f"[+] Salida : {OUT_DIR}")
    print(f"[+] conf_min={CONF_MIN}  min_gap={MIN_GAP} fotogramas\n")

    model = YOLO(str(MODEL_PATH))
    total_saved = defaultdict(int)

    # ── testRata1 (archivos pre-extraídos) ───────────────────────────────────
    vname = "testRata1"
    print(f"[1/3] {vname} — escaneando frames pre-extraídos (cada {STEP[vname]})...")
    files      = list(FRAMES1_DIR.glob("testRata1_*.jpg"))
    candidates = scan_images(model, files, STEP[vname])

    sel_by_cls = {}
    for cls_name, tgt in TARGETS[vname].items():
        cands = candidates.get(cls_name, [])
        sel   = select_diverse(cands, tgt, MIN_GAP)
        sel_by_cls[cls_name] = sel
        print(f"   {cls_name:15s}: {len(cands):4d} candidatos → {len(sel):3d} seleccionados")

    saved = save_from_files(sel_by_cls, vname, OUT_DIR)
    for cls_name, n in saved.items():
        total_saved[cls_name] += n

    # ── testRata2 y testRata3 (vídeo directo) ────────────────────────────────
    for seq_idx, vname in enumerate(["testRata2", "testRata3"], start=2):
        video_path = VIDEOS_DIR / f"{vname}.mp4"
        if not video_path.exists():
            print(f"\n[{seq_idx}/3] {vname} — no encontrado, saltando.")
            continue

        print(f"\n[{seq_idx}/3] {vname} — 1ª pasada: escaneando vídeo (cada {STEP[vname]} frames)...")
        candidates = scan_video(model, video_path, STEP[vname])

        sel_by_cls = {}
        for cls_name, tgt in TARGETS[vname].items():
            cands = candidates.get(cls_name, [])
            sel   = select_diverse(cands, tgt, MIN_GAP)
            # Descartar el 'extra' (no lo necesitamos para vídeo)
            sel_by_cls[cls_name] = [(fi, bl) for fi, bl, *_ in sel]
            print(f"   {cls_name:15s}: {len(cands):4d} candidatos → {len(sel):3d} seleccionados")

        print(f"   2ª pasada: guardando frames seleccionados...")
        saved = save_from_video(video_path, sel_by_cls, OUT_DIR)
        for cls_name, n in saved.items():
            total_saved[cls_name] += n

    # ── Resumen ───────────────────────────────────────────────────────────────
    print(f"\n{'='*45}")
    print(f" NUEVOS FRAMES AÑADIDOS A seleccion/")
    print(f"{'='*45}")
    for cls_name in CLASS_NAMES:
        print(f"  {cls_name:15s}: +{total_saved[cls_name]}")
    print(f"  {'TOTAL':15s}: +{sum(total_saved.values())}")
    print(f"\n[OK] Listo. Sube seleccion/ a Roboflow para etiquetar manualmente.")


if __name__ == "__main__":
    main()
