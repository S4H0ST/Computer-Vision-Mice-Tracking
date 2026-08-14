"""
Extractor de frames candidatos para anotación manual.

Corre el modelo entrenado sobre los videos con umbral muy bajo (conf=0.10) para
encontrar frames donde el ratón podría estar haciendo grooming o rearing.
También guarda un frame cada SCAN_EVERY_SEC segundos como red de seguridad.

Resultado:
    outputs/frame_candidates/
        grooming/   ← modelo detectó grooming aquí
        rearing/    ← modelo detectó rearing aquí
        scan/       ← frames del intervalo fijo (si el modelo se los perdió)

Naming de archivos:
    <video>_f<frame>_c<conf>.jpg   ← detectado por modelo
    <video>_f<frame>_scan.jpg      ← muestreo fijo (sin detección)

Uso:
    python scripts/tools/extract_label_candidates.py
    python scripts/tools/extract_label_candidates.py testRata1.mp4 testRata2.mp4
"""

import sys
import cv2
from pathlib import Path

# Añadir scripts/ al path para imports relativos
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import paths

# ── Configuración ─────────────────────────────────────────────────────────── #
TARGET_CLASSES  = {1: "grooming", 4: "rearing"}
MIN_CONF        = 0.10   # muy bajo para no perder detecciones escasas
MIN_GAP_FRAMES  = 20     # mínimo frames entre guardados de la misma clase (anti-burst)
SCAN_EVERY_SEC  = 20     # cada N segundos guardar frame de repaso en scan/
OUTPUT_DIR      = paths.root / "outputs" / "frame_candidates"


def _save(frame, folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{stem}.jpg"), frame)


def process_video(model, video_path: Path) -> dict:
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = max(1.0, cap.get(cv2.CAP_PROP_FPS))
    scan_interval = max(1, int(fps * SCAN_EVERY_SEC))

    # Último frame guardado por clase (para el anti-burst)
    last_saved = {name: -(MIN_GAP_FRAMES + 1) for name in TARGET_CLASSES.values()}
    last_scan  = -(scan_interval + 1)
    saved      = {name: 0 for name in TARGET_CLASSES.values()}
    saved["scan"] = 0
    frame_idx  = 0

    dur_s = total / fps
    print(f"\n[{video_path.name}]  {total} frames @ {fps:.1f} fps  ({dur_s/60:.1f} min)")
    print(f"  scan cada {scan_interval} frames ({SCAN_EVERY_SEC}s)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, verbose=False)[0]

        # ── Detecciones de clase objetivo ─────────────────────────────────── #
        detected: dict[str, float] = {}
        if results.boxes is not None:
            for box in results.boxes:
                cls  = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls in TARGET_CLASSES and conf >= MIN_CONF:
                    name = TARGET_CLASSES[cls]
                    if name not in detected or conf > detected[name]:
                        detected[name] = conf

        this_frame_saved_any = False
        for name, conf in detected.items():
            if frame_idx - last_saved[name] >= MIN_GAP_FRAMES:
                stem = f"{video_path.stem}_f{frame_idx:05d}_c{conf:.2f}"
                _save(frame, OUTPUT_DIR / name, stem)
                last_saved[name] = frame_idx
                saved[name] += 1
                this_frame_saved_any = True

        # ── Muestreo fijo de repaso (solo si no se guardó ya este frame) ──── #
        if not this_frame_saved_any and (frame_idx - last_scan >= scan_interval):
            stem = f"{video_path.stem}_f{frame_idx:05d}_scan"
            _save(frame, OUTPUT_DIR / "scan", stem)
            last_scan = frame_idx
            saved["scan"] += 1

        frame_idx += 1
        if frame_idx % 500 == 0:
            pct = frame_idx / total * 100
            print(f"  {frame_idx:5d}/{total} ({pct:.0f}%) "
                  f"grooming={saved['grooming']} rearing={saved['rearing']} scan={saved['scan']}")

    cap.release()
    return saved


def main():
    from ultralytics import YOLO

    model_path = paths.yolo_model
    if not model_path.exists():
        print(f"[ERROR] Modelo no encontrado: {model_path}")
        sys.exit(1)

    if len(sys.argv) > 1:
        videos = [paths.video_dir / v for v in sys.argv[1:]]
    else:
        videos = sorted(paths.video_dir.glob("*.mp4"))

    videos = [v for v in videos if v.exists()]
    if not videos:
        print("[ERROR] No se encontraron videos.")
        sys.exit(1)

    print(f"Modelo: {model_path}")
    print(f"Videos: {[v.name for v in videos]}")
    print(f"Destino: {OUTPUT_DIR}\n")

    model  = YOLO(str(model_path))
    totals = {name: 0 for name in TARGET_CLASSES.values()}
    totals["scan"] = 0

    for v in videos:
        r = process_video(model, v)
        for k, n in r.items():
            totals[k] += n

    print("\n" + "=" * 50)
    print("RESUMEN FINAL")
    print("=" * 50)
    print(f"  grooming/ : {totals['grooming']} candidatos")
    print(f"  rearing/  : {totals['rearing']} candidatos")
    print(f"  scan/     : {totals['scan']} frames de repaso")
    print(f"\nRevisa las carpetas en:\n  {OUTPUT_DIR}")
    print("\nPasos siguientes:")
    print("  1. Abre grooming/ y elige 20-30 imágenes que muestren claramente grooming")
    print("  2. Haz lo mismo con rearing/")
    print("  3. Si no hay suficientes, busca en scan/ frames candidatos manualmente")
    print("  4. Etiquétalos en Roboflow y añádelos al dataset")


if __name__ == "__main__":
    main()
