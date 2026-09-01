"""
Menu principal interactivo del pipeline de deteccion y entrenamiento.

Funciones:
    _pick_file  — abre un explorador de archivos (imagen o video) y devuelve la ruta elegida.
    _pick_video — abre un explorador de archivos filtrado a videos y devuelve la ruta elegida.
    main        — bucle de menu que orquesta calibracion, entrenamiento y deteccion.
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
from config.config import paths, train_cfg, detect_cfg
from detection.trainer import YOLOTrainer
from detection.detector import RatDetector
from calibration.calibrator_image import ImageCalibrator
from utils.stats_generator import StatsGenerator


VIDEO_EXTS: set[str] = {".mp4", ".avi", ".mov", ".mkv"}


def _ask_box_size(coords_path: Path) -> None:
    """Pregunta el tamano fisico de la caja y lo guarda en coords.json."""
    print("\n[Calibrador] Tamano fisico de la caja (Enter para omitir):")
    try:
        w_str = input("  Ancho de la caja en cm: ").strip()
        h_str = input("  Alto  de la caja en cm: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    try:
        box_w = float(w_str) if w_str else None
        box_h = float(h_str) if h_str else None
    except ValueError:
        print("[!] Valor no valido — tamano fisico no guardado.")
        return

    if box_w is None and box_h is None:
        return

    if not coords_path.exists():
        return

    with open(coords_path, "r") as f:
        data = json.load(f)

    if box_w is not None:
        data["box_width_cm"]  = box_w
    if box_h is not None:
        data["box_height_cm"] = box_h

    with open(coords_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[OK] Tamano guardado: {box_w} x {box_h} cm")


def _pick_video() -> Path | None:
    """Abre un explorador de archivos filtrado a videos y devuelve la ruta elegida, o None si se cancela."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root_tk = tk.Tk()
        root_tk.withdraw()
        root_tk.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(
            title="Selecciona el video a analizar",
            filetypes=[
                ("Videos", "*.mp4 *.avi *.mov *.mkv"),
                ("Todos", "*.*"),
            ],
        )
        root_tk.destroy()
        return Path(chosen) if chosen else None
    except Exception as e:
        print(f"[!] No se pudo abrir el explorador: {e}")
        return None


def main() -> None:
    paths.check_dirs()

    while True:
        print("\n" + "=" * 40)
        print(" [(;)] RAT MODEL MANAGER (Entrenamiento & IA)")
        print("=" * 40)
        print("1. Entrenar Modelo YOLO")
        print("2. Ejecutar Deteccion y Analisis")
        print("3. Salir")

        opt: str = input("\n[?] Opcion: ")

        if opt == "1":
            trainer = YOLOTrainer(train_cfg)
            trainer.run()

        elif opt == "2":
            import cv2

            print("\n[Detector] Selecciona el video a analizar...")
            video_path = _pick_video()
            if video_path is None:
                print("[!] Ningun video seleccionado.")
                continue

            cap = cv2.VideoCapture(str(video_path))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                print(f"[!] No se pudo leer el video: {video_path.name}")
                continue

            recalibrar = True
            if paths.coords_json.exists():
                resp = input("[?] Ya existe una calibracion guardada. Recalibrar? (s/N): ").strip().lower()
                recalibrar = resp == "s"

            if recalibrar:
                tmp_frame = Path(tempfile.gettempdir()) / "rat_calib_frame.jpg"
                cv2.imwrite(str(tmp_frame), frame)

                print(f"[Calibrador] Primer frame extraido de: {video_path.name}")
                print("[Calibrador] Marca las zonas y pulsa S para guardar, Q para cancelar.")
                calib = ImageCalibrator(tmp_frame, paths.coords_json)
                calib.run()
                if paths.coords_json.exists():
                    _ask_box_size(paths.coords_json)

            if not paths.coords_json.exists():
                print("[!] Sin calibracion disponible. No se ejecutara la deteccion.")
                continue

            today      = datetime.now().strftime("%Y%m%d")
            stem       = video_path.stem
            run_folder = paths.detect_dir / f"{stem}_{today}"
            run_folder.mkdir(parents=True, exist_ok=True)

            paths.video_source = video_path
            paths.output_video = run_folder / f"{stem}_{today}.mp4"

            detector = RatDetector(detect_cfg, show_skeleton=False, show_preview=False,
                                   dual_output=True)
            detector.run()

            csv_path = paths.output_video.with_suffix(".csv")
            if csv_path.exists():
                print("\n[Stats] Generando estadisticas...")
                StatsGenerator(csv_path, coords_json=paths.coords_json).generate(
                    run_folder / "stats"
                )

            print(f"\n[OK] Resultados guardados en: {run_folder}")

        elif opt == "3":
            print("[*] Saliendo...")
            break

        else:
            print("[!] Opcion no valida.")


if __name__ == "__main__":
    main()
