"""
Menu principal interactivo del pipeline de deteccion y entrenamiento.

Funciones:
    _pick_file  — abre un explorador de archivos (imagen o video) y devuelve la ruta elegida.
    _pick_video — abre un explorador de archivos filtrado a videos y devuelve la ruta elegida.
    main        — bucle de menu que orquesta calibracion, entrenamiento y deteccion.
"""

import sys
import tempfile
from pathlib import Path
from datetime import datetime
from config.config import paths, train_cfg, detect_cfg
from detection.calibrator import ZoneCalibrator
from detection.trainer import YOLOTrainer
from detection.detector import RatDetector
from utils.calibrator_image import ImageCalibrator
from utils.stats_generator import StatsGenerator


IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS: set[str] = {".mp4", ".avi", ".mov", ".mkv"}


def _pick_file() -> Path | None:
    """Abre un explorador de archivos y devuelve la ruta elegida, o None si se cancela."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root_tk = tk.Tk()
        root_tk.withdraw()
        root_tk.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(
            title="Selecciona imagen o video para calibrar",
            filetypes=[
                ("Imagenes y videos", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv"),
                ("Imagenes", "*.jpg *.jpeg *.png *.bmp"),
                ("Videos", "*.mp4 *.avi *.mov *.mkv"),
                ("Todos", "*.*"),
            ],
        )
        root_tk.destroy()
        return Path(chosen) if chosen else None
    except Exception as e:
        print(f"[!] No se pudo abrir el explorador: {e}")
        return None


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
        print("1. Calibrar Zonas (Paredes/Agujeros)")
        print("2. Entrenar Modelo YOLO")
        print("3. Ejecutar Deteccion y Analisis")
        print("4. Salir")

        opt: str = input("\n[?] Opcion: ")

        if opt == "1":
            print("\n[Calibrador] Abriendo explorador de archivos...")
            file_path = _pick_file()

            if file_path is None:
                print("[!] Ningun archivo seleccionado.")
                continue

            ext = file_path.suffix.lower()
            if ext in IMAGE_EXTS:
                print(f"[Calibrador] Imagen seleccionada: {file_path.name}")
                calib = ImageCalibrator(file_path, paths.coords_json)
                calib.run()
            elif ext in VIDEO_EXTS:
                print(f"[Calibrador] Video seleccionado: {file_path.name}")
                calib = ZoneCalibrator(file_path)
                calib.run()
            else:
                print(f"[!] Formato no reconocido: {ext}. Usa imagen o video.")

        elif opt == "2":
            trainer = YOLOTrainer(train_cfg)
            trainer.run()

        elif opt == "3":
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

        elif opt == "4":
            print("[*] Saliendo...")
            break

        else:
            print("[!] Opcion no valida.")


if __name__ == "__main__":
    main()
