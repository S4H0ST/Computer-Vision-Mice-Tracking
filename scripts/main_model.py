"""
Menu principal interactivo del pipeline de deteccion y entrenamiento.

Opciones disponibles:
  1. Entrenar Modelo YOLO       - lanza YOLOTrainer con los parametros de config.py
  2. Deteccion desde Video      - selector de archivo, calibracion, deteccion y stats
  3. Deteccion desde Camara     - captura frame de camara, calibracion, deteccion en vivo
  4. Salir

Funciones:
    _ask_box_size  - pregunta el tamano fisico de la caja y lo guarda en coords.json.
    _pick_video    - abre un explorador de archivos filtrado a videos.
    _run_detection - logica compartida entre modo video y modo camara (calibracion + detector + stats).
    main           - bucle del menu.
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


def _select_device() -> None:
    """Detecta GPU/CPU al arrancar y permite elegir el dispositivo de inferencia."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n[HW] GPU detectada: {gpu_name}")
            choice = input("[?] Usar GPU (Enter) o CPU? [G/c]: ").strip().lower()
            if choice == "c":
                detect_cfg.device = "cpu"
                train_cfg.device  = "cpu"
                print("[HW] Dispositivo seleccionado: CPU")
            else:
                detect_cfg.device = "0"
                train_cfg.device  = "0"
                print("[HW] Dispositivo seleccionado: GPU")
        else:
            detect_cfg.device = "cpu"
            train_cfg.device  = "cpu"
            print("\n[HW] Sin GPU CUDA disponible — usando CPU.")
    except ImportError:
        detect_cfg.device = "cpu"
        train_cfg.device  = "cpu"
        print("\n[HW] PyTorch no encontrado — usando CPU.")


def _ask_box_size(coords_path: Path) -> None:
    """
    Pregunta al usuario el tamano fisico de la caja y lo guarda en coords_path.
    Este dato permite a StatsGenerator convertir pixeles a centimetros en el Excel.
    Si el usuario deja los campos vacios, no se guarda nada (es opcional).
    """
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
    """
    Abre un explorador de archivos filtrado a formatos de video.
    Devuelve la ruta elegida o None si el usuario cancela.
    """
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


def _run_detection(frame_for_calib, source_label: str,
                   camera_index: int | None = None,
                   video_path: Path | None = None) -> None:
    """
    Logica compartida de calibracion + deteccion + estadisticas.
    Sirve tanto para modo video como para modo camara en vivo.

    frame_for_calib : frame BGR (numpy array) que se usara para calibrar.
    source_label    : nombre de la fuente para los mensajes de consola.
    camera_index    : si no es None, RatDetector usara la camara en lugar de archivo.
    video_path      : ruta del video (solo en modo video).
    """
    import cv2

    # -- Calibracion --
    recalibrar = True
    if paths.coords_json.exists():
        resp = input("[?] Ya existe una calibracion guardada. Recalibrar? (s/N): ").strip().lower()
        recalibrar = resp == "s"

    if recalibrar:
        # Guardamos el frame en un temporal y lanzamos el calibrador visual
        tmp_frame = Path(tempfile.gettempdir()) / "rat_calib_frame.jpg"
        cv2.imwrite(str(tmp_frame), frame_for_calib)
        print(f"[Calibrador] Frame extraido de: {source_label}")
        print("[Calibrador] Marca las zonas y pulsa S para guardar, Q para cancelar.")
        calib = ImageCalibrator(tmp_frame, paths.coords_json)
        calib.run()
        if paths.coords_json.exists():
            _ask_box_size(paths.coords_json)

    if not paths.coords_json.exists():
        print("[!] Sin calibracion disponible. Cancelando deteccion.")
        return

    # -- Rutas de salida --
    today = datetime.now().strftime("%Y%m%d_%H%M%S")
    if video_path is not None:
        stem = video_path.stem
    else:
        stem = "camera"

    run_folder = paths.detect_dir / f"{stem}_{today}"
    run_folder.mkdir(parents=True, exist_ok=True)

    if video_path is not None:
        paths.video_source = video_path

    paths.output_video = run_folder / f"{stem}_{today}.mp4"

    # -- Deteccion --
    # show_preview=True en camara para que el usuario vea y pueda parar con Q
    show_preview = camera_index is not None
    detector = RatDetector(detect_cfg,
                           show_preview=show_preview,
                           camera_index=camera_index)
    detector.run()

    # -- Estadisticas --
    csv_path = paths.output_video.with_suffix(".csv")
    if csv_path.exists():
        print("\n[Stats] Generando estadisticas...")
        StatsGenerator(csv_path, coords_json=paths.coords_json).generate(
            run_folder / "stats"
        )

    print(f"\n[OK] Resultados guardados en: {run_folder}")


def main() -> None:
    """Bucle principal del menu. Orquesta las opciones del pipeline."""
    paths.check_dirs()
    _select_device()

    while True:
        print("\n" + "=" * 45)
        print("  RAT MODEL MANAGER")
        print("=" * 45)
        print("1. Entrenar Modelo YOLO")
        print("2. Ejecutar Deteccion desde Video")
        print("3. Ejecutar Deteccion desde Camara (en vivo)")
        print("4. Salir")

        opt: str = input("\n[?] Opcion: ")

        # -- OPCION 1: Entrenamiento --
        if opt == "1":
            trainer = YOLOTrainer(train_cfg)
            trainer.run()

        # -- OPCION 2: Deteccion desde video --
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

            _run_detection(frame, source_label=video_path.name,
                           video_path=video_path)

        # -- OPCION 3: Deteccion en vivo desde camara --
        elif opt == "3":
            import cv2

            cam_idx = 0   # indice 0 = camara principal del sistema
            print(f"\n[Camara] Intentando abrir camara [{cam_idx}]...")
            cap = cv2.VideoCapture(cam_idx)

            if not cap.isOpened():
                print(f"[!] No se pudo abrir la camara [{cam_idx}].")
                print("    Asegurate de que la camara no esta siendo usada por otra aplicacion.")
                cap.release()
                continue

            ok, frame = cap.read()
            cap.release()

            if not ok:
                print("[!] La camara se abrio pero no devolvio ningun frame.")
                continue

            print(f"[OK] Camara abierta. Frame de muestra capturado para calibracion.")
            print("     Una vez calibrado, la deteccion se ejecutara en tiempo real.")
            print("     Pulsa Q en la ventana de preview para detener la grabacion.\n")

            _run_detection(frame, source_label=f"camara [{cam_idx}]",
                           camera_index=cam_idx)

        # -- OPCION 4: Salir --
        elif opt == "4":
            print("[*] Saliendo...")
            break

        else:
            print("[!] Opcion no valida.")


if __name__ == "__main__":
    main()
