import sys
from pathlib import Path
from config.config import paths, train_cfg, detect_cfg
from yolo_pose.calibrator import ZoneCalibrator
from yolo_pose.trainer import YOLOTrainer
from yolo_pose.detector import RatDetector
from rnn.trainer_manager import RNNTrainer
from tools.calibrator_image import ImageCalibrator


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _pick_file() -> Path | None:
    """Abre un explorador de archivos y devuelve la ruta elegida."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root_tk = tk.Tk()
        root_tk.withdraw()
        root_tk.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(
            title="Selecciona imagen o video para calibrar",
            filetypes=[
                ("Imágenes y videos", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv"),
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp"),
                ("Videos", "*.mp4 *.avi *.mov *.mkv"),
                ("Todos", "*.*"),
            ],
        )
        root_tk.destroy()
        return Path(chosen) if chosen else None
    except Exception as e:
        print(f"[!] No se pudo abrir el explorador: {e}")
        return None


def main():
    paths.check_dirs()

    while True:
        print("\n" + "=" * 40)
        print(" [(;)] RAT MODEL MANAGER (Entrenamiento & IA)")
        print("=" * 40)
        print("1. Calibrar Zonas (Paredes/Agujeros)")
        print("2. Entrenar Modelo YOLO")
        print("3. Ejecutar Deteccion y Analisis")
        print("4. Entrenar Cerebro RNN")
        print("5. Salir")

        opt = input("\n[?] Opción: ")

        if opt == "1":
            print("\n[Calibrador] Abriendo explorador de archivos...")
            file_path = _pick_file()

            if file_path is None:
                print("[!] Ningún archivo seleccionado.")
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
            if not paths.coords_json.exists():
                print("[¿?] ERROR: Primero debes calibrar las zonas (Opción 1 o calibrator_image.py).")
                continue

            detector = RatDetector(detect_cfg)
            detector.run()

        elif opt == "4":
            print("\n[Brain] Iniciando entrenamiento de la red temporal (LSTM)...")
            brain_trainer = RNNTrainer()
            brain_trainer.train()

        elif opt == "5":
            print("[*] Saliendo...")
            break

        else:
            print("[!] Opción no válida.")


if __name__ == "__main__":
    main()
