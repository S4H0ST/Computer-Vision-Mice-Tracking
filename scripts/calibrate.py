"""Lanza el calibrador de zonas para cualquier video. Uso:
    python calibrate.py --video ruta/al/video.mp4
Guarda las coordenadas en datasets/coords.json.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from helpers.configuracion import paths
from modules.core_yolo.calibrator import ZoneCalibrator

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, help="Ruta al video a calibrar")
args = parser.parse_args()

video_path = Path(args.video)
if not video_path.exists():
    print(f"[X] No encuentro el video: {video_path}")
    sys.exit(1)

print(f"[INFO] Video    : {video_path}")
print(f"[INFO] Guardará : {paths.coords_json}")
print()
print("  PASO 1 → Haz clic en 2 esquinas del borde EXTERIOR (rojo)")
print("  PASO 2 → Haz clic en 2 esquinas del borde INTERIOR (azul)")
print("  PASO 3 → Haz clic en los 4 centros de los AGUJEROS (verde)")
print("  Pulsa Q para guardar | R para resetear")
print()

calib = ZoneCalibrator(video_path)
calib.run()
