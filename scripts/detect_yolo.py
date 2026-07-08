"""Ejecuta la deteccion sobre un video. Uso:
    python detect_yolo.py --video ruta/al/video.mp4
Si no se pasa --video usa la ruta por defecto en configuracion.py.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from helpers.configuracion import paths, detect_cfg

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Ruta al video a analizar (mp4)")
args = parser.parse_args()

if args.video:
    paths.video_source = Path(args.video)
    paths.output_video = paths.output_dir / (Path(args.video).stem + "_resultado.mp4")

paths.check_dirs()
print(f"[INFO] Video   : {paths.video_source}")
print(f"[INFO] Modelo  : {paths.yolo_model}")
print(f"[INFO] Salida  : {paths.output_video}")

if not paths.yolo_model.exists():
    print("[X] No existe yolo_ratas.pt — entrena primero con train_yolo.py")
    sys.exit(1)

if not paths.video_source.exists():
    print(f"[X] No encuentro el video: {paths.video_source}")
    sys.exit(1)

from modules.core_yolo.detector import RatDetector
detector = RatDetector(detect_cfg)
detector.run()
