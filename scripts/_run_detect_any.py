"""Deteccion sobre cualquier video. Uso: python _run_detect_any.py testRata3.mp4"""
import sys, shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config.config import paths, detect_cfg
from yolo_pose.detector import RatDetector

if len(sys.argv) > 1:
    video_name = sys.argv[1]
    paths.video_source = paths.video_dir / video_name
    paths.output_video = paths.detect_dir / (Path(video_name).stem + "_resultado.mp4")

paths.check_dirs()
print(f"[INFO] Video: {paths.video_source}")
print(f"[INFO] Salida: {paths.output_video}")

detector = RatDetector(detect_cfg, show_skeleton=False, show_preview=False, dual_output=True)
detector.run()

# Copiar CSV al historico con fecha y nombre descriptivo
csv_src  = paths.output_video.with_suffix(".csv")
hist_dir = paths.history_dir
hist_dir.mkdir(parents=True, exist_ok=True)

if csv_src.exists():
    ts       = datetime.now().strftime("%Y%m%d_%H%M")
    stem     = paths.video_source.stem
    hist_dst = hist_dir / f"{stem}_{ts}.csv"
    shutil.copy(str(csv_src), str(hist_dst))
    print(f"[Historial] CSV copiado a: {hist_dst.name}")
