"""
Lanza la deteccion + estadisticas para testRata3, testRata4 y testRata5
sin interfaz grafica. Usa la calibracion actual (datasets/coords.json).

Uso:
    python run_detections.py
"""

import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from config.config import paths, detect_cfg
from detection.detector import RatDetector
from utils.stats_generator import StatsGenerator

VIDEOS = [
    PROJECT_ROOT / "media_original" / "videos" / "testRata3.mp4",
    PROJECT_ROOT / "media_original" / "videos" / "testRata4.mp4",
    PROJECT_ROOT / "media_original" / "videos" / "testRata5.mp4",
]

TODAY = datetime.now().strftime("%Y%m%d")


def run_video(video_path: Path) -> None:
    stem = video_path.stem
    run_folder = paths.detect_dir / f"{stem}_{TODAY}"
    run_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  VIDEO: {video_path.name}")
    print(f"  Salida: {run_folder}")
    print(f"{'='*50}")

    paths.video_source = video_path
    paths.output_video = run_folder / f"{stem}_{TODAY}.mp4"

    detector = RatDetector(detect_cfg, show_skeleton=False, show_preview=False, dual_output=True)
    detector.run()

    csv_path = paths.output_video.with_suffix(".csv")
    if csv_path.exists():
        print("\n[Stats] Generando estadisticas...")
        StatsGenerator(csv_path, coords_json=paths.coords_json).generate(run_folder / "stats")
        print(f"[OK] Stats guardadas en: {run_folder / 'stats'}")
    else:
        print(f"[!] CSV no encontrado: {csv_path}")


if __name__ == "__main__":
    paths.check_dirs()

    for video in VIDEOS:
        if not video.exists():
            print(f"[!] Video no encontrado, saltando: {video}")
            continue
        run_video(video)

    print("\n[OK] Todos los videos procesados.")
