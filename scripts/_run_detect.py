"""Script temporal para detección sin ventana — se puede borrar después."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import paths, detect_cfg
from yolo_pose.detector import RatDetector

paths.check_dirs()
detector = RatDetector(detect_cfg, show_skeleton=False, show_preview=False)
detector.run()
