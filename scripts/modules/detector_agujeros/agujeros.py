import json
import numpy as np
from pathlib import Path

class SpatialAnalyzer:
    def __init__(self, config_path):
        self.holes = []
        self.hole_radius = 20
        self._load_config(config_path)

    def _load_config(self, path):
        if Path(path).exists():
            with open(path, 'r') as f:
                data = json.load(f)
                self.holes = data.get('holes', [])
                self.hole_radius = data.get('hole_radius', 20)
            print(f"[OK] Cargados {len(self.holes)} agujeros para análisis espacial.")

    def check_dipping(self, head_box):
        """
        Devuelve True si el centro de la cabeza está dentro de un agujero
        """
        if not self.holes or head_box is None:
            return False

        x1, y1, x2, y2 = head_box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        head_point = np.array([cx, cy])

        for hole in self.holes:
            hole_point = np.array(hole)
            dist = np.linalg.norm(head_point - hole_point)
            if dist < self.hole_radius:
                return True
        return False