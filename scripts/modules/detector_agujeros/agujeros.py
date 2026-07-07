import json
import numpy as np
from pathlib import Path


class SpatialAnalyzer:
    """
    Analiza la posición espacial del ratón respecto a la caja y los agujeros
    usando los datos de calibración de ZoneCalibrator (coords.json).

    Detecta tres condiciones derivadas de 'rat_horizontal':
      - head_dipping : snout dentro del radio de un agujero
      - sniffing     : snout cerca de la pared interior pero sin estar en agujero
      - (walking/immobile se derivan de velocidad en detector.py)
    """

    # Snout debe estar dentro de este múltiplo del radio para contar como dipping.
    # 1.0 = exactamente en el borde del agujero; <1 = más exigente.
    DIPPING_RATIO = 0.9

    # Margen en píxeles desde la pared interior para considerar sniffing.
    # Con interior 628→1260 (632px ancho) un margen de 55px ≈ 8.7% del ancho.
    WALL_SNIFF_MARGIN_PX = 55

    def __init__(self, config_path):
        self.holes = []
        self.hole_radius = 20
        self.inner_limits = None   # dict {x_min, x_max, y_min, y_max}
        self._load_config(config_path)

    def _load_config(self, path):
        p = Path(path)
        if not p.exists():
            print(f"[!] SpatialAnalyzer: coords.json no encontrado en {p}")
            return

        with open(p, "r") as f:
            data = json.load(f)

        self.holes       = [tuple(h) for h in data.get("holes", [])]
        self.hole_radius = data.get("hole_radius", 20)
        self.inner_limits = data.get("limits_inner", None)

        print(f"[OK] SpatialAnalyzer: {len(self.holes)} agujeros | "
              f"radio={self.hole_radius}px | interior={self.inner_limits}")

    # ------------------------------------------------------------------ #
    #  HEAD DIPPING                                                        #
    #  Condición: snout (keypoint 0 de YOLO Pose) sobre un agujero        #
    # ------------------------------------------------------------------ #
    def check_dipping(self, snout_xy) -> bool:
        """
        True si el snout está dentro del radio de algún agujero.

        Args:
            snout_xy: (x, y) en píxeles del frame — keypoint[0] de YOLO Pose.
        """
        if not self.holes or snout_xy is None:
            return False

        pt = np.array(snout_xy[:2], dtype=float)
        threshold = self.hole_radius * self.DIPPING_RATIO

        for hole in self.holes:
            if np.linalg.norm(pt - np.array(hole, dtype=float)) < threshold:
                return True
        return False

    # ------------------------------------------------------------------ #
    #  SNIFFING                                                            #
    #  Condición: postura horizontal + snout cerca de la pared interior   #
    # ------------------------------------------------------------------ #
    def check_sniffing(self, snout_xy) -> bool:
        """
        True si el snout está cerca de una pared interior (sniffing de borde).
        Debe llamarse SOLO cuando YOLO ya clasificó la rata como rat_horizontal
        y check_dipping() devolvió False.

        Args:
            snout_xy: (x, y) en píxeles del frame.
        """
        if self.inner_limits is None or snout_xy is None:
            return False

        x, y = float(snout_xy[0]), float(snout_xy[1])
        lim = self.inner_limits
        m   = self.WALL_SNIFF_MARGIN_PX

        inside = (lim["x_min"] <= x <= lim["x_max"] and
                  lim["y_min"] <= y <= lim["y_max"])

        near_wall = (
            x < lim["x_min"] + m or   # pared izquierda
            x > lim["x_max"] - m or   # pared derecha
            y < lim["y_min"] + m or   # pared superior
            y > lim["y_max"] - m      # pared inferior (front)
        )

        return inside and near_wall

    # ------------------------------------------------------------------ #
    #  UTILIDAD                                                            #
    # ------------------------------------------------------------------ #
    def is_inside_inner(self, point_xy) -> bool:
        """True si el punto está dentro del área interior (suelo transitable)."""
        if self.inner_limits is None:
            return True
        x, y = float(point_xy[0]), float(point_xy[1])
        lim = self.inner_limits
        return (lim["x_min"] <= x <= lim["x_max"] and
                lim["y_min"] <= y <= lim["y_max"])

    def nearest_hole_dist(self, point_xy) -> float:
        """Distancia en px al agujero más cercano. Útil para debug."""
        if not self.holes or point_xy is None:
            return float("inf")
        pt = np.array(point_xy[:2], dtype=float)
        return float(min(np.linalg.norm(pt - np.array(h)) for h in self.holes))
