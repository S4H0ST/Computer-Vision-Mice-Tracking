import json
import numpy as np
from pathlib import Path


class SpatialAnalyzer:
    """
    Analiza la posición espacial del ratón respecto a la caja y los agujeros
    usando los datos de calibración de ZoneCalibrator (coords.json).
    """

    DIPPING_RATIO        = 0.9   # snout dentro de 90% del radio → head_dipping
    HOLE_SNIFF_FACTOR    = 2.5   # snout dentro de 2.5× el radio (pero fuera) → sniffing cerca de agujero
    WALL_SNIFF_MARGIN_PX = 55    # px desde pared interior → sniffing

    def __init__(self, config_path):
        self.holes        = []
        self.hole_radius  = 20
        self.inner_limits = None
        self.outer_limits = None
        self._calibrated  = False   # True solo si el json cargó correctamente
        self._load_config(config_path)

    def _load_config(self, path):
        p = Path(path)
        if not p.exists():
            print(f"[!] SpatialAnalyzer: coords.json no encontrado en {p}")
            return

        with open(p, "r") as f:
            data = json.load(f)

        self.holes        = [tuple(h) for h in data.get("holes", [])]
        self.hole_radius  = data.get("hole_radius", 20)
        self.inner_limits = data.get("limits_inner", None)
        self.outer_limits = data.get("limits_outer", None)
        self._calibrated  = len(self.holes) == 4 and self.inner_limits is not None

        print(f"[OK] SpatialAnalyzer: {len(self.holes)} agujeros | "
              f"radio={self.hole_radius}px | interior={self.inner_limits}")

    def is_valid_for(self, img_w: int, img_h: int) -> bool:
        """
        True si la calibración pertenece a este video.
        Verifica que al menos un agujero esté dentro del frame.
        """
        if not self._calibrated:
            return False
        for hx, hy in self.holes:
            if 0 <= hx < img_w and 0 <= hy < img_h:
                return True
        return False

    # ------------------------------------------------------------------ #
    #  HEAD DIPPING                                                        #
    # ------------------------------------------------------------------ #
    def check_dipping(self, snout_xy) -> bool:
        """True si el snout está dentro del radio de algún agujero."""
        if not self.holes or snout_xy is None:
            return False
        pt = np.array(snout_xy[:2], dtype=float)
        threshold = self.hole_radius * self.DIPPING_RATIO
        for hole in self.holes:
            if np.linalg.norm(pt - np.array(hole, dtype=float)) < threshold:
                return True
        return False

    # ------------------------------------------------------------------ #
    #  SNIFFING — PARED INTERIOR                                          #
    # ------------------------------------------------------------------ #
    def check_sniffing_wall(self, snout_point, margin: int = 30) -> bool:
        """
        Devuelve True si el snout está DENTRO del área interior y a menos de
        `margin` píxeles de cualquiera de los cuatro lados del rectángulo interior.

        La rata está olfateando la pared cuando:
          · su hocico está dentro de la caja (inside_inner = True)
          · pero muy cerca del borde — a menos de `margin` px de la pared

        Parámetros
        ----------
        snout_point : array-like con (x, y) en píxeles de la imagen.
        margin      : distancia máxima en píxeles desde la pared para activar (default 30).
        """
        if snout_point is None or self.inner_limits is None:
            return False

        x, y = float(snout_point[0]), float(snout_point[1])
        lim  = self.inner_limits

        inside = (lim["x_min"] <= x <= lim["x_max"] and
                  lim["y_min"] <= y <= lim["y_max"])
        if not inside:
            return False

        near_wall = (x < lim["x_min"] + margin or
                     x > lim["x_max"] - margin or
                     y < lim["y_min"] + margin or
                     y > lim["y_max"] - margin)
        return near_wall

    # ------------------------------------------------------------------ #
    #  SNIFFING — COMBINADO (pared + anillo de agujero)                   #
    #  Dos condiciones equivalentes (la primera que se cumpla):           #
    #    a) Snout cerca de la pared interior                              #
    #    b) Snout en el anillo exterior al agujero (cerca pero no dentro) #
    # ------------------------------------------------------------------ #
    def check_sniffing(self, snout_xy) -> bool:
        """
        True si el snout está cerca de una pared interior
        O cerca del borde exterior de un agujero (sin estar dentro).
        """
        if snout_xy is None:
            return False

        # a) Cerca de pared interior
        if self.inner_limits is not None:
            x, y = float(snout_xy[0]), float(snout_xy[1])
            lim = self.inner_limits
            m   = self.WALL_SNIFF_MARGIN_PX
            inside    = (lim["x_min"] <= x <= lim["x_max"] and
                         lim["y_min"] <= y <= lim["y_max"])
            near_wall = (x < lim["x_min"] + m or x > lim["x_max"] - m or
                         y < lim["y_min"] + m or y > lim["y_max"] - m)
            if inside and near_wall:
                return True

        # b) Anillo exterior al agujero (sniffing al borde del agujero)
        if self.holes:
            pt          = np.array(snout_xy[:2], dtype=float)
            inner_r     = self.hole_radius * self.DIPPING_RATIO
            outer_r     = self.hole_radius * self.HOLE_SNIFF_FACTOR
            for hole in self.holes:
                d = np.linalg.norm(pt - np.array(hole, dtype=float))
                if inner_r <= d < outer_r:
                    return True

        return False

    # ------------------------------------------------------------------ #
    #  UTILIDAD                                                            #
    # ------------------------------------------------------------------ #
    def is_inside_inner(self, point_xy) -> bool:
        """True si el punto está dentro del área interior."""
        if self.inner_limits is None:
            return True
        x, y = float(point_xy[0]), float(point_xy[1])
        lim = self.inner_limits
        return (lim["x_min"] <= x <= lim["x_max"] and
                lim["y_min"] <= y <= lim["y_max"])

    def bbox_entirely_inside(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """
        True si el bounding box completo está dentro del área interior.
        Más robusto que is_inside_inner(centroid): si cualquier borde
        del bbox toca la zona de pared el método devuelve False,
        indicando que la rata puede estar realmente escalando.
        """
        if self.inner_limits is None:
            return True
        lim = self.inner_limits
        return (x1 >= lim["x_min"] and x2 <= lim["x_max"] and
                y1 >= lim["y_min"] and y2 <= lim["y_max"])

    def nearest_hole_dist(self, point_xy) -> float:
        """Distancia en px al agujero más cercano."""
        if not self.holes or point_xy is None:
            return float("inf")
        pt = np.array(point_xy[:2], dtype=float)
        return float(min(np.linalg.norm(pt - np.array(h)) for h in self.holes))

    def draw_zones(self, img, alpha: float = 0.35):
        """
        Dibuja sobre img las zonas calibradas (borde interior + agujeros).
        Se llama una vez por frame en detector.py para verificación visual.
        """
        import cv2
        overlay = img.copy()

        # Borde interior (azul semitransparente)
        if self.inner_limits is not None:
            lim = self.inner_limits
            cv2.rectangle(overlay,
                          (lim["x_min"], lim["y_min"]),
                          (lim["x_max"], lim["y_max"]),
                          (255, 100, 0), 2)

        # Agujeros (círculo verde = zona dipping, círculo amarillo = zona sniffing)
        for hx, hy in self.holes:
            cv2.circle(overlay, (int(hx), int(hy)),
                       int(self.hole_radius * self.DIPPING_RATIO), (0, 255, 0), 1)
            cv2.circle(overlay, (int(hx), int(hy)),
                       int(self.hole_radius * self.HOLE_SNIFF_FACTOR), (0, 220, 220), 1)

        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        return img
