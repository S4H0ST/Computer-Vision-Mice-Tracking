"""
Logica espacial basada en la calibracion de zonas (coords.json).

Clases:
    SpatialAnalyzer — analiza la posicion del raton respecto a las paredes y los agujeros
                      de la caja, usando los datos generados por ZoneCalibrator.
"""

import json
import numpy as np
from pathlib import Path


class SpatialAnalyzer:
    """
    Analiza la posicion espacial del raton respecto a la caja y los agujeros
    usando los datos de calibracion de ZoneCalibrator (coords.json).
    """

    DIPPING_RATIO: float        = 0.9    # snout dentro del 90% del radio -> head_dipping
    HOLE_SNIFF_FACTOR: float    = 2.5    # snout en el anillo exterior (hasta 2.5x el radio) -> sniffing
    WALL_SNIFF_MARGIN_PX: int   = 55     # px desde la pared interior -> sniffing

    def __init__(self, config_path) -> None:
        self.holes: list[tuple] = []
        self.hole_radius: int = 20
        self.inner_limits: dict | None = None
        self.outer_limits: dict | None = None
        self._calibrated: bool = False
        self._load_config(config_path)

    def _load_config(self, path) -> None:
        """Carga los datos de calibracion desde coords.json."""
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
        Devuelve True si la calibracion es compatible con este video.
        Verifica que al menos un agujero este dentro de las dimensiones del frame.
        """
        if not self._calibrated:
            return False
        for hx, hy in self.holes:
            if 0 <= hx < img_w and 0 <= hy < img_h:
                return True
        return False

    def check_dipping(self, snout_xy) -> bool:
        """Devuelve True si el snout esta dentro del radio de alguno de los agujeros."""
        if not self.holes or snout_xy is None:
            return False
        pt = np.array(snout_xy[:2], dtype=float)
        threshold = self.hole_radius * self.DIPPING_RATIO
        for hole in self.holes:
            if np.linalg.norm(pt - np.array(hole, dtype=float)) < threshold:
                return True
        return False

    def check_sniffing_wall(self, snout_point, margin: int = 30) -> bool:
        """
        Devuelve True si el snout esta dentro del area interior y a menos de
        'margin' pixeles de cualquiera de los cuatro lados del rectangulo interior.

        snout_point : array-like con (x, y) en pixeles de la imagen.
        margin      : distancia maxima en pixeles desde la pared para activar.
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

    def check_sniffing(self, snout_xy) -> bool:
        """
        Devuelve True si el snout esta cerca de una pared interior
        o en el anillo exterior de un agujero (cerca pero sin estar dentro).

        Dos condiciones equivalentes (la primera que se cumpla):
          a) Snout cerca de la pared interior.
          b) Snout en el anillo exterior al agujero.
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

        # b) Anillo exterior al agujero
        if self.holes:
            pt      = np.array(snout_xy[:2], dtype=float)
            inner_r = self.hole_radius * self.DIPPING_RATIO
            outer_r = self.hole_radius * self.HOLE_SNIFF_FACTOR
            for hole in self.holes:
                d = np.linalg.norm(pt - np.array(hole, dtype=float))
                if inner_r <= d < outer_r:
                    return True

        return False

    def is_inside_inner(self, point_xy) -> bool:
        """Devuelve True si el punto esta dentro del area interior de la caja."""
        if self.inner_limits is None:
            return True
        x, y = float(point_xy[0]), float(point_xy[1])
        lim = self.inner_limits
        return (lim["x_min"] <= x <= lim["x_max"] and
                lim["y_min"] <= y <= lim["y_max"])

    def bbox_entirely_inside(self, x1: float, y1: float, x2: float, y2: float,
                             margin: int = 15) -> bool:
        """
        Devuelve True si el bounding box esta dentro del area interior.
        El parametro margin (px) exige que el bbox penetre al menos esa cantidad
        en la zona de pared antes de considerarse climbing: evita falsos positivos
        cuando el raton camina cerca del borde interior sin trepar.
        """
        if self.inner_limits is None:
            return True
        lim = self.inner_limits
        return (x1 >= lim["x_min"] - margin and x2 <= lim["x_max"] + margin and
                y1 >= lim["y_min"] - margin and y2 <= lim["y_max"] + margin)

    def nearest_hole_dist(self, point_xy) -> float:
        """Devuelve la distancia en pixeles al agujero mas cercano."""
        if not self.holes or point_xy is None:
            return float("inf")
        pt = np.array(point_xy[:2], dtype=float)
        return float(min(np.linalg.norm(pt - np.array(h)) for h in self.holes))

    def draw_zones(self, img: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        """
        Dibuja sobre img las zonas calibradas.
        - Rectangulo rojo  = borde EXTERIOR (paredes de la caja, limite superior)
        - Rectangulo azul  = borde INTERIOR (suelo, limite inferior de la pared)
        - Zona de pared    = franja entre ambos rectangulos
        - Circulo verde    = zona head_dipping (dentro del agujero)
        - Circulo cian     = zona sniffing alrededor del agujero
        """
        import cv2
        overlay = img.copy()

        # Borde exterior — rojo — limite superior de la zona de pared
        if self.outer_limits is not None:
            lim = self.outer_limits
            cv2.rectangle(overlay,
                          (lim["x_min"], lim["y_min"]),
                          (lim["x_max"], lim["y_max"]),
                          (0, 0, 220), 2)

        # Borde interior — azul — limite inferior de la zona de pared (suelo)
        if self.inner_limits is not None:
            lim = self.inner_limits
            cv2.rectangle(overlay,
                          (lim["x_min"], lim["y_min"]),
                          (lim["x_max"], lim["y_max"]),
                          (255, 100, 0), 2)

        # Circulo verde = zona dipping, circulo cian = zona sniffing
        for hx, hy in self.holes:
            cv2.circle(overlay, (int(hx), int(hy)),
                       int(self.hole_radius * self.DIPPING_RATIO), (0, 255, 0), 1)
            cv2.circle(overlay, (int(hx), int(hy)),
                       int(self.hole_radius * self.HOLE_SNIFF_FACTOR), (0, 220, 220), 1)

        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        return img
