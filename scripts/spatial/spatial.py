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

    DIPPING_RATIO: float = 0.9    # snout dentro del 90% del radio -> head_dipping

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

    def draw_zones(self, img: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        """
        Dibuja sobre img las zonas calibradas.
        - Rectangulo rojo = borde EXTERIOR (paredes de la caja)
        - Rectangulo azul = borde INTERIOR (suelo transitable)
        - Circulo verde   = zona head_dipping (radio activo del agujero)
        """
        import cv2
        overlay = img.copy()

        if self.outer_limits is not None:
            lim = self.outer_limits
            cv2.rectangle(overlay,
                          (lim["x_min"], lim["y_min"]),
                          (lim["x_max"], lim["y_max"]),
                          (0, 0, 220), 2)

        if self.inner_limits is not None:
            lim = self.inner_limits
            cv2.rectangle(overlay,
                          (lim["x_min"], lim["y_min"]),
                          (lim["x_max"], lim["y_max"]),
                          (255, 100, 0), 2)

        for hx, hy in self.holes:
            cv2.circle(overlay, (int(hx), int(hy)),
                       int(self.hole_radius * self.DIPPING_RATIO), (0, 255, 0), 1)

        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        return img
