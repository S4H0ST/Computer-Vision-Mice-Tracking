"""
Calibrador visual de zonas sobre el primer frame de un video.

Permite marcar manualmente:
  1. BORDE EXTERIOR - 2 esquinas opuestas (rectangulo rojo)
  2. BORDE INTERIOR - 2 esquinas opuestas (rectangulo azul)
  3. AGUJEROS       - 4 centros (circulos verdes)

Guarda coords.json con: exterior, interior, holes, hole_radius,
limits_inner y limits_outer.

Clases:
    ZoneCalibrator - calibrador interactivo sobre primer frame de video.

Controles de la ventana:
    Clic izquierdo - anadir punto
    R              - resetear todos los puntos
    Q              - guardar (si completo) y salir
"""

import cv2
import json
import numpy as np
from pathlib import Path
from config.interfaces import BaseModule
from config.config import paths


HOLE_RADIUS:   int = 20
DISPLAY_WIDTH: int = 1000   # ancho maximo de la ventana en pantalla (px)
WIN_NAME:      str = "CALIBRADOR"

RED   = (0,   0,   255)
BLUE  = (255, 0,   0)
GREEN = (0,   255, 0)
GRAY  = (210, 210, 210)
BLACK = (0,   0,   0)


class ZoneCalibrator(BaseModule):
    """
    Calibrador interactivo que abre el primer frame de un video y permite
    al usuario marcar: borde exterior, borde interior y 4 agujeros.

    Los puntos se guardan en coordenadas de la imagen ORIGINAL, no de la
    pantalla, para que SpatialAnalyzer pueda usarlos directamente contra
    los frames del video sin conversion adicional.
    """

    def __init__(self, video_path: Path) -> None:
        self.video_path  = Path(video_path)
        self.output_json = paths.coords_json

        self.exterior: list = []   # 2 puntos: esquinas opuestas del rectangulo exterior
        self.interior: list = []   # 2 puntos: esquinas opuestas del rectangulo interior
        self.holes:    list = []   # 4 puntos: centros de los agujeros

        self.img_raw: np.ndarray | None = None  # frame original sin modificar
        self.scale_x: float = 1.0               # factor display -> original (ancho)
        self.scale_y: float = 1.0               # factor display -> original (alto)

    def _to_orig(self, x: int, y: int) -> tuple[int, int]:
        """Convierte coordenadas de pantalla (display) a coordenadas del frame original."""
        return int(round(x * self.scale_x)), int(round(y * self.scale_y))

    def _click_event(self, event: int, x: int, y: int, flags, params) -> None:
        """
        Callback del raton. Recibe clics en coordenadas de pantalla,
        los convierte a coordenadas originales y los almacena.
        """
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Convertir de espacio display a espacio original del video
        ox, oy = self._to_orig(x, y)

        if len(self.exterior) < 2:
            self.exterior.append([ox, oy])
            print(f"[+] Exterior punto {len(self.exterior)}: ({ox}, {oy})")
        elif len(self.interior) < 2:
            self.interior.append([ox, oy])
            print(f"[+] Interior punto {len(self.interior)}: ({ox}, {oy})")
        elif len(self.holes) < 4:
            self.holes.append([ox, oy])
            print(f"[+] Agujero {len(self.holes)}: ({ox}, {oy})")

        self._refresh()

    def _draw_grid(self, img: np.ndarray, divisions: int = 12) -> None:
        """
        Dibuja una cuadricula semitransparente sobre img para facilitar
        la alineacion de los puntos con los bordes fisicos de la caja.

        divisions: numero de celdas por lado (12 -> cuadricula 12x12).
        alpha=0.15: la cuadricula es muy sutil para no tapar al raton.
        """
        h, w = img.shape[:2]
        overlay = img.copy()
        for i in range(1, divisions):
            x = int(w * i / divisions)
            cv2.line(overlay, (x, 0), (x, h), GRAY, 1)
        for j in range(1, divisions):
            y = int(h * j / divisions)
            cv2.line(overlay, (0, y), (w, y), GRAY, 1)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

    def _refresh(self) -> None:
        """
        Redibuja la ventana completa: imagen original redimensionada +
        cuadricula + rectangulos/circulos de los puntos marcados + banner.
        """
        if self.img_raw is None:
            return

        # 1. Partir de la imagen original y escalarla al tamano de pantalla
        h_orig, w_orig = self.img_raw.shape[:2]
        display_h = int(DISPLAY_WIDTH * h_orig / w_orig)
        img = cv2.resize(self.img_raw, (DISPLAY_WIDTH, display_h))

        # 2. Cuadricula sutil (se dibuja sobre la imagen ya escalada)
        self._draw_grid(img)

        # 3. Para dibujar los puntos en pantalla hay que escalarlos de vuelta
        #    (los puntos estan guardados en coordenadas originales)
        def to_disp(p):
            return (int(round(p[0] / self.scale_x)), int(round(p[1] / self.scale_y)))

        # Borde exterior (rojo)
        for pt in self.exterior:
            cv2.circle(img, to_disp(pt), 6, RED, -1)
        if len(self.exterior) == 2:
            cv2.rectangle(img, to_disp(self.exterior[0]),
                          to_disp(self.exterior[1]), RED, 2)

        # Borde interior (azul)
        for pt in self.interior:
            cv2.circle(img, to_disp(pt), 6, BLUE, -1)
        if len(self.interior) == 2:
            cv2.rectangle(img, to_disp(self.interior[0]),
                          to_disp(self.interior[1]), BLUE, 2)

        # Agujeros (verde) - el radio se dibuja a escala de pantalla
        r_disp = max(5, int(HOLE_RADIUS / self.scale_x))
        for pt in self.holes:
            cv2.circle(img, to_disp(pt), 5, GREEN, -1)
            cv2.circle(img, to_disp(pt), r_disp, GREEN, 2)

        # 4. Banner de instrucciones en la franja inferior
        self._draw_header(img)

        cv2.imshow(WIN_NAME, img)

    def _draw_header(self, img: np.ndarray) -> None:
        """Dibuja el banner de instrucciones en la parte inferior de img."""
        total = len(self.exterior) + len(self.interior) + len(self.holes)
        h = img.shape[0]

        if total < 2:
            msg   = f"PASO 1/3 - BORDE EXTERIOR (pared): [{len(self.exterior)}/2] clics"
            color = RED
        elif total < 4:
            msg   = f"PASO 2/3 - BORDE INTERIOR (suelo): [{len(self.interior)}/2] clics"
            color = BLUE
        elif total < 8:
            msg   = f"PASO 3/3 - AGUJEROS (centra el clic): [{len(self.holes)}/4] clics"
            color = GREEN
        else:
            msg   = "Completo - pulsa  Q  para guardar y salir   |   R  para repetir"
            color = (0, 220, 0)

        hint = "  R = resetear     Q = guardar y salir"

        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - 50), (img.shape[1], h), BLACK, -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        cv2.putText(img, msg,  (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        cv2.putText(img, hint, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY,  1, cv2.LINE_AA)

    def _save(self) -> None:
        """
        Serializa los puntos marcados a coords.json.
        Los limites (limits_inner/outer) son el formato rapido que usa
        SpatialAnalyzer para comprobar si un punto esta dentro de un rectangulo.
        """
        e = self.exterior
        i = self.interior
        data = {
            "exterior":    e,
            "interior":    i,
            "holes":       self.holes,
            "hole_radius": HOLE_RADIUS,
            "limits_inner": {
                "x_min": min(i[0][0], i[1][0]), "x_max": max(i[0][0], i[1][0]),
                "y_min": min(i[0][1], i[1][1]), "y_max": max(i[0][1], i[1][1]),
            },
            "limits_outer": {
                "x_min": min(e[0][0], e[1][0]), "x_max": max(e[0][0], e[1][0]),
                "y_min": min(e[0][1], e[1][1]), "y_max": max(e[0][1], e[1][1]),
            },
        }
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_json, "w") as f:
            json.dump(data, f, indent=4)
        print(f"\n[OK] Calibracion guardada en: {self.output_json}")
        print(f"     Exterior: {e}")
        print(f"     Interior: {i}")
        print(f"     Agujeros: {self.holes}")

    def _reset(self) -> None:
        """Borra todos los puntos marcados y actualiza la ventana."""
        self.exterior = []
        self.interior = []
        self.holes    = []
        print("[R] Puntos reseteados.")
        self._refresh()

    def run(self) -> None:
        """
        Punto de entrada: abre el video, lee el primer frame,
        lanza la ventana interactiva y espera a que el usuario termine.
        """
        if not self.video_path.exists():
            print(f"[!] Video no encontrado: {self.video_path}")
            return

        # Solo necesitamos el primer frame, no reproducir el video completo
        cap = cv2.VideoCapture(str(self.video_path))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("[!] No se pudo leer el primer frame del video.")
            return

        h_orig, w_orig = frame.shape[:2]
        display_h = int(DISPLAY_WIDTH * h_orig / w_orig)

        self.img_raw = frame.copy()
        # Factores para convertir clics en pantalla a coordenadas del frame original
        self.scale_x = w_orig / DISPLAY_WIDTH
        self.scale_y = h_orig / display_h

        cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN_NAME, self._click_event)

        print(f"\n=== CALIBRADOR ({self.video_path.name}) ===")
        print(f"  Frame original: {w_orig}x{h_orig} px  |  Mostrado: {DISPLAY_WIDTH}x{display_h} px")
        print("  Clic: anadir punto  |  R: resetear  |  Q: guardar y salir\n")

        self._refresh()

        while True:
            if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(20) & 0xFF

            if key in (ord('q'), ord('Q')):
                total = len(self.exterior) + len(self.interior) + len(self.holes)
                if total == 8:
                    self._save()
                else:
                    print(f"[!] Calibracion incompleta ({total}/8 puntos). Saliendo sin guardar.")
                break
            elif key in (ord('r'), ord('R')):
                self._reset()

        cv2.destroyAllWindows()
