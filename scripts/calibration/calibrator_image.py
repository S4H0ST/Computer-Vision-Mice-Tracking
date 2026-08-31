"""
Calibrador visual sobre imagen estática.

Permite marcar manualmente:
  1. BORDE EXTERIOR — 2 esquinas opuestas (rectángulo rojo)
  2. BORDE INTERIOR — 2 esquinas opuestas (rectángulo azul)
  3. AGUJEROS       — 4 centros (círculos verdes)

Guarda coords.json en datasets/ con: exterior, interior, holes,
hole_radius, limits_inner y limits_outer.

Uso:
    python scripts/tools/calibrator_image.py
Controles:
    Clic izquierdo — añadir punto
    S              — guardar y salir (solo cuando hay 8 puntos)
    R              — resetear todos los puntos
    Q              — salir sin guardar
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path

# ── Rutas relativas a la raíz del proyecto ───────────────────────────────── #
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

IMAGE_PATH  = PROJECT_ROOT / "media_original" / "cajaBordes.jpg"
OUTPUT_JSON = PROJECT_ROOT / "datasets" / "coords.json"

HOLE_RADIUS   = 20
DISPLAY_WIDTH = 1000   # ancho de la imagen mostrada en pantalla (px)
WIN_NAME      = "CALIBRADOR"

# Colores BGR
RED   = (0,   0,   255)
BLUE  = (255, 0,   0)
GREEN = (0,   255, 0)
WHITE = (255, 255, 255)
BLACK = (0,   0,   0)
GRAY  = (180, 180, 180)


class ImageCalibrator:
    def __init__(self, image_path: Path, output_json: Path):
        self.image_path  = image_path
        self.output_json = output_json

        self.exterior: list = []   # 2 puntos en coordenadas ORIGINALES
        self.interior: list = []   # 2 puntos en coordenadas ORIGINALES
        self.holes:    list = []   # 4 puntos en coordenadas ORIGINALES

        self.img_raw:     np.ndarray = None   # imagen original sin tocar
        self.scale_x: float = 1.0             # factor original→display
        self.scale_y: float = 1.0

    # ── Conversión de coordenadas display → original ─────────────────── #
    def _to_orig(self, x: int, y: int):
        return int(round(x * self.scale_x)), int(round(y * self.scale_y))

    # ── Conversión de coordenadas original → display ─────────────────── #
    def _to_disp(self, x: int, y: int):
        return int(round(x / self.scale_x)), int(round(y / self.scale_y))

    # ── Callback del ratón — coordenadas ya en espacio display ───────── #
    def _click(self, event, x, y, flags, params):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Convertir a coordenadas de imagen original
        ox, oy = self._to_orig(x, y)

        if len(self.exterior) < 2:
            self.exterior.append([ox, oy])
        elif len(self.interior) < 2:
            self.interior.append([ox, oy])
        elif len(self.holes) < 4:
            self.holes.append([ox, oy])

        self._refresh()

    # ── Redibujar la ventana ─────────────────────────────────────────── #
    def _refresh(self):
        if self.img_raw is None:
            return

        # 1. Dibujar geometría sobre la imagen ORIGINAL (coordenadas reales)
        img = self.img_raw.copy()

        for pt in self.exterior:
            cv2.circle(img, tuple(pt), max(6, int(6 * self.scale_x)), RED, -1)
        if len(self.exterior) == 2:
            cv2.rectangle(img, tuple(self.exterior[0]), tuple(self.exterior[1]), RED, max(2, int(2 * self.scale_x)))

        for pt in self.interior:
            cv2.circle(img, tuple(pt), max(6, int(6 * self.scale_x)), BLUE, -1)
        if len(self.interior) == 2:
            cv2.rectangle(img, tuple(self.interior[0]), tuple(self.interior[1]), BLUE, max(2, int(2 * self.scale_x)))

        for pt in self.holes:
            cv2.circle(img, tuple(pt), max(5, int(5 * self.scale_x)), GREEN, -1)
            cv2.circle(img, tuple(pt), max(HOLE_RADIUS, int(HOLE_RADIUS * self.scale_x)), GREEN, max(2, int(2 * self.scale_x)))

        # 2. Redimensionar al tamaño display (DISPLAY_WIDTH × display_h)
        h_orig, w_orig = img.shape[:2]
        display_h = int(DISPLAY_WIDTH * h_orig / w_orig)
        disp = cv2.resize(img, (DISPLAY_WIDTH, display_h))

        # 3. Dibujar texto sobre la imagen DISPLAY (tamaño fijo, independiente del original)
        self._draw_header(disp)

        cv2.imshow(WIN_NAME, disp)

    def _draw_header(self, disp: np.ndarray):
        """Banner de instrucciones encima de la imagen (tamaño fijo en pixels display)."""
        total = len(self.exterior) + len(self.interior) + len(self.holes)

        if total < 2:
            msg   = f"PASO 1/3 — BORDE EXTERIOR (pared superior): [{len(self.exterior)}/2] clics"
            color = RED
        elif total < 4:
            msg   = f"PASO 2/3 — BORDE INTERIOR (nivel del suelo): [{len(self.interior)}/2] clics"
            color = BLUE
        elif total < 8:
            msg   = f"PASO 3/3 — AGUJEROS (centra el clic en cada uno): [{len(self.holes)}/4] clics"
            color = GREEN
        else:
            msg   = "Completo — pulsa  S  para guardar   |   R  para repetir"
            color = (0, 220, 0)

        hint = "  R = repetir     S = guardar     Q = salir"

        # Fondo negro semitransparente en la franja superior
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, 0), (disp.shape[1], 48), BLACK, -1)
        cv2.addWeighted(overlay, 0.75, disp, 0.25, 0, disp)

        # Texto principal (fuente pequeña, escala 0.48)
        cv2.putText(disp, msg,  (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        # Texto secundario (escala 0.38)
        cv2.putText(disp, hint, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY,  1, cv2.LINE_AA)

    # ── Guardar JSON ─────────────────────────────────────────────────── #
    def _save(self):
        e = self.exterior
        i = self.interior
        data = {
            "exterior": e,
            "interior": i,
            "holes":    self.holes,
            "hole_radius": HOLE_RADIUS,
            "limits_inner": {
                "x_min": min(i[0][0], i[1][0]),
                "x_max": max(i[0][0], i[1][0]),
                "y_min": min(i[0][1], i[1][1]),
                "y_max": max(i[0][1], i[1][1]),
            },
            "limits_outer": {
                "x_min": min(e[0][0], e[1][0]),
                "x_max": max(e[0][0], e[1][0]),
                "y_min": min(e[0][1], e[1][1]),
                "y_max": max(e[0][1], e[1][1]),
            },
        }
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_json, "w") as f:
            json.dump(data, f, indent=4)
        print(f"\n[OK] coords.json guardado en: {self.output_json}")
        print(f"     Exterior : {e}")
        print(f"     Interior : {i}")
        print(f"     Agujeros : {self.holes}")

    def _reset(self):
        self.exterior = []
        self.interior = []
        self.holes    = []
        print("[R] Puntos reseteados.")
        self._refresh()

    # ── Bucle principal ──────────────────────────────────────────────── #
    def run(self):
        if not self.image_path.exists():
            print(f"[!] Imagen no encontrada: {self.image_path}")
            return

        self.img_raw = cv2.imread(str(self.image_path))
        if self.img_raw is None:
            print(f"[!] OpenCV no pudo cargar la imagen: {self.image_path}")
            return

        h_orig, w_orig = self.img_raw.shape[:2]
        display_h      = int(DISPLAY_WIDTH * h_orig / w_orig)

        # Factores de escala: display → original
        self.scale_x = w_orig / DISPLAY_WIDTH
        self.scale_y = h_orig / display_h

        # WINDOW_AUTOSIZE: la ventana se ajusta exactamente a la imagen mostrada.
        # Los eventos del ratón están en el mismo espacio de coordenadas que imshow.
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN_NAME, self._click)

        print(f"\n=== CALIBRADOR ({self.image_path.name}) ===")
        print(f"  Imagen original: {w_orig}x{h_orig} px  |  Mostrada: {DISPLAY_WIDTH}x{display_h} px")
        print("  Clic izq: añadir punto  |  S: guardar  |  R: resetear  |  Q: salir\n")

        self._refresh()

        while True:
            if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(20) & 0xFF

            if key in (ord('q'), ord('Q')):
                print("[Q] Saliendo sin guardar.")
                break
            elif key in (ord('s'), ord('S')):
                total = len(self.exterior) + len(self.interior) + len(self.holes)
                if total == 8:
                    self._save()
                    break
                else:
                    print(f"[!] Faltan {8 - total} puntos para completar la calibración.")
            elif key in (ord('r'), ord('R')):
                self._reset()

        cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    cal = ImageCalibrator(IMAGE_PATH, OUTPUT_JSON)
    cal.run()
