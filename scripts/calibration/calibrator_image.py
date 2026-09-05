"""
Calibrador visual de zonas sobre una imagen estatica.

Permite marcar manualmente:
  1. BORDE EXTERIOR - 2 esquinas opuestas (rectangulo rojo)
  2. BORDE INTERIOR - 2 esquinas opuestas (rectangulo azul)
  3. AGUJEROS       - 4 centros (circulos verdes)

Guarda coords.json en outputs/calibration/ con: exterior, interior, holes,
hole_radius, limits_inner y limits_outer.

Clases:
    ImageCalibrator - calibrador sobre imagen estatica (p.ej. primer frame de video).

Controles de la ventana:
    Clic izquierdo - anadir punto
    S              - guardar y salir (solo cuando hay 8 puntos)
    R              - resetear todos los puntos
    Q              - salir sin guardar
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path

# Ajustar sys.path para poder ejecutar el modulo directamente desde la raiz del proyecto
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

IMAGE_PATH  = PROJECT_ROOT / "media_original" / "cajaBordes.jpg"
OUTPUT_JSON = PROJECT_ROOT / "outputs" / "calibration" / "coords.json"

HOLE_RADIUS:   int = 15
DISPLAY_WIDTH: int = 1000   # ancho de la imagen mostrada en pantalla (px)
WIN_NAME:      str = "CALIBRADOR"

# Colores BGR (OpenCV usa BGR, no RGB)
RED   = (0,   0,   255)
BLUE  = (255, 0,   0)
GREEN = (0,   255, 0)
WHITE = (255, 255, 255)
BLACK = (0,   0,   0)
GRAY  = (210, 210, 210)


class ImageCalibrator:
    """
    Calibrador interactivo sobre imagen estatica.

    Escala la imagen a DISPLAY_WIDTH para mostrarla en pantalla, pero guarda
    todos los puntos en coordenadas de la imagen ORIGINAL para que
    SpatialAnalyzer pueda usarlos directamente contra los frames del video.
    """

    def __init__(self, image_path: Path, output_json: Path) -> None:
        self.image_path  = image_path
        self.output_json = output_json

        self.exterior: list = []   # 2 puntos en coordenadas originales
        self.interior: list = []   # 2 puntos en coordenadas originales
        self.holes:    list = []   # 4 puntos en coordenadas originales

        self.img_raw:  np.ndarray | None = None  # imagen original sin tocar
        self.scale_x:  float = 1.0               # factor display -> original (ancho)
        self.scale_y:  float = 1.0               # factor display -> original (alto)

    def _to_orig(self, x: int, y: int) -> tuple[int, int]:
        """Convierte coordenadas de pantalla (display) a coordenadas de la imagen original."""
        return int(round(x * self.scale_x)), int(round(y * self.scale_y))

    def _to_disp(self, x: int, y: int) -> tuple[int, int]:
        """Convierte coordenadas originales a coordenadas de pantalla (para dibujar)."""
        return int(round(x / self.scale_x)), int(round(y / self.scale_y))

    def _click(self, event: int, x: int, y: int, flags, params) -> None:
        """
        Callback del raton: recibe clics en espacio display, los convierte
        a coordenadas originales y los almacena en la lista correspondiente.
        """
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Convertir de pantalla a imagen original antes de guardar
        ox, oy = self._to_orig(x, y)

        if len(self.exterior) < 2:
            self.exterior.append([ox, oy])
        elif len(self.interior) < 2:
            self.interior.append([ox, oy])
        elif len(self.holes) < 4:
            self.holes.append([ox, oy])

        self._refresh()

    def _draw_grid(self, img: np.ndarray, divisions: int = 12) -> None:
        """
        Dibuja una cuadricula semitransparente sobre img para ayudar a
        alinear los puntos con los bordes fisicos de la caja.

        Se dibuja sobre la imagen de pantalla (ya escalada) para que las
        lineas tengan siempre el mismo grosor visual.
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
        Redibuja la ventana completa:
          1. Dibuja la geometria sobre la imagen original (coordenadas reales).
          2. Escala al tamano display.
          3. Dibuja la cuadricula sobre el display.
          4. Dibuja el banner de instrucciones.
        """
        if self.img_raw is None:
            return

        # 1. Dibujar sobre la imagen original (coordenadas reales)
        img = self.img_raw.copy()

        for pt in self.exterior:
            cv2.circle(img, tuple(pt), max(6, int(6 * self.scale_x)), RED, -1)
        if len(self.exterior) == 2:
            cv2.rectangle(img, tuple(self.exterior[0]), tuple(self.exterior[1]),
                          RED, max(2, int(2 * self.scale_x)))

        for pt in self.interior:
            cv2.circle(img, tuple(pt), max(6, int(6 * self.scale_x)), BLUE, -1)
        if len(self.interior) == 2:
            cv2.rectangle(img, tuple(self.interior[0]), tuple(self.interior[1]),
                          BLUE, max(2, int(2 * self.scale_x)))

        for pt in self.holes:
            cv2.circle(img, tuple(pt), max(5, int(5 * self.scale_x)), GREEN, -1)
            cv2.circle(img, tuple(pt),
                       max(HOLE_RADIUS, int(HOLE_RADIUS * self.scale_x)),
                       GREEN, max(2, int(2 * self.scale_x)))

        # 2. Escalar al tamano display para mostrar en pantalla
        h_orig, w_orig = img.shape[:2]
        display_h = int(DISPLAY_WIDTH * h_orig / w_orig)
        disp = cv2.resize(img, (DISPLAY_WIDTH, display_h))

        # 3. Cuadricula sobre la imagen ya escalada (lineas a tamano fijo en pantalla)
        self._draw_grid(disp)

        # 4. Banner de instrucciones en la franja inferior
        self._draw_header(disp)

        cv2.imshow(WIN_NAME, disp)

    def _draw_header(self, disp: np.ndarray) -> None:
        """Banner de instrucciones en la parte inferior de la imagen de pantalla."""
        total = len(self.exterior) + len(self.interior) + len(self.holes)
        h = disp.shape[0]

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
            msg   = "Completo - pulsa  S  para guardar   |   R  para repetir"
            color = (0, 220, 0)

        hint = "  R = resetear     S = guardar     Q = salir sin guardar"

        # Fondo semitransparente en la franja inferior para que el texto sea legible
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, h - 50), (disp.shape[1], h), BLACK, -1)
        cv2.addWeighted(overlay, 0.75, disp, 0.25, 0, disp)

        cv2.putText(disp, msg,  (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        cv2.putText(disp, hint, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY,  1, cv2.LINE_AA)

    def _save(self) -> None:
        """
        Serializa los puntos marcados a output_json.
        Los limites (limits_inner/outer) pre-calculan min/max para que
        SpatialAnalyzer compruebe pertenencia a zona con una comparacion simple.
        """
        e = self.exterior
        i = self.interior
        data = {
            "exterior":    e,
            "interior":    i,
            "holes":       self.holes,
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

    def _reset(self) -> None:
        """Borra todos los puntos marcados y actualiza la ventana."""
        self.exterior = []
        self.interior = []
        self.holes    = []
        print("[R] Puntos reseteados.")
        self._refresh()

    def run(self) -> None:
        """
        Punto de entrada: carga la imagen, calcula los factores de escala
        y lanza el bucle interactivo.
        """
        if not self.image_path.exists():
            print(f"[!] Imagen no encontrada: {self.image_path}")
            return

        self.img_raw = cv2.imread(str(self.image_path))
        if self.img_raw is None:
            print(f"[!] OpenCV no pudo cargar la imagen: {self.image_path}")
            return

        h_orig, w_orig = self.img_raw.shape[:2]
        display_h = int(DISPLAY_WIDTH * h_orig / w_orig)

        # Factores de escala: un clic en pantalla a (x, y) corresponde a
        # (x * scale_x, y * scale_y) en la imagen original
        self.scale_x = w_orig / DISPLAY_WIDTH
        self.scale_y = h_orig / display_h

        # WINDOW_AUTOSIZE: la ventana se ajusta exactamente al tamano de la imagen
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN_NAME, self._click)

        print(f"\n=== CALIBRADOR ({self.image_path.name}) ===")
        print(f"  Imagen original: {w_orig}x{h_orig} px  |  Mostrada: {DISPLAY_WIDTH}x{display_h} px")
        print("  Clic izq: anadir punto  |  S: guardar  |  R: resetear  |  Q: salir\n")

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
                    print(f"[!] Faltan {8 - total} puntos para completar la calibracion.")
            elif key in (ord('r'), ord('R')):
                self._reset()

        cv2.destroyAllWindows()


# -- Entry point: ejecutar directamente para calibrar la imagen estatica de la caja --
if __name__ == "__main__":
    cal = ImageCalibrator(IMAGE_PATH, OUTPUT_JSON)
    cal.run()
