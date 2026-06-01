import cv2
import json
import numpy as np
from pathlib import Path
from helpers.base import BaseModule
from helpers.configuracion import paths


class ZoneCalibrator(BaseModule):
    def __init__(self, video_path: Path):
        self.video_path = str(video_path)
        self.output_json = paths.coords_json

        # --- DATOS A RECOLECTAR ---
        self.rect_exterior = []  # 2 puntos (esquinas opuestas)
        self.rect_interior = []  # 2 puntos (esquinas opuestas)
        self.holes = []  # 4 puntos (centros)
        self.hole_radius = 20  # Radio visual

        self.img_raw = None  # Imagen original limpia
        self.img_display = None  # Imagen con dibujos

    def _click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # PASO 1: Borde Exterior (Paredes externas)
            if len(self.rect_exterior) < 2:
                self.rect_exterior.append((x, y))
                print(f"[+] Exterior Punto {len(self.rect_exterior)}: ({x}, {y})")

            # PASO 2: Borde Interior (Suelo transitable)
            elif len(self.rect_interior) < 2:
                self.rect_interior.append((x, y))
                print(f"[+] Interior Punto {len(self.rect_interior)}: ({x}, {y})")

            # PASO 3: Agujeros (Objetivos)
            elif len(self.holes) < 4:
                self.holes.append((x, y))
                print(f"[+] Agujero {len(self.holes)}: ({x}, {y})")

            # Actualizar dibujo inmediatamente
            self._refresh_display()

    def _refresh_display(self):
        """Redibuja todo en la imagen: instrucciones, rectángulos y puntos."""
        if self.img_raw is None: return

        # Reiniciamos la imagen (borrar dibujos anteriores)
        self.img_display = self.img_raw.copy()

        # --- DIBUJAR GEOMETRÍA ---

        # 1. Borde Exterior (ROJO)
        if len(self.rect_exterior) > 0:
            for pt in self.rect_exterior:
                cv2.circle(self.img_display, pt, 5, (0, 0, 255), -1)
            if len(self.rect_exterior) == 2:
                cv2.rectangle(self.img_display, self.rect_exterior[0], self.rect_exterior[1], (0, 0, 255), 2)

        # 2. Borde Interior (AZUL)
        if len(self.rect_interior) > 0:
            for pt in self.rect_interior:
                cv2.circle(self.img_display, pt, 5, (255, 0, 0), -1)
            if len(self.rect_interior) == 2:
                cv2.rectangle(self.img_display, self.rect_interior[0], self.rect_interior[1], (255, 0, 0), 2)

        # 3. Agujeros (VERDE)
        for pt in self.holes:
            cv2.circle(self.img_display, pt, 5, (0, 255, 0), -1)
            cv2.circle(self.img_display, pt, self.hole_radius, (0, 255, 0), 2)

        # --- DIBUJAR TEXTO DE INSTRUCCIONES ---
        # Fondo negro para el texto arriba
        cv2.rectangle(self.img_display, (0, 0), (800, 60), (0, 0, 0), -1)

        msg = "Completado. Pulsa 'Q' para Guardar y Salir."
        color = (0, 255, 0)  # Verde final

        if len(self.rect_exterior) < 2:
            msg = f"PASO 1: Marca 2 esquinas del BORDE SUPERIOR (Exterior) [{len(self.rect_exterior)}/2]"
            color = (0, 0, 255)  # Rojo
        elif len(self.rect_interior) < 2:
            msg = f"PASO 2: Marca 2 esquinas del BORDE INFERIOR (Suelo) [{len(self.rect_interior)}/2]"
            color = (255, 0, 0)  # Azul
        elif len(self.holes) < 4:
            msg = f"PASO 3: Marca el centro de los 4 AGUJEROS [{len(self.holes)}/4]"
            color = (0, 255, 255)  # Amarillo

        cv2.putText(self.img_display, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("CALIBRADOR", self.img_display)

    def _save_data(self):
        # Preparar estructura de datos útil
        data = {
            "exterior": self.rect_exterior,
            "interior": self.rect_interior,
            "holes": self.holes,
            "hole_radius": self.hole_radius
        }

        # Calcular límites simples (min/max) para uso rápido
        if len(self.rect_interior) == 2:
            x1, y1 = self.rect_interior[0]
            x2, y2 = self.rect_interior[1]
            data["limits_inner"] = {
                "x_min": min(x1, x2), "x_max": max(x1, x2),
                "y_min": min(y1, y2), "y_max": max(y1, y2)
            }

        with open(self.output_json, "w") as f:
            json.dump(data, f, indent=4)

        print(f"\n[OK] Datos de calibración guardados en: {self.output_json}")

    def run(self):
        if not Path(self.video_path).exists():
            print(f"[!] Error: No encuentro el video en {self.video_path}")
            return

        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("[!] Error leyendo frame del video.")
            return

        self.img_raw = frame.copy()

        # --- CAMBIO AQUÍ PARA REDIMENSIONAR LA VENTANA ---
        window_name = "CALIBRADOR"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Permite cambiar el tamaño

        # Obtenemos el tamaño original para mantener la proporción
        h, w = frame.shape[:2]
        aspect_ratio = w / h

        # Definimos un ancho fijo para portátiles (ej. 1000 píxeles)
        # y calculamos el alto proporcional
        new_w = 1000
        new_h = int(new_w / aspect_ratio)

        cv2.resizeWindow(window_name, new_w, new_h)
        # -------------------------------------------------

        self._refresh_display()
        cv2.setMouseCallback(window_name, self._click_event)

        print("\n--- INICIANDO CALIBRACIÓN VISUAL ---")
        print(f"Ventana ajustada a {new_w}x{new_h} para mejor visualización.")

        while True:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                if len(self.holes) == 4:
                    self._save_data()
                else:
                    print("[!] Saliste sin completar/guardar.")
                break
            elif key == ord('r'):
                self.rect_exterior = []
                self.rect_interior = []
                self.holes = []
                self._refresh_display()

        cv2.destroyAllWindows()