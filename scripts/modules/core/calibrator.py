import cv2
import json
import numpy as np
from pathlib import Path
from helpers.interfaces import BaseModule
from helpers.config import paths


class ZoneCalibrator(BaseModule):
    def __init__(self, video_path: Path):
        self.video_path = str(video_path)
        self.output_json = paths.coords_json

        # --- CONFIGURACIÓN ---
        self.hole_radius = 20  # <--- radio px de área de detección de agujeros

        # Listas de puntos
        self.rect_exterior = []
        self.rect_interior = []
        self.holes = []

        self.img_display = None

    def _click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.rect_exterior) < 2:
                self.rect_exterior.append((x, y))
                print(f"[+] Borde Exterior P{len(self.rect_exterior)}: ({x}, {y})")
            elif len(self.rect_interior) < 2:
                self.rect_interior.append((x, y))
                print(f"[+] Borde Interior P{len(self.rect_interior)}: ({x}, {y})")
            elif len(self.holes) < 4:
                self.holes.append((x, y))
                print(f"[+] Agujero {len(self.holes)}: ({x}, {y})")

            self._draw_state()

    def _draw_state(self):
        if self.img_display is None: return
        display = self.img_display.copy()

        # 1. DIBUJAR PUNTOS Y ETIQUETAS

        # A) Borde Exterior (Azul)
        for i, pt in enumerate(self.rect_exterior):
            cv2.circle(display, pt, 5, (255, 0, 0), -1)
            # Etiqueta cambiada a español completo
            label = "Borde Exterior" if i == 0 else "Borde Exterior 2"  # O simplemente enumerado
            label = f"Borde Exterior {i + 1}"
            cv2.putText(display, label, (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # B) Borde Interior (Verde)
        for i, pt in enumerate(self.rect_interior):
            cv2.circle(display, pt, 5, (0, 255, 0), -1)
            label = f"Borde Interior {i + 1}"
            cv2.putText(display, label, (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # C) Agujeros (Rojo)
        for i, pt in enumerate(self.holes):
            # Punto central
            cv2.circle(display, pt, 5, (0, 0, 255), -1)
            # ÁREA DE DETECCIÓN (Círculo vacío con radio actualizado)
            cv2.circle(display, pt, self.hole_radius, (0, 0, 255), 2)

            # Etiqueta "Agujero X"
            label = f"Agujero {i + 1}"
            cv2.putText(display, label, (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 2. DIBUJAR RECTÁNGULOS
        if len(self.rect_exterior) == 2:
            cv2.rectangle(display, self.rect_exterior[0], self.rect_exterior[1], (255, 0, 0), 2)
        if len(self.rect_interior) == 2:
            cv2.rectangle(display, self.rect_interior[0], self.rect_interior[1], (0, 255, 0), 2)

        # 3. INSTRUCCIONES SUPERIORES
        cv2.rectangle(display, (0, 0), (600, 30), (0, 0, 0), -1)
        if len(self.rect_exterior) < 2:
            msg = f"PASO 1: Marca Borde Exterior ({len(self.rect_exterior)}/2)"
        elif len(self.rect_interior) < 2:
            msg = f"PASO 2: Marca Borde Interior ({len(self.rect_interior)}/2)"
        elif len(self.holes) < 4:
            msg = f"PASO 3: Marca Agujeros ({len(self.holes)}/4)"
        else:
            msg = "CALIBRACION LISTA. 'q' para guardar."

        cv2.putText(display, msg, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('CALIBRATION_TOOL', display)

    def _save_config(self):
        if len(self.rect_interior) < 2:
            print("[!] Error: Calibración incompleta.")
            return

        x_coords = [p[0] for p in self.rect_interior]
        y_coords = [p[1] for p in self.rect_interior]

        data = {
            "limits": {
                "x_min": min(x_coords), "x_max": max(x_coords),
                "y_min": min(y_coords), "y_max": max(y_coords)
            },
            "holes": self.holes,
            "hole_radius": self.hole_radius
        }

        with open(self.output_json, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"[OK] Guardado en: {self.output_json}")

    def run(self):
        if not Path(self.video_path).exists():
            print(f"[!] No existe video: {self.video_path}")
            return

        print("--- MODO CALIBRACIÓN ---")
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret: return

        self.img_display = frame
        cv2.namedWindow('CALIBRATION_TOOL')
        cv2.setMouseCallback('CALIBRATION_TOOL', self._click_event)
        self._draw_state()

        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cv2.destroyAllWindows()
        self._save_config()