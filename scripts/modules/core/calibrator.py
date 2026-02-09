import cv2
import json
from pathlib import Path
from helpers.interfaces import BaseModule
from helpers.config import paths


class ZoneCalibrator(BaseModule):
    def __init__(self, video_path: Path):
        self.video_path = str(video_path)
        self.output_json = paths.coords_json
        self.holes = []
        self.hole_radius = 20

    def _click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.holes.append((x, y))
            print(f"[+] Agujero marcado en: ({x}, {y})")

            # Dibujar visualmente
            cv2.circle(self.img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.circle(self.img_display, (x, y), self.hole_radius, (0, 255, 0), 1)
            cv2.imshow("Calibrator", self.img_display)

    def run(self):
        if not Path(self.video_path).exists():
            print(f"[!] No encuentro el video: {self.video_path}")
            return

        print("--- CALIBRACIÓN DE AGUJEROS ---")
        print("1. Haz clic en el centro de cada agujero.")
        print("2. Presiona 's' para guardar y salir.")
        print("3. Presiona 'q' para salir sin guardar.")

        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("[!] Error leyendo frame del video.")
            return

        self.img_display = frame.copy()
        cv2.imshow("Calibrator", self.img_display)
        cv2.setMouseCallback("Calibrator", self._click_event)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                data = {"holes": self.holes, "hole_radius": self.hole_radius}
                with open(self.output_json, "w") as f:
                    json.dump(data, f, indent=4)
                print(f"[OK] Guardado en {self.output_json}")
                break
            elif key == ord("q"):
                print("[!] Cancelado.")
                break

        cv2.destroyAllWindows()