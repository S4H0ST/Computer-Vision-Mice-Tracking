import cv2
import csv
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import Tuple, Optional

from helpers.interfaces import BaseModule
from helpers.config import paths, DetectParams

# Importamos los módulos de inteligencia
from modules.brain.inference import ActionPredictor
from modules.logic.spatial import SpatialAnalyzer


class RatDetector(BaseModule):
    def __init__(self, config: DetectParams):
        self.cfg: DetectParams = config
        self.model: YOLO = None

        # Módulos de Inteligencia Híbrida
        self.rnn_brain: ActionPredictor = None
        self.spatial_logic: SpatialAnalyzer = None

    def _setup(self) -> None:
        # 1. Cargar YOLO
        if not paths.yolo_model.exists():
            raise FileNotFoundError(f"Modelo YOLO no encontrado en: {paths.yolo_model}")

        print(f"[Core] Cargando YOLO: {paths.yolo_model}")
        self.model = YOLO(str(paths.yolo_model))

        # 2. Inicializar RNN (SIN ARGUMENTOS, ya lo coge de config.py)
        # --- CORRECCIÓN AQUÍ ---
        self.rnn_brain = ActionPredictor()

        # 3. Inicializar Lógica Espacial
        self.spatial_logic = SpatialAnalyzer(config_path=paths.coords_json)

    def _get_color(self, label: str) -> Tuple[int, int, int]:
        label = label.lower()
        if "immobility" in label: return (0, 0, 255)  # Rojo
        if "walking" in label:    return (255, 0, 0)  # Azul
        if "horizontal" in label: return (255, 0, 0)  # Azul
        if "climbing" in label:   return (255, 0, 255)  # Magenta
        if "dipping" in label:    return (255, 165, 0)  # Naranja
        if "rearing" in label:    return (0, 255, 0)  # Verde
        if "head" in label:       return (200, 200, 200)  # Gris (Cabeza)
        return (128, 128, 128)

    def run(self) -> None:
        self._setup()

        # Usamos video_source de config.py
        cap = cv2.VideoCapture(str(paths.video_source))
        if not cap.isOpened():
            print(f"[X] Error abriendo video: {paths.video_source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Configuración de salida usando config.py
        out_vid = cv2.VideoWriter(str(paths.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        csv_path = paths.output_video.with_suffix(".csv")
        f_csv = open(csv_path, "w", newline="")
        writer = csv.writer(f_csv)
        # Cabecera del CSV
        writer.writerow(["frame", "time", "yolo_label", "final_label", "x1", "y1", "x2", "y2"])

        print(f"[>] Procesando video... Salida en: {paths.output_video}")
        frame_idx = 0

        # Inferencia
        results = self.model.predict(
            source=str(paths.video_source), stream=True,
            conf=self.cfg.conf_threshold, device=self.cfg.device, iou=0.5
        )

        for res in results:
            img = res.orig_img.copy()

            rat_box = None
            head_box = None
            yolo_label_rat = "Unknown"

            # 1. Extracción de datos de YOLO
            if res.boxes:
                boxes = res.boxes.xyxy.cpu().numpy()
                cls_ids = res.boxes.cls.cpu().numpy().astype(int)

                for box, cls_id in zip(boxes, cls_ids):
                    label = self.model.names[cls_id]
                    if "head" in label:
                        head_box = box
                    else:
                        rat_box = box
                        yolo_label_rat = label

            final_label = yolo_label_rat  # Por defecto confiamos en YOLO

            # 2. Lógica Híbrida (Solo si detectamos rata)
            if rat_box is not None:
                # A) Consultar RNN (Analiza el movimiento temporal)
                # Nota: Si no has entrenado la RNN aún, esto devolverá "Analyzing..." o nada.
                rnn_prediction = self.rnn_brain.update_and_predict(rat_box, w, h)

                if rnn_prediction and rnn_prediction != "Analyzing...":
                    final_label = rnn_prediction  # La RNN corrige a YOLO

                # B) Consultar Lógica Espacial (Head Dipping tiene prioridad absoluta)
                if head_box is not None:
                    is_dipping = self.spatial_logic.check_dipping(head_box)
                    if is_dipping:
                        final_label = "rat_head_dipping"

                    # Dibujar caja de la cabeza
                    hx1, hy1, hx2, hy2 = map(int, head_box)
                    cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (200, 200, 200), 1)

                # Dibujar Rata y Etiqueta
                color = self._get_color(final_label)
                rx1, ry1, rx2, ry2 = map(int, rat_box)
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color, 2)

                # Texto: ETIQUETA FINAL [ORIGINAL]
                text = f"{final_label}"
                cv2.putText(img, text, (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Guardar datos en CSV
                writer.writerow([frame_idx, f"{frame_idx / fps:.2f}", yolo_label_rat, final_label, rx1, ry1, rx2, ry2])

            out_vid.write(img)
            frame_idx += 1
            if frame_idx % 20 == 0: print(f"   Frame {frame_idx}...", end='\r')

        cap.release()
        out_vid.release()
        f_csv.close()
        print("\n[+] Proceso finalizado. CSV y Video generados.")