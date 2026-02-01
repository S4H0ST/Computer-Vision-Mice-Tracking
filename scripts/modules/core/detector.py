import cv2
import csv
from ultralytics import YOLO
from pathlib import Path
from typing import Tuple
from helpers.interfaces import BaseModule
from helpers.config import paths, DetectParams


# NOTA: Ya no importamos behavior_rules porque vamos a usar YOLO puro.

class RatDetector(BaseModule):
    def __init__(self, config: DetectParams):
        self.cfg: DetectParams = config
        self.model: YOLO = None
        # Eliminamos self.rules_engine

    def _setup(self) -> None:
        model_path = paths.models_dir / self.cfg.model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

        print(f"[...] Cargando modelo: {model_path}")
        self.model = YOLO(str(model_path))
        # Eliminamos la inicialización de reglas

    def _get_color(self, label: str) -> Tuple[int, int, int]:
        # Colores para dibujar las cajas (puedes añadir 'rat_head' aquí en el futuro)
        if "immobility" in label: return (0, 0, 255)  # Rojo
        if "sniffing" in label:   return (0, 255, 255)  # Amarillo
        if "walking" in label:    return (255, 0, 0)  # Azul
        if "climbing" in label:   return (255, 0, 255)  # Magenta
        if "dipping" in label:    return (255, 165, 0)  # Naranja
        if "rearing" in label:    return (0, 255, 0)  # Verde
        if "head" in label:       return (255, 255, 255)  # Blanco (Nueva clase)
        return (128, 128, 128)

    def run(self) -> None:
        """Función principal que ejecuta detección y guarda datos crudos (Raw Data)."""
        self._setup()

        cap = cv2.VideoCapture(str(paths.video_source))
        if not cap.isOpened():
            print(f"[X] Error abriendo video: {paths.video_source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Configuración de guardado de video
        out_vid = cv2.VideoWriter(str(paths.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Configuración de guardado de CSV (DATOS)
        # IMPORTANTE: Añadimos x1, y1, x2, y2. Esto es lo que necesita la RNN o tu lógica futura.
        csv_path = paths.output_video.with_suffix(".csv")
        f_csv = open(csv_path, "w", newline="")
        writer = csv.writer(f_csv)
        writer.writerow(["frame", "time_s", "cls_id", "label", "conf", "x1", "y1", "x2", "y2"])

        print(f"[>] Procesando... Salida: {paths.output_video}")

        frame_idx = 0

        # Inferencia con YOLO
        results = self.model.predict(
            source=str(paths.video_source), stream=True,
            conf=self.cfg.conf_threshold, device=self.cfg.device, iou=0.5
        )

        for res in results:
            img = res.orig_img.copy()

            if res.boxes:
                # Extraemos datos de la GPU a la CPU
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                cls_ids = res.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(boxes, confs, cls_ids):
                    label = self.model.names[cls_id]

                    # Coordenadas enteras para dibujar
                    x1, y1, x2, y2 = map(int, box)

                    # 1. DIBUJAR EN EL VIDEO
                    color = self._get_color(label)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # 2. GUARDAR EN CSV (Tu base de datos para la futura RNN/Lógica)
                    # Guardamos la caja exacta. Esto vale oro para el análisis posterior.
                    writer.writerow([frame_idx, f"{frame_idx / fps:.3f}", cls_id, label, f"{conf:.2f}", x1, y1, x2, y2])

            out_vid.write(img)
            frame_idx += 1
            if frame_idx % 50 == 0: print(f"   Frame {frame_idx}...", end='\r')

        cap.release()
        out_vid.release()
        f_csv.close()
        print("\n[+] Procesamiento terminado. Datos guardados en CSV.")