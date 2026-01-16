import cv2
import csv
from ultralytics import YOLO
from pathlib import Path
from typing import Tuple
from helpers.interfaces import BaseModule
from helpers.config import paths, DetectParams

# --- CORRECCIÓN: Importar desde 'modules.core' ---
from modules.core.behavior_rules import RatBehaviorRules

class RatDetector(BaseModule):
    def __init__(self, config: DetectParams):
        self.cfg: DetectParams = config
        self.model: YOLO = None
        self.rules_engine: RatBehaviorRules = None

    def _setup(self) -> None:
        model_path = paths.models_dir / self.cfg.model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

        print(f"[...] Cargando modelo: {model_path}")
        self.model = YOLO(str(model_path))

        # Inicializamos el motor de reglas
        self.rules_engine = RatBehaviorRules(paths.coords_json)

    def _get_color(self, label: str) -> Tuple[int, int, int]:
        if "immobility" in label: return (0, 0, 255)
        if "sniffing" in label:   return (0, 255, 255)
        if "walking" in label:    return (255, 0, 0)
        if "climbing" in label:   return (255, 0, 255)
        if "dipping" in label:    return (255, 165, 0)
        if "rearing" in label:    return (0, 255, 0)
        return (255, 255, 255)

    def run(self) -> None:
        """Función principal que ejecuta todo el proceso de detección."""
        self._setup()

        cap = cv2.VideoCapture(str(paths.video_source))
        if not cap.isOpened():
            print(f"[X] Error abriendo video: {paths.video_source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_vid = cv2.VideoWriter(str(paths.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        csv_path = paths.output_video.with_suffix(".csv")
        f_csv = open(csv_path, "w", newline="")
        writer = csv.writer(f_csv)
        writer.writerow(["frame", "time_s", "yolo_id", "raw_label", "FINAL_BEHAVIOR", "conf", "speed"])

        print(f"[>] Procesando... Salida: {paths.output_video}")

        frame_idx = 0
        # Inferencia
        results = self.model.predict(
            source=str(paths.video_source), stream=True,
            conf=self.cfg.conf_threshold, device=self.cfg.device, iou=0.5
        )

        for res in results:
            img = res.orig_img.copy()
            if res.boxes:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                cls_ids = res.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(boxes, confs, cls_ids):
                    raw_label = self.model.names[cls_id]

                    # --- APLICAMOS LAS REGLAS ---
                    final_label, speed = self.rules_engine.apply_rules(box, raw_label)

                    color = self._get_color(final_label)
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f"{final_label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    writer.writerow([frame_idx, f"{frame_idx / fps:.3f}", cls_id, raw_label, final_label, f"{conf:.2f}",
                                     f"{speed:.2f}"])

            out_vid.write(img)
            frame_idx += 1
            if frame_idx % 50 == 0: print(f"   Frame {frame_idx}...", end='\r')

        cap.release()
        out_vid.release()
        f_csv.close()
        print("\n[+] Procesamiento terminado.")