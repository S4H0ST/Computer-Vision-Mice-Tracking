"""
QThread que ejecuta el pipeline de deteccion y emite senales por frame.
Reimplementa el bucle principal de RatDetector para que la interfaz reciba
actualizaciones en vivo sin bloquear el hilo principal.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

from PyQt5.QtCore import QThread, pyqtSignal


# Copia local de _SpeedTracker (clase privada; evita importar internos del detector)
class _SpeedTracker:
    _MAX_PLAUSIBLE_SPEED: float = 8.0

    def __init__(self, smoothing: int = 5) -> None:
        self._history: deque = deque(maxlen=smoothing)
        self._prev = None

    def update(self, box: np.ndarray, img_w: int, img_h: int) -> float:
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        speed = 0.0
        if self._prev is not None:
            speed = np.sqrt((cx - self._prev[0]) ** 2 + (cy - self._prev[1]) ** 2) * 100.0
            if speed > self._MAX_PLAUSIBLE_SPEED:
                speed = self._history[-1] if self._history else 0.0
        self._prev = (cx, cy)
        self._history.append(speed)
        return float(np.mean(self._history))

    def reset(self) -> None:
        self._prev = None
        self._history.clear()


BEHAVIOR_KEYS = ("immobile", "walking", "sniffing", "climbing", "rearing", "dipping", "grooming")

LABEL_COLOR: dict[str, tuple] = {
    "sniffing_immobile": (200, 100, 180),
    "sniffing_walking":  (0,   200, 255),
    "immobile":          (180, 180, 180),
    "walking":           (0,   255, 255),
    "rat_climbing":      (255,   0, 255),
    "rat_head_dipping":  (0,   165, 255),
    "rat_rearing":       (0,   255,   0),
    "rat_grooming":      (180, 255, 180),
}


def _label_color(label: str) -> tuple:
    for key, color in LABEL_COLOR.items():
        if key in label:
            return color
    return (128, 128, 128)


def _label_to_stat_key(label: str) -> str:
    if "immobile" in label and "sniffing" not in label:
        return "immobile"
    if "walking" in label and "sniffing" not in label:
        return "walking"
    if "sniffing" in label:
        return "sniffing"
    if "climbing" in label:
        return "climbing"
    if "rearing" in label:
        return "rearing"
    if "head_dipping" in label:
        return "dipping"
    if "grooming" in label:
        return "grooming"
    return ""


class DetectionWorker(QThread):
    """
    Senales
    -------
    frame_ready  : frame anotado (BGR ndarray), diccionario de stats, indice de frame
    log_msg      : linea de log para el widget de consola
    finished     : dict con rutas de archivos de salida y valores de resumen
    error        : cadena con el mensaje de error
    """
    frame_ready = pyqtSignal(object, dict, int)
    log_msg     = pyqtSignal(str)
    finished    = pyqtSignal(dict)
    error       = pyqtSignal(str)

    def __init__(self, source, output_dir: Path, coords_json: Path) -> None:
        super().__init__()
        self._source     = source        # Path (video) o int (indice de camara)
        self._output_dir = output_dir
        self._coords_json = coords_json
        self._stop       = False

    def request_stop(self) -> None:
        self._stop = True

    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            self._detect()
        except Exception as exc:
            self.error.emit(str(exc))

    def _detect(self) -> None:
        from ultralytics import YOLO
        from spatial.spatial import SpatialAnalyzer
        from behavior.behavior_classifier import BehaviorClassifier
        from output.writers import VideoOutput, CsvOutput
        from config.config import paths
        from utils.stats_generator import StatsGenerator

        # ---- Inicializacion ----
        self.log_msg.emit(f"Cargando modelo: {paths.yolo_model.name}")
        model = YOLO(str(paths.yolo_model))

        spatial    = SpatialAnalyzer(self._coords_json)
        classifier = BehaviorClassifier(spatial_logic=spatial)
        speed_tracker = _SpeedTracker(smoothing=5)

        # ---- Informacion de la fuente ----
        is_camera = isinstance(self._source, int)
        cv2_source = self._source if is_camera else str(self._source)
        stem = "camara" if is_camera else Path(self._source).stem

        cap = cv2.VideoCapture(cv2_source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # ---- Rutas de salida ----
        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_video = self._output_dir / f"{stem}_anotado.mp4"

        vid_out = VideoOutput(output_video, fps, w, h, dual_output=True)
        csv_out = CsvOutput(output_video.with_suffix(".csv"))

        spatial_ok = spatial.is_valid_for(w, h)
        if not spatial_ok:
            self.log_msg.emit("[!] Calibracion parcial: head_dipping no usara referencias espaciales.")

        # ---- Bucle de deteccion ----
        stats: dict[str, int] = {k: 0 for k in BEHAVIOR_KEYS}

        import torch
        device = "0" if torch.cuda.is_available() else "cpu"
        self.log_msg.emit(f"Dispositivo: {'GPU (CUDA)' if device == '0' else 'CPU'}")

        results = model.predict(
            source=cv2_source, stream=True,
            conf=0.18, device=device, iou=0.5,
        )

        frame_idx = 0
        for res in results:
            if self._stop:
                self.log_msg.emit("Deteccion detenida por el usuario.")
                break

            img_base  = res.orig_img.copy()
            img       = img_base.copy()
            img_clean = img_base.copy()

            if spatial_ok:
                spatial.draw_zones(img)

            rat_box    = None
            yolo_label = "Unknown"
            final_label = "Unknown"
            snout_kp   = None
            tail_kp    = None
            speed_val  = 0.0

            if res.boxes and len(res.boxes) > 0:
                confs      = res.boxes.conf.cpu().numpy()
                best       = int(np.argmax(confs))
                rat_box    = res.boxes.xyxy[best].cpu().numpy()
                cls_id     = int(res.boxes.cls[best].cpu())
                yolo_label = model.names.get(cls_id, "Unknown")

            if rat_box is not None and res.keypoints is not None:
                kps_xy   = res.keypoints.xy
                kps_conf = res.keypoints.conf
                if len(kps_xy) > 0:
                    # hocico (kp 0)
                    snout = kps_xy[0][0].cpu().numpy()
                    c0 = float(kps_conf[0][0].cpu()) if kps_conf is not None else 1.0
                    if c0 >= 0.3 and not (snout[0] < 1.0 and snout[1] < 1.0):
                        snout_kp = np.array([snout[0], snout[1], c0])
                    # cola (kp 2)
                    if kps_xy[0].shape[0] > 2:
                        tail = kps_xy[0][2].cpu().numpy()
                        c2 = float(kps_conf[0][2].cpu()) if kps_conf is not None else 1.0
                        if c2 >= 0.3 and not (tail[0] < 1.0 and tail[1] < 1.0):
                            tail_kp = tail

            if rat_box is not None:
                speed_val   = speed_tracker.update(rat_box, w, h)
                final_label = classifier.classify(yolo_label, speed_val, snout_kp, rat_box, spatial_ok)

                color = _label_color(final_label)
                rx1, ry1, rx2, ry2 = map(int, rat_box)

                if final_label.startswith("sniffing_"):
                    label_txt = f"sniffing [{final_label.split('_', 1)[1]}]"
                else:
                    label_txt = final_label.removeprefix("rat_").replace("_", " ")

                for target in (img, img_clean):
                    cv2.rectangle(target, (rx1, ry1), (rx2, ry2), color, 2)
                    cv2.putText(target, label_txt, (rx1, ry1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                csv_out.write_row(frame_idx, fps, yolo_label, final_label,
                                  rat_box, snout_kp, tail_kp, speed_val)

                key = _label_to_stat_key(final_label)
                if key:
                    stats[key] += 1

            vid_out.write(img, img_clean)
            self.frame_ready.emit(img, dict(stats), frame_idx)
            frame_idx += 1

        vid_out.release()
        csv_out.close()
        self.log_msg.emit(f"Deteccion completada: {frame_idx} frames.")

        # ---- Postprocesado: estadisticas ----
        stats_dir = self._output_dir / "stats"
        try:
            self.log_msg.emit("Generando estadisticas...")
            StatsGenerator(csv_out.path, coords_json=self._coords_json).generate(stats_dir)
        except Exception as exc:
            self.log_msg.emit(f"[!] Stats: {exc}")

        csv_stem = csv_out.path.stem
        output_paths = {
            "video_annotated": output_video,
            "video_clean":     vid_out.clean_path,
            "excel":           stats_dir / f"stats_{csv_stem}.xlsx",
            "trajectory":      stats_dir / f"trajectory_{csv_stem}.png",
            "heatmap":         stats_dir / f"heatmap_{csv_stem}.png",
            "folder":          self._output_dir,
            "frame_count":     frame_idx,
            "duration_s":      frame_idx / fps if fps > 0 else 0,
        }
        self.finished.emit(output_paths)
