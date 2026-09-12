"""
QThread que ejecuta el pipeline de deteccion y emite senales por frame.
Reimplementa el bucle principal de RatDetector para que la interfaz reciba
actualizaciones en vivo sin bloquear el hilo principal.

Deteccion: usa model.predict(stream=True) sobre la fuente completa, igual que la
version original, garantizando la calidad del tracker interno de YOLO (ByteTrack).
Anadidos respecto a la version base:
  - _draw_skeleton dibuja los 3 keypoints (snout/spine/tail) sin texto sobre el frame.
  - hole_idx en CSV: agujero activo durante head_dipping (0-3), -1 en otro caso.
  - Sin dual_output: solo se genera el video anotado.
"""

import csv as csv_mod
import json
import cv2
import numpy as np
from pathlib import Path
from collections import deque

from PyQt5.QtCore import QThread, pyqtSignal


# ---- Indices de keypoints (snout=0, spine=1, tail=2) ----
KP_SNOUT: int = 0
KP_SPINE: int = 1
KP_TAIL:  int = 2


def _kp_swap_fix(
    snout_kp:   "np.ndarray | None",
    tail_kp:    "np.ndarray | None",
    prev_snout: "np.ndarray | None",
    prev_tail:  "np.ndarray | None",
) -> tuple:
    """
    Detecta y corrige el intercambio snout<->tail comparando el coste de
    asignacion (distancia total al frame anterior) de la asignacion normal
    frente a la invertida. Solo actua si la inversion reduce el coste >20 %.
    Medida temporal hasta ampliar el dataset de entrenamiento.
    """
    if snout_kp is None or tail_kp is None:
        return snout_kp, tail_kp
    if prev_snout is None or prev_tail is None:
        return snout_kp, tail_kp

    sn = snout_kp[:2]
    tl = tail_kp[:2]
    ps = prev_snout[:2]
    pt = prev_tail[:2]

    cost_normal  = float(np.linalg.norm(sn - ps) + np.linalg.norm(tl - pt))
    cost_swapped = float(np.linalg.norm(sn - pt) + np.linalg.norm(tl - ps))

    if cost_swapped < cost_normal * 0.80:
        new_snout = np.array([tail_kp[0], tail_kp[1], snout_kp[2]])
        new_tail  = snout_kp[:2].copy()
        return new_snout, new_tail
    return snout_kp, tail_kp


def _draw_kps(
    img:   "np.ndarray",
    snout: "np.ndarray | None",
    spine: "np.ndarray | None",
    tail:  "np.ndarray | None",
) -> None:
    """Dibuja los 3 keypoints (posiciones ya corregidas) y sus conexiones."""
    kps_with_color = [(snout, (0, 0, 255)), (spine, (0, 255, 0)), (tail, (255, 80, 0))]
    pts: list = []
    for kp, color in kps_with_color:
        if kp is not None and not (kp[0] < 1.0 and kp[1] < 1.0):
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(img, (x, y), 5, color, -1)
            pts.append((x, y))
        else:
            pts.append(None)
    for a, b in ((0, 1), (1, 2)):
        if pts[a] is not None and pts[b] is not None:
            cv2.line(img, pts[a], pts[b], (0, 220, 255), 2)


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


def _draw_skeleton(img: np.ndarray, res, detection_idx: int = 0) -> None:
    """Dibuja snout/spine/tail y sus conexiones sobre img (sin texto)."""
    if res.keypoints is None:
        return
    kps_xy   = res.keypoints.xy
    kps_conf = res.keypoints.conf
    if len(kps_xy) <= detection_idx:
        return

    kps  = kps_xy[detection_idx].cpu().numpy()
    conf = kps_conf[detection_idx].cpu().numpy() if kps_conf is not None else None

    kp_colors = [(0, 0, 255), (0, 255, 0), (255, 80, 0)]  # snout rojo, spine verde, tail naranja

    visible: list[bool] = []
    for i, (kp, color) in enumerate(zip(kps, kp_colors)):
        c = float(conf[i]) if conf is not None else 1.0
        if c < 0.3 or (kp[0] < 1.0 and kp[1] < 1.0):
            visible.append(False)
            continue
        visible.append(True)
        cv2.circle(img, (int(kp[0]), int(kp[1])), 5, color, -1)

    connections = [(KP_SNOUT, KP_SPINE), (KP_SPINE, KP_TAIL)]
    for a, b in connections:
        if a < len(kps) and b < len(kps) and len(visible) > max(a, b):
            if visible[a] and visible[b]:
                pa = (int(kps[a][0]), int(kps[a][1]))
                pb = (int(kps[b][0]), int(kps[b][1]))
                cv2.line(img, pa, pb, (0, 220, 255), 2)


class DetectionWorker(QThread):
    """
    Senales
    -------
    frame_ready  : frame anotado con keypoints (BGR ndarray), dict de stats, indice de frame
    log_msg      : linea de log para el widget de consola
    finished     : dict con rutas de archivos de salida y valores de resumen
    error        : cadena con el mensaje de error
    """
    frame_ready = pyqtSignal(object, dict, int)
    log_msg     = pyqtSignal(str)
    finished    = pyqtSignal(dict)
    error       = pyqtSignal(str)

    def __init__(self, source, output_dir: Path, coords_json: Path,
                 kp_swap_fix: bool = False) -> None:
        super().__init__()
        self._source       = source        # Path (video) o int (indice de camara)
        self._output_dir   = output_dir
        self._coords_json  = coords_json
        self._stop         = False
        self._kp_swap_fix  = kp_swap_fix

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
        is_camera  = isinstance(self._source, int)
        cv2_source = self._source if is_camera else str(self._source)
        stem       = "camara" if is_camera else Path(self._source).stem

        cap = cv2.VideoCapture(cv2_source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ---- Recorte al borde exterior (mejora velocidad y visualizacion) ----
        x1_c = y1_c = 0
        x2_c, y2_c  = w, h
        crop_w, crop_h = w, h
        if spatial.outer_limits is not None:
            lim  = spatial.outer_limits
            x1_c = max(0, int(lim["x_min"]))
            y1_c = max(0, int(lim["y_min"]))
            x2_c = min(w, int(lim["x_max"]))
            y2_c = min(h, int(lim["y_max"]))
            crop_w = x2_c - x1_c
            crop_h = y2_c - y1_c
            spatial.apply_crop_offset(x1_c, y1_c)
            self.log_msg.emit(f"Recorte activo: {w}x{h} → {crop_w}x{crop_h} px")

        # ---- Rutas de salida ----
        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_video = self._output_dir / f"{stem}_anotado.mp4"

        vid_out = VideoOutput(output_video, fps, crop_w, crop_h)
        csv_out = CsvOutput(output_video.with_suffix(".csv"))

        spatial_ok = spatial.is_valid_for(crop_w, crop_h)
        if not spatial_ok:
            self.log_msg.emit("[!] Calibracion parcial: head_dipping no usara referencias espaciales.")

        # ---- Dispositivo ----
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"
        self.log_msg.emit(f"Dispositivo: {'GPU (CUDA)' if device == '0' else 'CPU'}")

        # ---- Bucle de deteccion frame a frame con recorte ----
        stats: dict[str, int] = {k: 0 for k in BEHAVIOR_KEYS}

        # Estado para la correccion de intercambio snout<->tail
        _prev_snout: "np.ndarray | None" = None
        _prev_tail:  "np.ndarray | None" = None

        frame_idx = 0
        while not self._stop:
            ret, raw = cap.read()
            if not ret:
                break

            # Recortar al borde exterior antes de pasarlo a YOLO
            frame = raw[y1_c:y2_c, x1_c:x2_c]

            preds = model.predict(
                source=frame, conf=0.18, device=device, iou=0.5,
                verbose=False, stream=False,
            )
            if not preds:
                vid_out.write(frame)
                self.frame_ready.emit(frame, dict(stats), frame_idx)
                frame_idx += 1
                continue

            res      = preds[0]
            img      = res.orig_img.copy()

            if spatial_ok:
                spatial.draw_zones(img)

            rat_box     = None
            yolo_label  = "Unknown"
            final_label = "Unknown"
            snout_kp    = None
            spine_kp    = None
            tail_kp     = None
            speed_val   = 0.0
            hole_idx    = -1
            best        = 0

            if res.boxes and len(res.boxes) > 0:
                confs      = res.boxes.conf.cpu().numpy()
                best       = int(np.argmax(confs))
                rat_box    = res.boxes.xyxy[best].cpu().numpy()
                cls_id     = int(res.boxes.cls[best].cpu())
                yolo_label = model.names.get(cls_id, "Unknown")

            # Extraer keypoints de la mejor deteccion
            if rat_box is not None and res.keypoints is not None:
                kps_xy   = res.keypoints.xy
                kps_conf = res.keypoints.conf
                if len(kps_xy) > best:
                    snout = kps_xy[best][KP_SNOUT].cpu().numpy()
                    c0 = float(kps_conf[best][KP_SNOUT].cpu()) if kps_conf is not None else 1.0
                    if c0 >= 0.3 and not (snout[0] < 1.0 and snout[1] < 1.0):
                        snout_kp = np.array([snout[0], snout[1], c0])
                    if kps_xy[best].shape[0] > KP_TAIL:
                        tail = kps_xy[best][KP_TAIL].cpu().numpy()
                        c2 = float(kps_conf[best][KP_TAIL].cpu()) if kps_conf is not None else 1.0
                        if c2 >= 0.3 and not (tail[0] < 1.0 and tail[1] < 1.0):
                            tail_kp = tail
                    if kps_xy[best].shape[0] > KP_SPINE:
                        sp = kps_xy[best][KP_SPINE].cpu().numpy()
                        c1 = float(kps_conf[best][KP_SPINE].cpu()) if kps_conf is not None else 1.0
                        if c1 >= 0.3 and not (sp[0] < 1.0 and sp[1] < 1.0):
                            spine_kp = sp

            if rat_box is not None:
                # Correccion de intercambio snout<->tail (heuristica temporal)
                if self._kp_swap_fix:
                    snout_kp, tail_kp = _kp_swap_fix(snout_kp, tail_kp, _prev_snout, _prev_tail)
                if snout_kp is not None:
                    _prev_snout = snout_kp
                if tail_kp is not None:
                    _prev_tail = tail_kp

                speed_val   = speed_tracker.update(rat_box, crop_w, crop_h)
                final_label = classifier.classify(yolo_label, speed_val, snout_kp, rat_box, spatial_ok)

                if final_label == "rat_head_dipping" and snout_kp is not None and spatial_ok:
                    hole_idx = spatial.check_dipping_hole(snout_kp)

                color = _label_color(final_label)
                rx1, ry1, rx2, ry2 = map(int, rat_box)

                if final_label.startswith("sniffing_"):
                    label_txt = f"sniffing [{final_label.split('_', 1)[1]}]"
                else:
                    label_txt = final_label.removeprefix("rat_").replace("_", " ")

                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color, 2)
                cv2.putText(img, label_txt, (rx1, ry1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                _draw_kps(img, snout_kp, spine_kp, tail_kp)

                csv_out.write_row(frame_idx, fps, yolo_label, final_label,
                                  rat_box, snout_kp, tail_kp, speed_val, hole_idx)

                key = _label_to_stat_key(final_label)
                if key:
                    stats[key] += 1

            vid_out.write(img)
            self.frame_ready.emit(img, dict(stats), frame_idx)
            frame_idx += 1

        if self._stop:
            self.log_msg.emit("Deteccion detenida por el usuario.")

        cap.release()
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
            "video_clean":     None,
            "csv":             csv_out.path,
            "coords_json":     self._coords_json,
            "excel":           stats_dir / f"stats_{csv_stem}.xlsx",
            "trajectory":      stats_dir / f"trajectory_{csv_stem}.png",
            "heatmap":         stats_dir / f"heatmap_{csv_stem}.png",
            "folder":          self._output_dir,
            "frame_count":     frame_idx,
            "duration_s":      frame_idx / fps if fps > 0 else 0,
        }
        self.finished.emit(output_paths)
