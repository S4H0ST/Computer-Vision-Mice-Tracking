import cv2
import csv
import numpy as np
from collections import deque
from ultralytics import YOLO
from pathlib import Path
from typing import Optional, Tuple

from helpers.base import BaseModule
from helpers.configuracion import paths, DetectParams
from modules.brain_rnn.inference import ActionPredictor
from modules.detector_agujeros.agujeros import SpatialAnalyzer


# ── Etiquetas finales que ve el investigador ──────────────────────────────── #
LABEL_ES = {
    "rat_climbing":     "Trepar / Escalar",
    "rat_grooming":     "Acicalamiento / Limpieza",
    "rat_head_dipping": "Asomarse por agujero",
    "rat_horizontal":   "Horizontal (sin clasificar)",
    "rat_rearing":      "Incorporarse / Erguirse",
    "walking":          "Caminando",
    "immobile":         "Inmóvil",
    "sniffing":         "Olfateando pared",
}

# ── Umbrales de velocidad para walking vs immobile ───────────────────────── #
# Speed = desplazamiento normalizado del centroide × 100 (igual que en RNN).
# Ajustar si el vídeo tiene fps muy distintos o la caja es muy pequeña.
WALK_SPEED_THRESHOLD   = 0.45   # por encima → walking
STILL_SPEED_THRESHOLD  = 0.18   # por debajo → immobile
# Entre ambos umbrales → estado ambiguo, se mantiene la etiqueta de YOLO.

# ── Índices de keypoints según kpt_shape: [snout, spine, tail] ───────────── #
KP_SNOUT = 0
KP_SPINE = 1
KP_TAIL  = 2


class _SpeedTracker:
    """
    Calcula velocidad del centroide del bounding box entre frames.
    Misma fórmula que ActionPredictor para consistencia.
    """

    def __init__(self, smoothing: int = 5):
        self._history = deque(maxlen=smoothing)
        self._prev: Optional[Tuple[float, float]] = None

    def update(self, box, img_w: int, img_h: int) -> float:
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h

        speed = 0.0
        if self._prev is not None:
            speed = np.sqrt((cx - self._prev[0]) ** 2 +
                            (cy - self._prev[1]) ** 2) * 100.0
        self._prev = (cx, cy)
        self._history.append(speed)
        return float(np.mean(self._history))

    def reset(self):
        self._prev = None
        self._history.clear()


class RatDetector(BaseModule):

    def __init__(self, config: DetectParams):
        self.cfg            = config
        self.model: YOLO    = None
        self.rnn_brain      = None
        self.spatial_logic  = None
        self._speed_tracker = _SpeedTracker(smoothing=5)

    # ------------------------------------------------------------------ #
    def _setup(self) -> None:
        if not paths.yolo_model.exists():
            raise FileNotFoundError(f"Modelo YOLO no encontrado: {paths.yolo_model}")

        print(f"[Core] Cargando YOLO Pose: {paths.yolo_model}")
        self.model = YOLO(str(paths.yolo_model))

        self.rnn_brain     = ActionPredictor()
        self.spatial_logic = SpatialAnalyzer(config_path=paths.coords_json)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_color(label: str) -> Tuple[int, int, int]:
        label = label.lower()
        if "immobile"  in label: return (0,   0,   255)  # rojo
        if "walking"   in label: return (255, 180,   0)  # naranja
        if "sniffing"  in label: return (0,   200, 255)  # cian
        if "horizontal"in label: return (200, 200,   0)  # amarillo
        if "climbing"  in label: return (255,   0, 255)  # magenta
        if "dipping"   in label: return (0,   165, 255)  # naranja oscuro
        if "rearing"   in label: return (0,   255,   0)  # verde
        if "grooming"  in label: return (180, 255, 180)  # verde claro
        return (128, 128, 128)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_snout(res, detection_idx: int = 0) -> Optional[np.ndarray]:
        """
        Extrae las coordenadas (x, y) del snout (keypoint 0) del primer ratón.
        Devuelve None si YOLO no produce keypoints o la confianza es muy baja.
        """
        if res.keypoints is None:
            return None
        kps_xy   = res.keypoints.xy    # (N, K, 2) tensor
        kps_conf = res.keypoints.conf  # (N, K)   tensor o None

        if len(kps_xy) <= detection_idx:
            return None

        snout = kps_xy[detection_idx][KP_SNOUT].cpu().numpy()  # (2,)

        # Descartar si la confianza del keypoint es demasiado baja
        if kps_conf is not None:
            conf = float(kps_conf[detection_idx][KP_SNOUT].cpu())
            if conf < 0.3:
                return None

        # Descartar si el keypoint está en (0,0) — YOLO usa (0,0) para "no visible"
        if snout[0] < 1.0 and snout[1] < 1.0:
            return None

        return snout

    # ------------------------------------------------------------------ #
    def _derive_horizontal(self,
                           snout_kp: Optional[np.ndarray],
                           rat_box: np.ndarray,
                           img_w: int,
                           img_h: int) -> str:
        """
        Desambigua rat_horizontal en tres comportamientos finales:
          1. sniffing   — snout cerca de la pared interior
          2. walking    — desplazamiento rápido
          3. immobile   — desplazamiento mínimo

        Prioridad: sniffing > walking/immobile.
        Si la RNN está activa, ésta reemplaza el criterio de velocidad para
        walking/immobile (la RNN tiene más contexto temporal).
        """
        # 1. Sniffing: snout cerca de pared (prioridad sobre velocidad)
        if snout_kp is not None and self.spatial_logic.check_sniffing(snout_kp):
            return "sniffing"

        # 2. RNN temporal (si está entrenada)
        rnn_pred = self.rnn_brain.update_and_predict(rat_box, img_w, img_h)
        if rnn_pred and rnn_pred not in ("Analyzing...", "rat_horizontal"):
            return rnn_pred   # la RNN distingue walking/immobile con su memoria temporal

        # 3. Fallback: velocidad del centroide
        speed = self._speed_tracker.update(rat_box, img_w, img_h)
        if speed >= WALK_SPEED_THRESHOLD:
            return "walking"
        if speed <= STILL_SPEED_THRESHOLD:
            return "immobile"

        # Zona ambigua: devolver horizontal sin refinar
        return "rat_horizontal"

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self._setup()

        cap = cv2.VideoCapture(str(paths.video_source))
        if not cap.isOpened():
            print(f"[X] Error abriendo video: {paths.video_source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        out_vid  = cv2.VideoWriter(str(paths.output_video),
                                   cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        csv_path = paths.output_video.with_suffix(".csv")
        f_csv    = open(csv_path, "w", newline="")
        writer   = csv.writer(f_csv)
        writer.writerow(["frame", "time_s", "yolo_label", "final_label",
                         "x1", "y1", "x2", "y2",
                         "snout_x", "snout_y", "speed"])

        print(f"[>] Procesando: {paths.video_source.name}")
        print(f"    Salida    : {paths.output_video}")

        # Imprimir leyenda
        print("\n" + "=" * 55)
        print("  LEYENDA")
        for k, v in LABEL_ES.items():
            print(f"  {k:<22}  {v}")
        print("=" * 55 + "\n")

        results = self.model.predict(
            source=str(paths.video_source), stream=True,
            conf=self.cfg.conf_threshold, device=self.cfg.device, iou=0.5
        )

        frame_idx  = 0
        last_label = "—"

        for res in results:
            img = res.orig_img.copy()

            rat_box      = None
            yolo_label   = "Unknown"
            final_label  = "Unknown"
            snout_kp     = None
            speed_val    = 0.0

            # ── 1. Extraer detecciones de YOLO ──────────────────────────
            if res.boxes and len(res.boxes) > 0:
                # Tomar la detección con mayor confianza
                confs   = res.boxes.conf.cpu().numpy()
                best    = int(np.argmax(confs))
                rat_box = res.boxes.xyxy[best].cpu().numpy()
                cls_id  = int(res.boxes.cls[best].cpu())
                yolo_label = self.model.names.get(cls_id, "Unknown")

            # ── 2. Extraer snout keypoint ────────────────────────────────
            if rat_box is not None and res.keypoints is not None:
                snout_kp = self._extract_snout(res, detection_idx=0)

            # ── 3. Lógica híbrida ────────────────────────────────────────
            if rat_box is not None:
                final_label = yolo_label  # base: confiar en YOLO

                # A) HEAD DIPPING: prioridad absoluta — snout sobre agujero
                if snout_kp is not None and self.spatial_logic.check_dipping(snout_kp):
                    final_label = "rat_head_dipping"

                # B) Desambiguar horizontal → walking / immobile / sniffing
                elif yolo_label == "rat_horizontal":
                    final_label = self._derive_horizontal(snout_kp, rat_box, w, h)
                    speed_val   = self._speed_tracker._history[-1] if self._speed_tracker._history else 0.0

                # C) Para otros estados (rearing, grooming, climbing):
                #    YOLO es suficiente; la RNN puede refinar si está activa.
                else:
                    rnn_pred = self.rnn_brain.update_and_predict(rat_box, w, h)
                    if rnn_pred and rnn_pred not in ("Analyzing...", yolo_label):
                        final_label = rnn_pred

                # ── Dibujar ─────────────────────────────────────────────
                color = self._get_color(final_label)
                rx1, ry1, rx2, ry2 = map(int, rat_box)
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color, 2)
                cv2.putText(img, final_label, (rx1, ry1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Dibujar snout keypoint
                if snout_kp is not None:
                    sx, sy = int(snout_kp[0]), int(snout_kp[1])
                    cv2.circle(img, (sx, sy), 5, (0, 0, 255), -1)
                    cv2.putText(img, "snout", (sx + 6, sy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                # ── CSV ──────────────────────────────────────────────────
                snout_x = float(snout_kp[0]) if snout_kp is not None else -1
                snout_y = float(snout_kp[1]) if snout_kp is not None else -1
                writer.writerow([frame_idx, f"{frame_idx / fps:.2f}",
                                 yolo_label, final_label,
                                 rx1, ry1, rx2, ry2,
                                 f"{snout_x:.1f}", f"{snout_y:.1f}",
                                 f"{speed_val:.3f}"])
                last_label = final_label

            out_vid.write(img)
            frame_idx += 1

            if frame_idx % 30 == 0:
                es = LABEL_ES.get(last_label, last_label)
                print(f"   Frame {frame_idx:>6}  |  {last_label:<22}  ({es})", end="\r")

        out_vid.release()
        f_csv.close()
        print(f"\n[+] Finalizado. {frame_idx} frames → {paths.output_video.name}")
        print(f"    CSV: {csv_path.name}")
