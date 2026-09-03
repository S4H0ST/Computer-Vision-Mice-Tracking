"""
Detector principal de comportamiento del raton usando YOLO Pose + logica hibrida.

Clases:
    _SpeedTracker — calcula y suaviza la velocidad del centroide entre frames.
    RatDetector   — orquesta la inferencia YOLO, delega la clasificacion en
                    BehaviorClassifier, y escribe video y CSV via VideoOutput/CsvOutput.
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

from config.interfaces import BaseModule
from config.config import paths, DetectParams
from spatial.spatial import SpatialAnalyzer
from behavior.behavior_classifier import BehaviorClassifier
from output.writers import VideoOutput, CsvOutput


# Traducciones de etiquetas internas para la leyenda impresa en consola
LABEL_ES: dict[str, str] = {
    "rat_climbing":      "Trepar / Escalar",
    "rat_grooming":      "Acicalamiento / Limpieza",
    "rat_head_dipping":  "Asomarse por agujero",
    "rat_rearing":       "Incorporarse / Erguirse",
    "walking":           "Caminando",
    "immobile":          "Inmovil",
    "sniffing_walking":  "Olfateando (en movimiento)",
    "sniffing_immobile": "Olfateando (parado)",
}

# Indices de keypoints segun kpt_shape: [snout, spine, tail]
KP_SNOUT: int = 0
KP_SPINE: int = 1
KP_TAIL: int  = 2


class _SpeedTracker:
    """
    Calcula la velocidad del centroide del bounding box entre frames consecutivos.
    Usa la misma formula que ActionPredictor para garantizar consistencia.
    """

    # Salto maximo plausible entre frames a 15fps en caja normalizada.
    _MAX_PLAUSIBLE_SPEED: float = 8.0

    def __init__(self, smoothing: int = 5) -> None:
        self._history: deque = deque(maxlen=smoothing)
        self._prev: tuple[float, float] | None = None

    def update(self, box: np.ndarray, img_w: int, img_h: int) -> float:
        """
        Calcula la velocidad media suavizada para el frame actual.

        box   : array [x1, y1, x2, y2] en pixeles.
        img_w : ancho de la imagen (para normalizar).
        img_h : alto de la imagen (para normalizar).
        """
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h

        speed = 0.0
        if self._prev is not None:
            speed = np.sqrt((cx - self._prev[0]) ** 2 +
                            (cy - self._prev[1]) ** 2) * 100.0
            if speed > self._MAX_PLAUSIBLE_SPEED:
                speed = self._history[-1] if self._history else 0.0
        self._prev = (cx, cy)
        self._history.append(speed)
        return float(np.mean(self._history))

    def reset(self) -> None:
        """Reinicia el historial y el estado previo."""
        self._prev = None
        self._history.clear()


class RatDetector(BaseModule):
    """
    Detecta y clasifica el comportamiento del raton frame a frame.
    Combina YOLO Pose, BehaviorClassifier y la logica espacial calibrada.
    """

    def __init__(self, config: DetectParams,
                 show_skeleton: bool = False,
                 show_preview: bool = True,
                 dual_output: bool = False,
                 camera_index: int | None = None) -> None:
        self.cfg: DetectParams                     = config
        self.show_skeleton: bool                   = show_skeleton
        self.show_preview: bool                    = show_preview
        self.dual_output: bool                     = dual_output
        # camera_index != None activa el modo camara en vivo (0 = camara por defecto)
        self.camera_index: int | None              = camera_index
        self.model: YOLO | None                    = None
        self.spatial_logic: SpatialAnalyzer | None = None
        self._speed_tracker: _SpeedTracker         = _SpeedTracker(smoothing=5)
        self._behavior_classifier: BehaviorClassifier | None = None

    def _setup(self) -> None:
        """Carga el modelo YOLO, el analizador espacial y el clasificador de comportamiento."""
        if not paths.yolo_model.exists():
            raise FileNotFoundError(f"Modelo YOLO no encontrado: {paths.yolo_model}")

        print(f"[Core] Cargando YOLO Pose: {paths.yolo_model}")
        self.model = YOLO(str(paths.yolo_model))

        self.spatial_logic = SpatialAnalyzer(config_path=paths.coords_json)
        self._behavior_classifier = BehaviorClassifier(spatial_logic=self.spatial_logic)

    @staticmethod
    def _get_color(label: str) -> tuple[int, int, int]:
        """Devuelve el color BGR asociado a cada etiqueta de comportamiento."""
        label = label.lower()
        if "sniffing_immobile" in label: return (200, 100, 180)
        if "sniffing"   in label: return (0,   200, 255)
        if "immobile"   in label: return (180, 180, 180)
        if "walking"    in label: return (0,   255, 255)
        if "climbing"   in label: return (255,   0, 255)
        if "dipping"    in label: return (0,   165, 255)
        if "rearing"    in label: return (0,   255,   0)
        if "grooming"   in label: return (180, 255, 180)
        return (128, 128, 128)

    @staticmethod
    def _extract_snout(res, detection_idx: int = 0) -> np.ndarray | None:
        """
        Extrae las coordenadas (x, y) del snout (keypoint 0) del raton indicado.
        Devuelve None si YOLO no produce keypoints o la confianza es menor de 0.3.
        """
        if res.keypoints is None:
            return None
        kps_xy   = res.keypoints.xy
        kps_conf = res.keypoints.conf

        if len(kps_xy) <= detection_idx:
            return None

        snout = kps_xy[detection_idx][KP_SNOUT].cpu().numpy()

        conf_val = 1.0
        if kps_conf is not None:
            conf_val = float(kps_conf[detection_idx][KP_SNOUT].cpu())
            if conf_val < 0.3:
                return None

        if snout[0] < 1.0 and snout[1] < 1.0:
            return None

        # Devuelve [x, y, conf] para que el clasificador pueda ponderar la fiabilidad
        return np.array([snout[0], snout[1], conf_val], dtype=float)

    @staticmethod
    def _extract_keypoint(res, kp_idx: int, detection_idx: int = 0,
                          conf_threshold: float = 0.3) -> np.ndarray | None:
        """Extrae las coordenadas (x, y) de un keypoint generico por indice."""
        if res.keypoints is None:
            return None
        kps_xy   = res.keypoints.xy
        kps_conf = res.keypoints.conf
        if len(kps_xy) <= detection_idx:
            return None
        kp = kps_xy[detection_idx][kp_idx].cpu().numpy()
        if kps_conf is not None:
            if float(kps_conf[detection_idx][kp_idx].cpu()) < conf_threshold:
                return None
        if kp[0] < 1.0 and kp[1] < 1.0:
            return None
        return kp

    @staticmethod
    def _draw_skeleton(img: np.ndarray, res, detection_idx: int = 0) -> None:
        """Dibuja los keypoints snout/spine/tail y sus conexiones sobre img."""
        if res.keypoints is None:
            return
        kps_xy   = res.keypoints.xy
        kps_conf = res.keypoints.conf
        if len(kps_xy) <= detection_idx:
            return

        kps  = kps_xy[detection_idx].cpu().numpy()
        conf = kps_conf[detection_idx].cpu().numpy() if kps_conf is not None else None

        kp_colors = [(0, 0, 255), (0, 255, 0), (255, 80, 0)]
        kp_names  = ["snout", "spine", "tail"]

        visible: list[bool] = []
        for i, (kp, color, name) in enumerate(zip(kps, kp_colors, kp_names)):
            c = float(conf[i]) if conf is not None else 1.0
            if c < 0.3 or (kp[0] < 1.0 and kp[1] < 1.0):
                visible.append(False)
                continue
            visible.append(True)
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(img, (x, y), 5, color, -1)
            cv2.putText(img, name, (x + 6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        connections: list[tuple[int, int]] = [(KP_SNOUT, KP_SPINE), (KP_SPINE, KP_TAIL)]
        for a, b in connections:
            if a < len(kps) and b < len(kps) and len(visible) > max(a, b):
                if visible[a] and visible[b]:
                    pa = (int(kps[a][0]), int(kps[a][1]))
                    pb = (int(kps[b][0]), int(kps[b][1]))
                    cv2.line(img, pa, pb, (0, 220, 255), 2)

    def run(self) -> None:
        """
        Procesa el video (o camara en vivo) y escribe el video anotado y el CSV.
        Si camera_index no es None, usa la camara en lugar del archivo de video.
        """
        self._setup()

        # Determinar la fuente: archivo de video o indice de camara
        if self.camera_index is not None:
            source_cv2 = self.camera_index          # cv2.VideoCapture(0)
            source_yolo = self.camera_index         # model.predict(source=0, ...)
            source_name = f"camara [{self.camera_index}]"
        else:
            source_cv2  = str(paths.video_source)
            source_yolo = str(paths.video_source)
            source_name = paths.video_source.name

        cap = cv2.VideoCapture(source_cv2)
        if not cap.isOpened():
            print(f"[X] Error abriendo fuente: {source_name}")
            return

        fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w: int     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h: int     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        vid_out = VideoOutput(paths.output_video, fps, w, h, dual_output=self.dual_output)
        csv_out = CsvOutput(paths.output_video.with_suffix(".csv"))

        print(f"[>] Procesando: {source_name}")
        print(f"    Salida    : {paths.output_video}")
        if vid_out.clean_path is not None:
            print(f"    Limpio    : {vid_out.clean_path}")

        print("\n" + "=" * 55)
        print("  LEYENDA")
        for k, v in LABEL_ES.items():
            print(f"  {k:<22}  {v}")
        print("=" * 55 + "\n")

        spatial_ok: bool = self.spatial_logic.is_valid_for(w, h)
        if not spatial_ok:
            print(f"[!] AVISO: coords.json no esta calibrado para esta fuente ({w}x{h}).")
            print(f"    head_dipping y sniffing no usaran referencias espaciales.")

        results = self.model.predict(
            source=source_yolo, stream=True,
            conf=self.cfg.conf_threshold, device=self.cfg.device, iou=0.5
        )

        frame_idx: int  = 0
        last_label: str = "—"

        for res in results:
            img_base  = res.orig_img.copy()
            img       = img_base.copy()
            img_clean: np.ndarray | None = img_base.copy() if self.dual_output else None

            if spatial_ok:
                self.spatial_logic.draw_zones(img)

            rat_box: np.ndarray | None = None
            yolo_label: str  = "Unknown"
            final_label: str = "Unknown"
            snout_kp: np.ndarray | None = None
            tail_kp:  np.ndarray | None = None
            speed_val: float = 0.0

            # 1. Extraer la deteccion de mayor confianza de YOLO
            if res.boxes and len(res.boxes) > 0:
                confs      = res.boxes.conf.cpu().numpy()
                best       = int(np.argmax(confs))
                rat_box    = res.boxes.xyxy[best].cpu().numpy()
                cls_id     = int(res.boxes.cls[best].cpu())
                yolo_label = self.model.names.get(cls_id, "Unknown")

            # 2. Extraer keypoints del snout y la cola
            if rat_box is not None and res.keypoints is not None:
                snout_kp = self._extract_snout(res, detection_idx=0)
                tail_kp  = self._extract_keypoint(res, KP_TAIL, detection_idx=0)

            # 3. Calcular velocidad y delegar clasificacion
            if rat_box is not None:
                speed_val   = self._speed_tracker.update(rat_box, w, h)
                final_label = self._behavior_classifier.classify(
                    yolo_label, speed_val, snout_kp, rat_box, spatial_ok
                )

                # Dibujar bbox y etiqueta
                color = self._get_color(final_label)
                rx1, ry1, rx2, ry2 = map(int, rat_box)

                if final_label.startswith("sniffing_"):
                    motion = final_label.split("_", 1)[1]
                    label_txt = f"sniffing [{motion}]"
                else:
                    display = final_label.removeprefix("rat_").replace("_", " ")
                    yolo_ruido = yolo_label in ("rat_horizontal", "Unknown")
                    if not yolo_ruido and yolo_label != final_label:
                        yolo_disp = yolo_label.removeprefix("rat_").replace("_", " ")
                        label_txt = f"{display} [{yolo_disp}]"
                    else:
                        label_txt = display

                for target in ([img] + ([img_clean] if img_clean is not None else [])):
                    cv2.rectangle(target, (rx1, ry1), (rx2, ry2), color, 2)
                    cv2.putText(target, label_txt, (rx1, ry1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if self.show_skeleton:
                        self._draw_skeleton(target, res, detection_idx=0)

                csv_out.write_row(frame_idx, fps, yolo_label, final_label,
                                  rat_box, snout_kp, tail_kp, speed_val)
                last_label = final_label

            vid_out.write(img, img_clean)

            if self.show_preview:
                preview = img.copy()
                skel_state = "ON" if self.show_skeleton else "OFF"
                hint = f"[K] Skeleton: {skel_state}   [Q] Salir"
                cv2.putText(preview, hint, (10, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
                cv2.imshow("RatDetector - Preview", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('k') or key == ord('K'):
                    self.show_skeleton = not self.show_skeleton
                    print(f"\n[Preview] Skeleton {'ON' if self.show_skeleton else 'OFF'}")
                elif key == ord('q') or key == ord('Q'):
                    print("\n[Preview] Detencion manual (Q)")
                    break

            frame_idx += 1

            if frame_idx % 30 == 0:
                es = LABEL_ES.get(last_label, last_label)
                print(f"   Frame {frame_idx:>6}  |  {last_label:<22}  ({es})", end="\r")

        if self.show_preview:
            cv2.destroyAllWindows()
        vid_out.release()
        csv_out.close()
        print(f"\n[+] Finalizado. {frame_idx} frames -> {paths.output_video.name}")
        if vid_out.clean_path is not None:
            print(f"    Limpio    : {vid_out.clean_path.name}")
        print(f"    CSV: {csv_out.path.name}")
