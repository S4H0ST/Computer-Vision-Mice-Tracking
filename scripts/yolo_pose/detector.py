import cv2
import csv
import numpy as np
from collections import deque
from ultralytics import YOLO
from pathlib import Path
from typing import Optional, Tuple

from config.interfaces import BaseModule
from config.config import paths, DetectParams
from rnn.inference import ActionPredictor
from logic.spatial import SpatialAnalyzer


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
WALK_SPEED_THRESHOLD   = 0.35   # por encima → walking  (calibrado p60 speed distribución)
STILL_SPEED_THRESHOLD  = 0.15   # por debajo → immobile (calibrado p25 speed distribución)
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

    # Salto máximo plausible entre frames a 15fps en caja normalizada.
    # Un salto mayor indica detección errónea — se ignora preservando la última velocidad.
    _MAX_PLAUSIBLE_SPEED = 8.0

    def update(self, box, img_w: int, img_h: int) -> float:
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

    def reset(self):
        self._prev = None
        self._history.clear()


class RatDetector(BaseModule):

    def __init__(self, config: DetectParams,
                 show_skeleton: bool = False,
                 show_preview: bool = True,
                 dual_output: bool = False):
        self.cfg             = config
        self.show_skeleton   = show_skeleton
        self.show_preview    = show_preview
        self.dual_output     = dual_output   # genera también un video limpio (sin zonas)
        self.model: YOLO     = None
        self.rnn_brain       = None
        self.spatial_logic   = None
        self._speed_tracker  = _SpeedTracker(smoothing=5)

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
    @staticmethod
    def _draw_skeleton(img: np.ndarray, res, detection_idx: int = 0) -> None:
        if res.keypoints is None:
            return
        kps_xy   = res.keypoints.xy
        kps_conf = res.keypoints.conf
        if len(kps_xy) <= detection_idx:
            return

        kps  = kps_xy[detection_idx].cpu().numpy()   # (K, 2)
        conf = kps_conf[detection_idx].cpu().numpy() if kps_conf is not None else None

        kp_colors = [(0, 0, 255), (0, 255, 0), (255, 80, 0)]   # snout=rojo, spine=verde, tail=azul
        kp_names  = ["snout", "spine", "tail"]

        visible = []
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

        # Líneas de conexión: snout→spine→tail
        connections = [(KP_SNOUT, KP_SPINE), (KP_SPINE, KP_TAIL)]
        for a, b in connections:
            if a < len(kps) and b < len(kps) and len(visible) > max(a, b):
                if visible[a] and visible[b]:
                    pa = (int(kps[a][0]), int(kps[a][1]))
                    pb = (int(kps[b][0]), int(kps[b][1]))
                    cv2.line(img, pa, pb, (0, 220, 255), 2)

    # ------------------------------------------------------------------ #
    def _derive_horizontal(self,
                           snout_kp: Optional[np.ndarray],
                           rat_box: np.ndarray,
                           img_w: int,
                           img_h: int) -> str:
        """
        Desambigua rat_horizontal (y climbing mal clasificado) en:
          1. sniffing   — snout cerca de pared interior
          2. walking    — desplazamiento rápido
          3. immobile   — desplazamiento mínimo

        Prioridad: sniffing > walking/immobile.
        El speed tracker ya fue actualizado en el bucle principal antes de llamar aquí.
        """
        # 1. Sniffing: snout cerca de pared interior (prioridad sobre velocidad)
        if snout_kp is not None and self.spatial_logic.check_sniffing_wall(snout_kp):
            return "sniffing"

        # 2. RNN temporal (si está entrenada)
        rnn_pred = self.rnn_brain.update_and_predict(rat_box, img_w, img_h)
        if rnn_pred and rnn_pred not in ("Analyzing...", "rat_horizontal"):
            return rnn_pred

        # 3. Velocidad del centroide (ya suavizada en el tracker)
        speed = float(np.mean(self._speed_tracker._history)) if self._speed_tracker._history else 0.0
        if speed >= WALK_SPEED_THRESHOLD:
            return "walking"
        if speed <= STILL_SPEED_THRESHOLD:
            return "immobile"

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

        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        out_vid  = cv2.VideoWriter(str(paths.output_video), fourcc, fps, (w, h))

        # Video limpio (sin superposición de zonas) — para presentación
        out_clean     = None
        clean_path    = None
        if self.dual_output:
            clean_path = paths.output_video.with_name(
                paths.output_video.stem + "_limpio.mp4"
            )
            out_clean = cv2.VideoWriter(str(clean_path), fourcc, fps, (w, h))

        csv_path = paths.output_video.with_suffix(".csv")
        f_csv    = open(csv_path, "w", newline="")
        writer   = csv.writer(f_csv)
        writer.writerow(["frame", "time_s", "yolo_label", "final_label",
                         "x1", "y1", "x2", "y2",
                         "snout_x", "snout_y", "speed"])

        print(f"[>] Procesando: {paths.video_source.name}")
        print(f"    Salida    : {paths.output_video}")
        if out_clean is not None:
            print(f"    Limpio    : {clean_path}")

        # Imprimir leyenda
        print("\n" + "=" * 55)
        print("  LEYENDA")
        for k, v in LABEL_ES.items():
            print(f"  {k:<22}  {v}")
        print("=" * 55 + "\n")

        # Verificar si la calibración es válida para este vídeo
        spatial_ok = self.spatial_logic.is_valid_for(w, h)
        if not spatial_ok:
            print(f"[!] AVISO: coords.json no está calibrado para este vídeo ({w}×{h}).")
            print(f"    head_dipping y sniffing no usarán referencias espaciales.")
            print(f"    Ejecuta: python calibrate.py --video <video.mp4>")

        results = self.model.predict(
            source=str(paths.video_source), stream=True,
            conf=self.cfg.conf_threshold, device=self.cfg.device, iou=0.5
        )

        frame_idx  = 0
        last_label = "—"

        for res in results:
            img_base  = res.orig_img.copy()
            img       = img_base.copy()              # con zonas + anotaciones
            img_clean = img_base.copy() if self.dual_output else None

            # Dibujar zonas calibradas solo en el frame principal (no en el limpio)
            if spatial_ok:
                self.spatial_logic.draw_zones(img)

            rat_box      = None
            yolo_label   = "Unknown"
            final_label  = "Unknown"
            snout_kp     = None
            speed_val    = 0.0

            # ── 1. Extraer detecciones de YOLO ──────────────────────────
            if res.boxes and len(res.boxes) > 0:
                confs      = res.boxes.conf.cpu().numpy()
                best       = int(np.argmax(confs))
                rat_box    = res.boxes.xyxy[best].cpu().numpy()
                cls_id     = int(res.boxes.cls[best].cpu())
                yolo_label = self.model.names.get(cls_id, "Unknown")

            # ── 2. Extraer snout keypoint ────────────────────────────────
            if rat_box is not None and res.keypoints is not None:
                snout_kp = self._extract_snout(res, detection_idx=0)

            # ── 3. Lógica híbrida ────────────────────────────────────────
            if rat_box is not None:
                # Actualizar speed tracker en TODOS los frames con detección
                # (no solo en horizontal) para que la historia sea continua.
                speed_val = self._speed_tracker.update(rat_box, w, h)

                final_label = yolo_label  # base: confiar en YOLO

                # A) HEAD DIPPING — solo si calibración válida y snout sobre agujero
                if spatial_ok and snout_kp is not None and self.spatial_logic.check_dipping(snout_kp):
                    final_label = "rat_head_dipping"

                # A2) YOLO dice head_dipping pero snout no está en agujero
                #     → reclasificar y desambiguar con velocidad/sniffing
                elif yolo_label == "rat_head_dipping":
                    if not spatial_ok or snout_kp is None:
                        final_label = "rat_horizontal"
                    elif not self.spatial_logic.check_dipping(snout_kp):
                        final_label = "rat_horizontal"

                # B) Desambiguar horizontal → walking / immobile / sniffing
                if final_label == "rat_horizontal":
                    final_label = self._derive_horizontal(snout_kp, rat_box, w, h)

                # C) climbing: confirmado solo si el snout está en la zona de pared
                #    (fuera del área interior). Climbing real = rata pegada a la pared
                #    con el hocico fuera del interior.
                #    Si el snout está dentro del interior → no es climbing real,
                #    desambiguar como horizontal/sniffing/walking/immobile.
                elif yolo_label == "rat_climbing" and spatial_ok:
                    snout_in_wall_zone = (
                        snout_kp is not None and
                        not self.spatial_logic.is_inside_inner(snout_kp)
                    )
                    if snout_in_wall_zone:
                        final_label = "rat_climbing"   # confirmado: hocico en la pared
                    else:
                        final_label = self._derive_horizontal(snout_kp, rat_box, w, h)

                # D) Para rearing / grooming: RNN puede refinar si está activa
                elif yolo_label in ("rat_rearing", "rat_grooming"):
                    rnn_pred = self.rnn_brain.update_and_predict(rat_box, w, h)
                    if rnn_pred and rnn_pred not in ("Analyzing...", yolo_label):
                        final_label = rnn_pred

                # ── Dibujar ─────────────────────────────────────────────
                color = self._get_color(final_label)
                rx1, ry1, rx2, ry2 = map(int, rat_box)
                label_txt = final_label
                if yolo_label != final_label and yolo_label != "Unknown":
                    label_txt = f"{final_label} [{yolo_label}]"

                for target in ([img] + ([img_clean] if img_clean is not None else [])):
                    cv2.rectangle(target, (rx1, ry1), (rx2, ry2), color, 2)
                    cv2.putText(target, label_txt, (rx1, ry1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if self.show_skeleton:
                        self._draw_skeleton(target, res, detection_idx=0)

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
            if out_clean is not None:
                out_clean.write(img_clean)

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
                    estado = "ON" if self.show_skeleton else "OFF"
                    print(f"\n[Preview] Skeleton {estado}")
                elif key == ord('q') or key == ord('Q'):
                    print("\n[Preview] Detención manual (Q)")
                    break

            frame_idx += 1

            if frame_idx % 30 == 0:
                es = LABEL_ES.get(last_label, last_label)
                print(f"   Frame {frame_idx:>6}  |  {last_label:<22}  ({es})", end="\r")

        if self.show_preview:
            cv2.destroyAllWindows()
        out_vid.release()
        if out_clean is not None:
            out_clean.release()
        f_csv.close()
        print(f"\n[+] Finalizado. {frame_idx} frames -> {paths.output_video.name}")
        if out_clean is not None:
            print(f"    Limpio    : {clean_path.name}")
        print(f"    CSV: {csv_path.name}")
