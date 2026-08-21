"""
Detector principal de comportamiento del raton usando YOLO Pose + logica hibrida.

Clases:
    _SpeedTracker — calcula y suaviza la velocidad del centroide entre frames.
    RatDetector   — orquesta la inferencia YOLO, la logica espacial, la RNN y la escritura
                    de video y CSV con la etiqueta de comportamiento final.
"""

import cv2
import csv
import numpy as np
from collections import deque
from ultralytics import YOLO
from pathlib import Path

from config.interfaces import BaseModule
from config.config import paths, DetectParams
from spatial.spatial import SpatialAnalyzer


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

# Umbrales de velocidad para walking vs immobile.
# Speed = desplazamiento normalizado del centroide x 100 (igual que en la RNN).
WALK_SPEED_THRESHOLD: float  = 0.35   # por encima -> walking
STILL_SPEED_THRESHOLD: float = 0.15   # por debajo -> immobile
# Entre ambos umbrales el estado es ambiguo: se mantiene la etiqueta de YOLO.

# Aspect ratio minimo (altura/ancho del bbox) para confirmar rearing.
# Un raton erguido tiene el bbox claramente mas alto que ancho (ratio > 1.0).
# Un raton inmovil/horizontal tiene el bbox mas ancho que alto (ratio ~ 0.4-0.7).
# Con 0.80 se acepta rearing cuando la caja es casi cuadrada o mas alta que ancha,
# descartando poses claramente horizontales que YOLO confunde con rearing.
REARING_ASPECT_RATIO: float = 0.70

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
    # Un salto mayor indica deteccion erronea — se descarta preservando la ultima velocidad.
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


class _LabelStabilizer:
    """
    Evita el parpadeo de etiquetas en el video aplicando histéresis temporal.

    Una etiqueta solo reemplaza a la actual si aparece durante al menos
    `hold_frames` frames consecutivos. Las etiquetas en TRANSPARENT se tratan
    como señal nula: no activan el temporizador de cambio ni se muestran en video.
    """

    TRANSPARENT: frozenset[str] = frozenset({"rat_horizontal", "Unknown"})

    def __init__(self, hold_frames: int = 8,
                 fast_labels: dict[str, int] | None = None) -> None:
        self._stable: str    = ""
        self._candidate: str = ""
        self._count: int     = 0
        self._hold: int      = hold_frames
        # Etiquetas con umbral de confirmacion mas bajo (respuesta mas rapida)
        self._fast: dict[str, int] = fast_labels or {}

    def update(self, label: str) -> str:
        # Etiquetas transparentes: devolver lo que hay estable (o la propia si no hay nada aun)
        if label in self.TRANSPARENT:
            return self._stable if self._stable else label

        # Primera etiqueta real: aceptar directamente sin esperar
        if not self._stable:
            self._stable = label
            return self._stable

        # Misma etiqueta que la estable: confirmar, resetear candidato
        if label == self._stable:
            self._candidate = ""
            self._count = 0
            return self._stable

        # Umbral de confirmacion: fast_labels aplica en ambas direcciones.
        # Si la etiqueta entrante ES rapida, la acepta pronto.
        # Si la etiqueta estable actual ES rapida, tambien se sale de ella pronto.
        required = self._fast.get(label, self._hold)
        if self._stable in self._fast:
            required = min(required, self._fast[self._stable])

        # Acumular o cambiar candidato
        if label == self._candidate:
            self._count += 1
            if self._count >= required:
                self._stable    = self._candidate
                self._candidate = ""
                self._count     = 0
        else:
            self._candidate = label
            self._count     = 1

        return self._stable


class RatDetector(BaseModule):
    """
    Detecta y clasifica el comportamiento del raton frame a frame.
    Combina YOLO Pose, la RNN de movimiento y la logica espacial calibrada.
    """

    def __init__(self, config: DetectParams,
                 show_skeleton: bool = False,
                 show_preview: bool = True,
                 dual_output: bool = False) -> None:
        self.cfg: DetectParams           = config
        self.show_skeleton: bool         = show_skeleton
        self.show_preview: bool          = show_preview
        self.dual_output: bool           = dual_output
        self.model: YOLO | None                    = None
        self.spatial_logic: SpatialAnalyzer | None = None
        self._speed_tracker: _SpeedTracker       = _SpeedTracker(smoothing=5)
        self._label_stabilizer: _LabelStabilizer  = _LabelStabilizer(
            hold_frames=8,
            # fast_labels aplica en entrada Y salida (bidireccional)
            fast_labels={"rat_climbing": 3, "rat_head_dipping": 3,
                         "rat_grooming": 3, "rat_rearing": 3},
        )

    def _setup(self) -> None:
        """Carga el modelo YOLO, la RNN y el analizador espacial."""
        if not paths.yolo_model.exists():
            raise FileNotFoundError(f"Modelo YOLO no encontrado: {paths.yolo_model}")

        print(f"[Core] Cargando YOLO Pose: {paths.yolo_model}")
        self.model = YOLO(str(paths.yolo_model))

        self.spatial_logic = SpatialAnalyzer(config_path=paths.coords_json)

    @staticmethod
    def _get_color(label: str) -> tuple[int, int, int]:
        """Devuelve el color BGR asociado a cada etiqueta de comportamiento."""
        label = label.lower()
        # sniffing_immobile antes del check generico "sniffing" para que no
        # quede capturado por el y tome el color equivocado.
        if "sniffing_immobile" in label: return (200, 100, 180)  # lila/orquidea
        if "sniffing"   in label: return (0,   200, 255)  # ambar/dorado  (sniffing_walking)
        if "immobile"   in label: return (180, 180, 180)  # gris
        if "walking"    in label: return (0,   255, 255)  # amarillo
        if "climbing"   in label: return (255,   0, 255)  # magenta
        if "dipping"    in label: return (0,   165, 255)  # naranja oscuro
        if "rearing"    in label: return (0,   255,   0)  # verde
        if "grooming"   in label: return (180, 255, 180)  # verde claro
        return (128, 128, 128)

    @staticmethod
    def _extract_snout(res, detection_idx: int = 0) -> np.ndarray | None:
        """
        Extrae las coordenadas (x, y) del snout (keypoint 0) del raton indicado.
        Devuelve None si YOLO no produce keypoints o la confianza es menor de 0.3.
        """
        if res.keypoints is None:
            return None
        kps_xy   = res.keypoints.xy    # (N, K, 2)
        kps_conf = res.keypoints.conf  # (N, K) o None

        if len(kps_xy) <= detection_idx:
            return None

        snout = kps_xy[detection_idx][KP_SNOUT].cpu().numpy()

        if kps_conf is not None:
            conf = float(kps_conf[detection_idx][KP_SNOUT].cpu())
            if conf < 0.3:
                return None

        # YOLO representa keypoints no visibles como (0, 0)
        if snout[0] < 1.0 and snout[1] < 1.0:
            return None

        return snout

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

        kp_colors = [(0, 0, 255), (0, 255, 0), (255, 80, 0)]  # snout=rojo, spine=verde, tail=azul
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

    def _derive_horizontal(self, snout_kp: np.ndarray | None) -> str:
        """
        Desambigua rat_horizontal en:
          1. sniffing  — snout cerca de pared interior (prioridad)
          2. walking   — desplazamiento rapido del centroide
          3. immobile  — desplazamiento minimo

        El speed tracker debe haberse actualizado antes de llamar a este metodo.
        """
        if snout_kp is not None and self.spatial_logic.check_sniffing_wall(snout_kp):
            return "sniffing"

        speed = float(np.mean(self._speed_tracker._history)) if self._speed_tracker._history else 0.0
        if speed >= WALK_SPEED_THRESHOLD:
            return "walking"
        if speed <= STILL_SPEED_THRESHOLD:
            return "immobile"

        return "rat_horizontal"

    def run(self) -> None:
        """Procesa el video completo y escribe el video anotado y el CSV de resultados."""
        self._setup()

        cap = cv2.VideoCapture(str(paths.video_source))
        if not cap.isOpened():
            print(f"[X] Error abriendo video: {paths.video_source}")
            return

        fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w: int     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h: int     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        out_vid = cv2.VideoWriter(str(paths.output_video), fourcc, fps, (w, h))

        # Video limpio (sin superposicion de zonas) para presentacion
        out_clean: cv2.VideoWriter | None = None
        clean_path: Path | None = None
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
                         "snout_x", "snout_y", "speed", "tail_x", "tail_y"])

        print(f"[>] Procesando: {paths.video_source.name}")
        print(f"    Salida    : {paths.output_video}")
        if out_clean is not None:
            print(f"    Limpio    : {clean_path}")

        print("\n" + "=" * 55)
        print("  LEYENDA")
        for k, v in LABEL_ES.items():
            print(f"  {k:<22}  {v}")
        print("=" * 55 + "\n")

        spatial_ok: bool = self.spatial_logic.is_valid_for(w, h)
        if not spatial_ok:
            print(f"[!] AVISO: coords.json no esta calibrado para este video ({w}x{h}).")
            print(f"    head_dipping y sniffing no usaran referencias espaciales.")
            print(f"    Ejecuta: python calibrate.py --video <video.mp4>")

        results = self.model.predict(
            source=str(paths.video_source), stream=True,
            conf=self.cfg.conf_threshold, device=self.cfg.device, iou=0.5
        )

        frame_idx: int  = 0
        last_label: str = "—"

        for res in results:
            img_base  = res.orig_img.copy()
            img       = img_base.copy()
            img_clean: np.ndarray | None = img_base.copy() if self.dual_output else None

            # Las zonas calibradas solo se dibujan en el video principal
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
                confs   = res.boxes.conf.cpu().numpy()
                best    = int(np.argmax(confs))
                rat_box = res.boxes.xyxy[best].cpu().numpy()
                cls_id  = int(res.boxes.cls[best].cpu())
                yolo_label = self.model.names.get(cls_id, "Unknown")

            # 2. Extraer keypoints del snout y la cola
            if rat_box is not None and res.keypoints is not None:
                snout_kp = self._extract_snout(res, detection_idx=0)
                tail_kp  = self._extract_keypoint(res, KP_TAIL, detection_idx=0)

            # 3. Logica hibrida de clasificacion
            if rat_box is not None:
                # Actualizar el tracker en TODOS los frames con deteccion
                # para que el historial de velocidad sea continuo
                speed_val = self._speed_tracker.update(rat_box, w, h)

                final_label = yolo_label

                # A) HEAD DIPPING — el snout cae dentro del radio del agujero.
                #    YOLOv8 Pose siempre estima la posicion del snout aunque este
                #    ocluido (lo infiere del contexto corporal), por lo que la
                #    confianza del keypoint no es una senal fiable de oclusión.
                #    El trigger correcto es espacial: si el snout estimado cae
                #    dentro del radio del agujero, el raton esta haciendo dipping.
                if spatial_ok and snout_kp is not None and self.spatial_logic.check_dipping(snout_kp):
                    final_label = "rat_head_dipping"

                # A2) YOLO dice head_dipping pero el snout no esta sobre un agujero: reclasificar.
                elif yolo_label == "rat_head_dipping":
                    if not spatial_ok or snout_kp is None:
                        final_label = "rat_horizontal"
                    elif not self.spatial_logic.check_dipping(snout_kp):
                        final_label = "rat_horizontal"

                # B) Desambiguar horizontal -> walking / immobile / sniffing
                if final_label == "rat_horizontal":
                    final_label = self._derive_horizontal(snout_kp)

                # C) Climbing: confirmado si el bbox penetra al menos 15px en la zona
                #    de pared (entre borde interior y exterior). El margen evita falsos
                #    positivos cuando el raton camina cerca del borde sin trepar.
                elif yolo_label == "rat_climbing" and spatial_ok:
                    x1c, y1c, x2c, y2c = rat_box
                    if not self.spatial_logic.bbox_entirely_inside(x1c, y1c, x2c, y2c):
                        final_label = "rat_climbing"
                    else:
                        final_label = self._derive_horizontal(snout_kp)

                # D) Rearing: confirmado por aspect ratio del bbox (altura/ancho).
                #    Un raton erguido tiene el bbox claramente mas alto que ancho.
                #    Si el bbox es horizontal (ratio < REARING_ASPECT_RATIO) YOLO
                #    ha confundido una pose inmovil o caminando con rearing.
                elif yolo_label == "rat_rearing":
                    x1r, y1r, x2r, y2r = rat_box
                    h_box = y2r - y1r
                    w_box = x2r - x1r
                    aspect = h_box / w_box if w_box > 0 else 1.0
                    if aspect >= REARING_ASPECT_RATIO:
                        final_label = "rat_rearing"
                    else:
                        final_label = self._derive_horizontal(snout_kp)

                # E) Climbing recovery eliminado: exp8 tiene recall=0.887 y detecta
                #    climbing de forma nativa. La recovery rule causaba falsos positivos
                #    cuando el raton caminaba cerca de la pared (bbox rozando el borde
                #    interior). Solo se reintroduciria si se usa un modelo con recall bajo.

                # Estabilizacion temporal: la etiqueta solo cambia tras hold_frames
                # consecutivos con la misma senyal. rat_horizontal es transparente
                # (no aparece en video ni activa el temporizador de cambio).
                final_label = self._label_stabilizer.update(final_label)

                # Sub-estado de sniffing: distingue si el raton se mueve o esta parado.
                # Se aplica DESPUES del estabilizador (que trabaja con "sniffing" como
                # unidad) para que el movimiento se muestre en tiempo real sin retardo.
                if final_label == "sniffing":
                    spd = float(np.mean(self._speed_tracker._history)) if self._speed_tracker._history else 0.0
                    final_label = "sniffing_walking" if spd >= WALK_SPEED_THRESHOLD else "sniffing_immobile"

                # Dibujar bbox y etiqueta
                color = self._get_color(final_label)
                rx1, ry1, rx2, ry2 = map(int, rat_box)

                # Formato de texto para la superposicion:
                # - rat_ se elimina del prefijo para que sea mas limpio visualmente
                # - sniffing_* -> "sniffing [walking]" / "sniffing [immobile]"
                # - resto       -> "nombre [yolo_nombre]" si difieren y yolo no es ruido
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

                # Escribir fila en el CSV
                snout_x = float(snout_kp[0]) if snout_kp is not None else -1.0
                snout_y = float(snout_kp[1]) if snout_kp is not None else -1.0
                tail_x  = float(tail_kp[0])  if tail_kp  is not None else -1.0
                tail_y  = float(tail_kp[1])  if tail_kp  is not None else -1.0
                writer.writerow([frame_idx, f"{frame_idx / fps:.2f}",
                                 yolo_label, final_label,
                                 rx1, ry1, rx2, ry2,
                                 f"{snout_x:.1f}", f"{snout_y:.1f}",
                                 f"{speed_val:.3f}",
                                 f"{tail_x:.1f}", f"{tail_y:.1f}"])
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
                    print("\n[Preview] Detencion manual (Q)")
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
