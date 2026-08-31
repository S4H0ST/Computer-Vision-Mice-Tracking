"""
Clasificador de comportamiento basado en logica hibrida (YOLO + velocidad + espacio).

Clases:
    _LabelStabilizer   — histeresis temporal para evitar parpadeo de etiquetas.
    BehaviorClassifier — aplica la cadena de reglas (head_dipping, rearing,
                         climbing, walking/immobile/sniffing) y estabiliza la salida.
"""

import numpy as np
from spatial.spatial import SpatialAnalyzer


WALK_SPEED_THRESHOLD: float  = 0.35   # por encima -> walking
STILL_SPEED_THRESHOLD: float = 0.15   # por debajo -> immobile
REARING_ASPECT_RATIO: float  = 0.70   # altura/ancho minimo para confirmar rearing


class _LabelStabilizer:
    """
    Evita el parpadeo de etiquetas en el video aplicando histeresis temporal.

    Una etiqueta solo reemplaza a la actual si aparece durante al menos
    `hold_frames` frames consecutivos. Las etiquetas en TRANSPARENT se tratan
    como senal nula: no activan el temporizador de cambio ni se muestran en video.
    """

    TRANSPARENT: frozenset[str] = frozenset({"rat_horizontal", "Unknown"})

    def __init__(self, hold_frames: int = 8,
                 fast_labels: dict[str, int] | None = None) -> None:
        self._stable: str    = ""
        self._candidate: str = ""
        self._count: int     = 0
        self._hold: int      = hold_frames
        self._fast: dict[str, int] = fast_labels or {}

    def update(self, label: str) -> str:
        if label in self.TRANSPARENT:
            return self._stable if self._stable else label

        if not self._stable:
            self._stable = label
            return self._stable

        if label == self._stable:
            self._candidate = ""
            self._count = 0
            return self._stable

        required = self._fast.get(label, self._hold)
        if self._stable in self._fast:
            required = min(required, self._fast[self._stable])

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


class BehaviorClassifier:
    """
    Aplica la logica hibrida de clasificacion sobre la etiqueta YOLO bruta.
    Mantiene estado interno para la estabilizacion temporal de etiquetas.

    Uso:
        clf = BehaviorClassifier(spatial_logic=analyzer)
        for frame in ...:
            label = clf.classify(yolo_label, speed, snout_kp, rat_box, spatial_ok)
    """

    def __init__(self, spatial_logic: SpatialAnalyzer | None = None,
                 hold_frames: int = 8) -> None:
        self._spatial = spatial_logic
        self._stabilizer = _LabelStabilizer(
            hold_frames=hold_frames,
            fast_labels={"rat_climbing": 3, "rat_head_dipping": 3,
                         "rat_grooming": 3, "rat_rearing": 3},
        )

    def _derive_horizontal(self, snout_kp: np.ndarray | None, speed: float) -> str:
        """
        Desambigua rat_horizontal en sniffing / walking / immobile.
        Prioridad: sniffing_wall > walking > immobile.
        """
        if (snout_kp is not None and self._spatial is not None
                and self._spatial.check_sniffing_wall(snout_kp)):
            return "sniffing"

        if speed >= WALK_SPEED_THRESHOLD:
            return "walking"
        if speed <= STILL_SPEED_THRESHOLD:
            return "immobile"

        # zona ambigua: mantener como horizontal para que el estabilizador
        # lo trate como senal nula y conserve la etiqueta previa estable.
        return "rat_horizontal"

    def classify(self, yolo_label: str, speed: float,
                 snout_kp: np.ndarray | None, rat_box: np.ndarray,
                 spatial_ok: bool) -> str:
        """
        Devuelve la etiqueta final de comportamiento para el frame actual.

        yolo_label : clase predicha por YOLO Pose.
        speed      : velocidad suavizada del centroide (misma escala que _SpeedTracker).
        snout_kp   : coordenadas (x, y) del keypoint snout, o None si no esta visible.
        rat_box    : array [x1, y1, x2, y2] en pixeles del bounding box.
        spatial_ok : True si coords.json esta calibrado para las dimensiones del video.
        """
        final_label = yolo_label

        # A) HEAD DIPPING — el snout cae dentro del radio del agujero.
        if spatial_ok and self._spatial is not None and snout_kp is not None:
            if self._spatial.check_dipping(snout_kp):
                final_label = "rat_head_dipping"

        # A2) YOLO dice head_dipping pero el snout no confirma: reclasificar.
        elif yolo_label == "rat_head_dipping":
            if not spatial_ok or snout_kp is None:
                final_label = "rat_horizontal"
            elif self._spatial is not None and not self._spatial.check_dipping(snout_kp):
                final_label = "rat_horizontal"

        # B) Desambiguar horizontal -> walking / immobile / sniffing
        if final_label == "rat_horizontal":
            final_label = self._derive_horizontal(snout_kp, speed)

        # C) Climbing: confirmado si el bbox penetra en la zona de pared.
        elif yolo_label == "rat_climbing" and spatial_ok and self._spatial is not None:
            x1, y1, x2, y2 = rat_box
            if not self._spatial.bbox_entirely_inside(x1, y1, x2, y2):
                final_label = "rat_climbing"
            else:
                final_label = self._derive_horizontal(snout_kp, speed)

        # D) Rearing: confirmado por aspect ratio del bbox (altura/ancho).
        elif yolo_label == "rat_rearing":
            x1, y1, x2, y2 = rat_box
            h_box = y2 - y1
            w_box = x2 - x1
            aspect = h_box / w_box if w_box > 0 else 1.0
            if aspect >= REARING_ASPECT_RATIO:
                final_label = "rat_rearing"
            else:
                final_label = self._derive_horizontal(snout_kp, speed)

        # Estabilizacion temporal.
        final_label = self._stabilizer.update(final_label)

        # Sub-estado de sniffing: distingue si el raton se mueve o esta parado.
        # Se aplica DESPUES del estabilizador para que el movimiento se muestre
        # en tiempo real sin retardo.
        if final_label == "sniffing":
            final_label = ("sniffing_walking" if speed >= WALK_SPEED_THRESHOLD
                           else "sniffing_immobile")

        return final_label
