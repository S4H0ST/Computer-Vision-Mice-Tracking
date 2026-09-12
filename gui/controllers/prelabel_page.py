# -*- coding: utf-8 -*-
"""
Pagina de pre-etiquetado (indice 4 del stackedWidget principal).

Flujo multi-paso:
    Sub-pagina 0  — Seleccion de video + calibracion opcional de arena.
    Sub-pagina 1  — Pre-procesado YOLO (barra de progreso).
    Sub-pagina 2  — Interfaz de etiquetado con reproduccion y generacion de dataset.
"""

import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox, QProgressBar,
    QComboBox, QDialog, QDialogButtonBox, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QProgressDialog, QFormLayout,
    QGroupBox, QSizePolicy, QFrame, QAbstractItemView,
    QInputDialog, QSpinBox,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from config.config import paths, detect_cfg

DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR   = PROJECT_ROOT / "models"
HOLE_RADIUS  = 15

# Nombres de visualizacion de etiquetas por idioma
_KEY_DISPLAY = {
    "climbing":     {"es": "1: Escalando",      "en": "1: Climbing"},
    "grooming":     {"es": "2: Acicalamiento",  "en": "2: Grooming"},
    "head_dipping": {"es": "3: Agujero",        "en": "3: Head-dip"},
    "horizontal":   {"es": "4: Horizontal",     "en": "4: Horizontal"},
    "rearing":      {"es": "5: Erguido",        "en": "5: Rearing"},
    "sniffing":     {"es": "6: Olfateando",     "en": "6: Sniffing"},
    "immobile":     {"es": "7: Inmovil",        "en": "7: Immobile"},
}

# (Qt.Key, internal_name, display_name_es, hex_color, bgr_color_tuple)
PRELABEL_KEYS = [
    (Qt.Key_1, "climbing",     "1: Escalando",      "#ff00ff", (255,   0, 255)),
    (Qt.Key_2, "grooming",     "2: Acicalamiento",  "#b4ffb4", (180, 255, 180)),
    (Qt.Key_3, "head_dipping", "3: Agujero",        "#ffa500", (0,   165, 255)),
    (Qt.Key_4, "horizontal",   "4: Horizontal",     "#ffff00", (0,   255, 255)),
    (Qt.Key_5, "rearing",      "5: Erguido",        "#00ff00", (0,   255,   0)),
    (Qt.Key_6, "sniffing",     "6: Olfateando",     "#00c8ff", (255, 200,   0)),
    (Qt.Key_7, "immobile",     "7: Inmovil",        "#b4b4b4", (180, 180, 180)),
]
CLASS_NAMES = [k[1] for k in PRELABEL_KEYS]

# Textos traducibles
_PL_T = {
    "step0_title":     {"es": "Pre-Etiquetado — Paso 1: Seleccion de Video",
                        "en": "Pre-Labeling — Step 1: Video Selection"},
    "step1_title":     {"es": "Pre-Etiquetado — Paso 2: Procesando Video",
                        "en": "Pre-Labeling — Step 2: Processing Video"},
    "no_video":        {"es": "Selecciona un video para mostrar el primer frame",
                        "en": "Select a video to display the first frame"},
    "browse_video":    {"es": "Seleccionar Video...",    "en": "Browse Video..."},
    "skip_calib":      {"es": "Omitir calibracion (no se dibujaran zonas en el video)",
                        "en": "Skip calibration (zones will not be shown in the video)"},
    "next_btn":        {"es": "Siguiente →",        "en": "Next →"},
    "cancel_btn":      {"es": "Cancelar",                "en": "Cancel"},
    "hole_size_grp":   {"es": "Tamano agujero",          "en": "Hole size"},
    "radius_lbl":      {"es": "Radio (px):",             "en": "Radius (px):"},
    "grp_keys":        {"es": "Teclas de etiquetado",    "en": "Label Keys"},
    "grp_stats":       {"es": "Estadisticas",            "en": "Statistics"},
    "finish_btn":      {"es": "Finalizar y Generar Dataset", "en": "Finish and Generate Dataset"},
    "edit_labels_btn": {"es": "Editar Etiquetas...",     "en": "Edit Labels..."},
    "no_label":        {"es": "Sin etiqueta",            "en": "No label"},
    "vel_lbl":         {"es": "Vel:",                    "en": "Speed:"},
    "tt_rev":          {"es": "Reproducir hacia atras",  "en": "Play backward"},
    "tt_prev":         {"es": "Frame anterior",          "en": "Previous frame"},
    "tt_pause":        {"es": "Pausar / Reanudar",       "en": "Pause / Resume"},
    "tt_next":         {"es": "Frame siguiente",         "en": "Next frame"},
    "tt_fwd":          {"es": "Reproducir hacia adelante", "en": "Play forward"},
    "tt_occlude":      {"es": "Marcar hocico oculto (O)", "en": "Mark snout occluded (O)"},
    "calib_step1":     {"es": "Paso 1/4 — Clic en BORDE EXTERIOR: 2 esquinas opuestas ({n}/2)",
                        "en": "Step 1/4 — Click EXTERIOR border: 2 opposite corners ({n}/2)"},
    "calib_step2":     {"es": "Paso 2/4 — Clic en BORDE INTERIOR: 2 esquinas opuestas ({n}/2)",
                        "en": "Step 2/4 — Click INTERIOR border: 2 opposite corners ({n}/2)"},
    "calib_step3":     {"es": "Paso 3/4 — Clic en los 4 AGUJEROS ({n}/4)  |  Arrastrar borde del circulo para ajustar radio",
                        "en": "Step 3/4 — Click 4 HOLE centers ({n}/4)  |  Drag circle edge to resize"},
    "calib_step4":     {"es": "Paso 4/4 — BORDE CENTRAL (OPCIONAL): 2 esquinas ({n}/2)",
                        "en": "Step 4/4 — CENTRAL BORDER (OPTIONAL): 2 corners ({n}/2)"},
    "calib_done":      {"es": "Calibracion completa.",
                        "en": "Calibration complete."},
    "calib_done_c":    {"es": "Calibracion completa (con borde central).",
                        "en": "Calibration complete (with central border)."},
    "occlude_key":        {"es": "O",  "en": "O"},
    "bouts_unit":         {"es": "bouts", "en": "bouts"},
    "import_coords_grp":  {"es": "Cargar coordenadas",   "en": "Import coordinates"},
    "import_btn":         {"es": "Cargar JSON...",        "en": "Load JSON..."},
    "output_folder_grp":  {"es": "Carpeta dataset",      "en": "Dataset folder"},
    "output_default":     {"es": "Predeterminada: datasets/", "en": "Default: datasets/"},
    "browse_btn":         {"es": "Cambiar...",            "en": "Browse..."},
}


def _fmt_time(s: float) -> str:
    """Convierte segundos a formato M:SS."""
    m = int(s) // 60
    sec = int(s) % 60
    return f"{m}:{sec:02d}"


# ---------------------------------------------------------------------------
# TimelineWidget
# ---------------------------------------------------------------------------

class TimelineWidget(QWidget):
    """Barra de tiempo coloreada con etiquetas, posicion actual y marcas de tiempo."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(28)
        self.setMaximumHeight(28)
        self._total_frames: int = 1
        self._current_frame: int = 0
        self._label_map: dict = {}
        self._label_colors: dict = {}
        self._fps: float = 25.0

    def update_state(
        self,
        total_frames: int,
        current_frame: int,
        label_map: dict,
        label_colors_hex: dict,
        fps: float = 25.0,
    ) -> None:
        self._total_frames   = max(total_frames, 1)
        self._current_frame  = current_frame
        self._label_map      = label_map
        self._label_colors   = label_colors_hex
        self._fps            = max(fps, 1.0)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        w, h = self.width(), self.height()
        TICK_H = 8   # altura de las marcas de tiempo en la parte superior

        # Fondo
        painter.fillRect(0, TICK_H, w, h - TICK_H, QColor("#444"))

        # Columnas coloreadas por etiqueta
        for fidx, name in self._label_map.items():
            hex_c = self._label_colors.get(name, "#888")
            x = int(fidx / self._total_frames * w)
            painter.setPen(QPen(QColor(hex_c), 1))
            painter.drawLine(x, TICK_H, x, h)

        # Marcas de tiempo cada 30 s
        total_s = self._total_frames / self._fps
        interval = 30.0 if total_s <= 300 else 60.0
        t = interval
        painter.setPen(QPen(QColor("#aaa"), 1))
        font = painter.font()
        font.setPointSize(6)
        painter.setFont(font)
        while t < total_s:
            x = int(t / total_s * w)
            painter.drawLine(x, 0, x, TICK_H)
            lbl = _fmt_time(t)
            painter.drawText(x + 1, TICK_H - 1, lbl)
            t += interval

        # Linea de posicion actual
        cx = int(self._current_frame / self._total_frames * w)
        painter.setPen(QPen(QColor("black"), 2))
        painter.drawLine(cx, 0, cx, h)

        # Borde
        painter.setPen(QPen(QColor("#222"), 1))
        painter.drawRect(0, TICK_H, w - 1, h - TICK_H - 1)


# ---------------------------------------------------------------------------
# PreprocessWorker
# ---------------------------------------------------------------------------

class PreprocessWorker(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, video_path: str | Path, parent=None) -> None:
        super().__init__(parent)
        self._video_path = str(video_path)

    def run(self) -> None:
        try:
            self._process()
        except Exception as exc:
            self.error.emit(str(exc))

    def _process(self) -> None:
        from ultralytics import YOLO

        cap = cv2.VideoCapture(self._video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        model  = YOLO(str(paths.yolo_model))
        device = detect_cfg.device

        results_iter = model.predict(
            source=self._video_path,
            stream=True,
            conf=0.18,
            device=device,
            iou=0.5,
            verbose=False,
        )

        detections: dict = {}
        frame_idx = 0

        for res in results_iter:
            box_out = kps_xy_out = kps_conf_out = None
            best_out = 0

            if res.boxes and len(res.boxes) > 0:
                confs    = res.boxes.conf.cpu().numpy()
                best_out = int(np.argmax(confs))
                box_out  = res.boxes.xyxy[best_out].cpu().numpy()
                if res.keypoints is not None:
                    kps_xy_out   = res.keypoints.xy.cpu()
                    kps_conf_out = res.keypoints.conf.cpu() if res.keypoints.conf is not None else None

            detections[frame_idx] = (box_out, kps_xy_out, kps_conf_out, best_out)
            frame_idx += 1
            self.progress.emit(frame_idx, total if total > 0 else frame_idx)

        self.finished.emit(detections)


# ---------------------------------------------------------------------------
# EditLabelsDialog
# ---------------------------------------------------------------------------

class EditLabelsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Editar Etiquetas / Clases")
        self.setMinimumSize(360, 400)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self._original_classes: list[str] = []
        self._load_classes()
        self._build_ui()

    def _load_classes(self) -> None:
        yaml_path = DATASETS_DIR / "data.yaml"
        if yaml_path.exists():
            try:
                import yaml as _yaml
                with open(yaml_path, "r", encoding="utf-8") as f:
                    data = _yaml.safe_load(f)
                self._original_classes = list(data.get("names", CLASS_NAMES))
                return
            except Exception:
                pass
        self._original_classes = list(CLASS_NAMES)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        lbl = QLabel("Clases del dataset (una por linea):")
        lbl.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl)
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        for name in self._original_classes:
            self._list.addItem(QListWidgetItem(name))
        layout.addWidget(self._list)
        btn_row = QHBoxLayout()
        for text, slot in [("Agregar", self._on_add), ("Renombrar", self._on_rename), ("Eliminar", self._on_remove)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)
        box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        box.accepted.connect(self._on_save)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

    def _on_add(self) -> None:
        name, ok = QInputDialog.getText(self, "Agregar clase", "Nombre de la nueva clase:")
        if ok and name.strip():
            self._list.addItem(QListWidgetItem(name.strip()))

    def _on_rename(self) -> None:
        items = self._list.selectedItems()
        if not items:
            return
        new, ok = QInputDialog.getText(self, "Renombrar clase", "Nuevo nombre:", text=items[0].text())
        if ok and new.strip():
            items[0].setText(new.strip())

    def _on_remove(self) -> None:
        row = self._list.currentRow()
        if row >= 0:
            self._list.takeItem(row)

    def _on_save(self) -> None:
        new_classes = [self._list.item(i).text() for i in range(self._list.count())]
        changed = set(new_classes) != set(self._original_classes)
        try:
            DATASETS_DIR.mkdir(parents=True, exist_ok=True)
            _write_data_yaml(new_classes, DATASETS_DIR / "data.yaml")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"No se pudo guardar data.yaml:\n{exc}")
            return
        if changed:
            QMessageBox.warning(self, "Clases modificadas",
                                "Al anadir o eliminar clases se creara un modelo nuevo.")
        else:
            QMessageBox.information(self, "Sin cambios", "Las clases no han cambiado.")
        self.accept()


# ---------------------------------------------------------------------------
# Helpers de escritura
# ---------------------------------------------------------------------------

def _write_data_yaml(class_names: list[str], yaml_path: Path) -> None:
    lines = [
        f"path: {str(DATASETS_DIR)}",
        "train: images/train",
        "val: images/valid",
        "test: images/test",
        f"nc: {len(class_names)}",
        f"names: {class_names!r}",
        "kpt_shape: [3, 3]",
        "",
    ]
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# PrelabelPage
# ---------------------------------------------------------------------------

_HANDLE_HIT_PX = 18   # radio de acierto para handles de agujero (en coords originales)
_HANDLE_SIZE   = 5    # mitad del lado del cuadrado de handle (en coords originales)


class PrelabelPage(QWidget):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self._lang: str = "es"

        # Calibracion
        self._pl_calib_exterior: list[tuple] = []
        self._pl_calib_interior: list[tuple] = []
        self._pl_calib_holes:    list[tuple] = []
        self._pl_calib_center:   list[tuple] = []
        self._pl_calib_frame:    np.ndarray | None = None
        self._pl_calib_scale_x:  float = 1.0
        self._pl_calib_scale_y:  float = 1.0
        self._pl_calib_offset_x: int = 0
        self._pl_calib_offset_y: int = 0
        self._pl_hole_radius:    int = HOLE_RADIUS

        # Estado de arrastre para redimensionar agujeros
        self._pl_drag_hole_idx:  int = -1
        self._pl_drag_handle:    int = -1   # 0=NW,1=NE,2=SW,3=SE

        # Re-edicion de zona individual
        self._pl_calib_edit_zone: int | None = None
        self._pl_calib_zone_btns: list = []

        # Carpeta de salida del dataset (override de DATASETS_DIR)
        self._output_dir_override: Path | None = None

        # Video
        self._video_path: Path | None = None

        # Detecciones
        self._detections: dict = {}

        # Reproduccion
        self._cap: cv2.VideoCapture | None = None
        self._frame_idx: int = 0
        self._total_frames: int = 0
        self._fps: float = 25.0
        self._playing: bool = False
        self._direction: int = 1
        self._speed: float = 1.0
        self._label_map: dict[int, str] = {}
        self._current_label: str | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)

        # Keypoints ocultos: frame_idx -> coords (x,y) o None para usar deteccion
        self._occluded_snout: set[int] = set()

        # Widget refs para traduccion
        self._stat_labels:   dict[str, QLabel] = {}
        self._label_buttons: dict[str, QPushButton] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # Eventos
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._inner_stack.currentIndex() == 0 and self._pl_calib_frame is not None:
            self._pl_display_calib_frame()

    # ------------------------------------------------------------------
    # Construccion UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._inner_stack = QStackedWidget()
        root.addWidget(self._inner_stack)
        self._inner_stack.addWidget(self._build_step0())
        self._inner_stack.addWidget(self._build_step1())
        self._inner_stack.addWidget(self._build_step2())

    # ---- Sub-pagina 0 ------------------------------------------------

    def _build_step0(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        self._lbl_step0_title = QLabel(_PL_T["step0_title"][self._lang])
        self._lbl_step0_title.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #CB0017;")
        layout.addWidget(self._lbl_step0_title)

        # Seleccion de video
        vid_row = QHBoxLayout()
        self._lbl_video_path = QLabel("Ningun video seleccionado")
        self._lbl_video_path.setStyleSheet("color: #666; font-size: 11px;")
        self._lbl_video_path.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        vid_row.addWidget(self._lbl_video_path)
        self._btn_browse_vid = QPushButton(_PL_T["browse_video"][self._lang])
        self._btn_browse_vid.clicked.connect(self._on_browse_video)
        vid_row.addWidget(self._btn_browse_vid)
        layout.addLayout(vid_row)

        # Frame de calibracion + panel derecho
        content_row = QHBoxLayout()
        content_row.setSpacing(10)

        self._lbl_pl_frame = QLabel()
        self._lbl_pl_frame.setMinimumHeight(320)
        self._lbl_pl_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._lbl_pl_frame.setAlignment(Qt.AlignCenter)
        self._lbl_pl_frame.setText(_PL_T["no_video"][self._lang])
        self._lbl_pl_frame.setStyleSheet("background-color: #111; color: #888; font-size: 13px;")
        self._lbl_pl_frame.mousePressEvent   = self._on_pl_calib_mouse_press
        self._lbl_pl_frame.mouseMoveEvent    = self._on_pl_calib_mouse_move
        self._lbl_pl_frame.mouseReleaseEvent = self._on_pl_calib_mouse_release
        self._lbl_pl_frame.setMouseTracking(True)
        content_row.addWidget(self._lbl_pl_frame, stretch=1)

        # Panel derecho
        self._right_calib_panel = QWidget()
        self._right_calib_panel.setFixedWidth(150)
        right_v = QVBoxLayout(self._right_calib_panel)
        right_v.setContentsMargins(4, 0, 0, 0)
        right_v.setSpacing(8)

        self._grp_hole_size = QGroupBox(_PL_T["hole_size_grp"][self._lang])
        grp_r = QVBoxLayout(self._grp_hole_size)
        grp_r.setSpacing(4)
        self._lbl_hole_radius = QLabel(_PL_T["radius_lbl"][self._lang])
        self._lbl_hole_radius.setStyleSheet("font-size: 11px;")
        grp_r.addWidget(self._lbl_hole_radius)
        self._spin_hole_radius = QSpinBox()
        self._spin_hole_radius.setRange(5, 200)
        self._spin_hole_radius.setValue(HOLE_RADIUS)
        self._spin_hole_radius.setSuffix(" px")
        self._spin_hole_radius.valueChanged.connect(self._on_hole_radius_changed)
        grp_r.addWidget(self._spin_hole_radius)
        right_v.addWidget(self._grp_hole_size)

        # Importar coordenadas
        self._grp_pl_import = QGroupBox(_PL_T["import_coords_grp"][self._lang])
        grp_imp_layout = QVBoxLayout(self._grp_pl_import)
        grp_imp_layout.setSpacing(4)
        self._btn_pl_import_coords = QPushButton(_PL_T["import_btn"][self._lang])
        self._btn_pl_import_coords.clicked.connect(self._on_pl_import_coords)
        grp_imp_layout.addWidget(self._btn_pl_import_coords)
        self._lbl_pl_import_status = QLabel("")
        self._lbl_pl_import_status.setStyleSheet("color: #95a5a6; font-size: 10px;")
        self._lbl_pl_import_status.setWordWrap(True)
        grp_imp_layout.addWidget(self._lbl_pl_import_status)
        right_v.addWidget(self._grp_pl_import)

        # Carpeta de salida del dataset
        self._grp_pl_output = QGroupBox(_PL_T["output_folder_grp"][self._lang])
        grp_out_layout = QVBoxLayout(self._grp_pl_output)
        grp_out_layout.setSpacing(4)
        self._edit_pl_output = QLabel(_PL_T["output_default"][self._lang])
        self._edit_pl_output.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        self._edit_pl_output.setWordWrap(True)
        grp_out_layout.addWidget(self._edit_pl_output)
        self._btn_pl_select_output = QPushButton(_PL_T["browse_btn"][self._lang])
        self._btn_pl_select_output.clicked.connect(self._on_pl_select_output)
        grp_out_layout.addWidget(self._btn_pl_select_output)
        right_v.addWidget(self._grp_pl_output)

        right_v.addStretch()
        content_row.addWidget(self._right_calib_panel)

        layout.addLayout(content_row, stretch=1)

        # Botones de re-edicion de zona
        _ZONE_DEFS_PL = [
            ("Borde Exterior", "Exterior Border", "#e74c3c"),
            ("Borde Interior", "Interior Border", "#3498db"),
            ("Agujeros",       "Holes",           "#27ae60"),
            ("Borde Central",  "Central Border",  "#d4ac0d"),
        ]
        zone_row_pl = QHBoxLayout()
        zone_row_pl.setSpacing(6)
        self._pl_calib_zone_btns = []
        for i, (es_txt, _en_txt, color) in enumerate(_ZONE_DEFS_PL):
            btn = QPushButton(es_txt)
            btn.setCheckable(True)
            btn.setStyleSheet(
                f"QPushButton {{ border: 1.5px solid {color}; color: {color}; background: transparent; "
                f"border-radius: 3px; padding: 3px 8px; font-size: 11px; }}"
                f"QPushButton:checked {{ background-color: {color}; color: white; }}"
                f"QPushButton:hover:!checked {{ background-color: rgba(0,0,0,0.04); }}"
            )
            btn.clicked.connect(lambda checked, z=i: self._on_pl_zone_btn(z, checked))
            self._pl_calib_zone_btns.append(btn)
            zone_row_pl.addWidget(btn)
        zone_row_pl.addStretch()
        layout.addLayout(zone_row_pl)

        self._lbl_pl_calib_instr = QLabel("")
        self._lbl_pl_calib_instr.setStyleSheet("color: #1a6a2a; font-size: 11px;")
        layout.addWidget(self._lbl_pl_calib_instr)

        self._chk_skip_calib = QCheckBox(_PL_T["skip_calib"][self._lang])
        self._chk_skip_calib.stateChanged.connect(self._update_step0_next)
        layout.addWidget(self._chk_skip_calib)

        self._btn_step0_next = QPushButton(_PL_T["next_btn"][self._lang])
        self._btn_step0_next.setEnabled(False)
        self._btn_step0_next.setFixedHeight(36)
        self._btn_step0_next.setStyleSheet(
            "QPushButton { background-color: #CB0017; color: white; font-weight: bold; "
            "font-size: 13px; border-radius: 5px; } "
            "QPushButton:disabled { background-color: #aaa; } "
            "QPushButton:hover:!disabled { background-color: #a80013; }"
        )
        self._btn_step0_next.clicked.connect(self._on_step0_next)
        layout.addWidget(self._btn_step0_next)

        return w

    # ---- Sub-pagina 1 ------------------------------------------------

    def _build_step1(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(40, 0, 40, 0)

        outer.addStretch(1)

        self._lbl_step1_title = QLabel(_PL_T["step1_title"][self._lang])
        self._lbl_step1_title.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #CB0017;")
        self._lbl_step1_title.setAlignment(Qt.AlignCenter)
        outer.addWidget(self._lbl_step1_title)

        outer.addSpacing(16)

        self._lbl_preprocess_status = QLabel("Iniciando YOLO...")
        self._lbl_preprocess_status.setStyleSheet("font-size: 13px; color: #333;")
        self._lbl_preprocess_status.setAlignment(Qt.AlignCenter)
        outer.addWidget(self._lbl_preprocess_status)

        outer.addSpacing(8)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(28)
        self._progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #ccc; border-radius: 4px; "
            "background: #f0f0f0; text-align: center; }"
            "QProgressBar::chunk { background-color: #CB0017; border-radius: 3px; }"
        )
        outer.addWidget(self._progress_bar)

        outer.addSpacing(16)

        self._btn_cancel_preprocess = QPushButton(_PL_T["cancel_btn"][self._lang])
        self._btn_cancel_preprocess.setFixedHeight(32)
        self._btn_cancel_preprocess.setMaximumWidth(160)
        self._btn_cancel_preprocess.setStyleSheet(
            "QPushButton { background-color: #888; color: white; font-weight: bold; "
            "border-radius: 4px; } QPushButton:hover { background-color: #666; }"
        )
        self._btn_cancel_preprocess.clicked.connect(self._on_cancel_preprocess)
        cancel_row = QHBoxLayout()
        cancel_row.addStretch()
        cancel_row.addWidget(self._btn_cancel_preprocess)
        cancel_row.addStretch()
        outer.addLayout(cancel_row)

        outer.addStretch(1)
        return w

    # ---- Sub-pagina 2 ------------------------------------------------

    def _build_step2(self) -> QWidget:
        w = QWidget()
        w.setFocusPolicy(Qt.StrongFocus)
        main_layout = QHBoxLayout(w)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # --- Columna izquierda ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self._lbl_pl_video = QLabel()
        self._lbl_pl_video.setStyleSheet("background-color: black;")
        self._lbl_pl_video.setAlignment(Qt.AlignCenter)
        self._lbl_pl_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self._lbl_pl_video, stretch=1)

        self._timeline = TimelineWidget()
        left_layout.addWidget(self._timeline)

        # Controles de reproduccion
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)

        def _ctrl_btn(text: str, tip: str = "") -> QPushButton:
            b = QPushButton(text)
            b.setFixedSize(52, 34)
            b.setToolTip(tip)
            b.setStyleSheet("font-size: 12px; font-weight: bold;")
            return b

        # Frame counter (izquierda)
        self._lbl_frame_counter = QLabel("0 / 0")
        self._lbl_frame_counter.setStyleSheet("font-size: 11px; color: #333; min-width: 120px;")
        ctrl.addWidget(self._lbl_frame_counter)

        ctrl.addStretch()

        # Botones (centro)
        self._btn_rev   = _ctrl_btn("<<",  _PL_T["tt_rev"][self._lang])
        self._btn_prev  = _ctrl_btn("< 1", _PL_T["tt_prev"][self._lang])
        self._btn_pause = _ctrl_btn("||",  _PL_T["tt_pause"][self._lang])
        self._btn_next  = _ctrl_btn("1 >", _PL_T["tt_next"][self._lang])
        self._btn_fwd   = _ctrl_btn(">>",  _PL_T["tt_fwd"][self._lang])

        self._btn_rev.clicked.connect(self._on_play_rev)
        self._btn_prev.clicked.connect(self._on_step_prev)
        self._btn_pause.clicked.connect(self._on_toggle_pause)
        self._btn_next.clicked.connect(self._on_step_next)
        self._btn_fwd.clicked.connect(self._on_play_fwd)

        for b in [self._btn_rev, self._btn_prev, self._btn_pause,
                  self._btn_next, self._btn_fwd]:
            ctrl.addWidget(b)

        ctrl.addStretch()

        # Velocidad (derecha)
        self._lbl_speed = QLabel(_PL_T["vel_lbl"][self._lang])
        self._lbl_speed.setStyleSheet("font-size: 11px;")
        ctrl.addWidget(self._lbl_speed)
        self._cmb_speed = QComboBox()
        for s in ["0.25x", "0.5x", "1x", "1.25x", "1.5x", "2x", "3x", "4x", "5x"]:
            self._cmb_speed.addItem(s)
        self._cmb_speed.setCurrentText("1x")
        self._cmb_speed.currentTextChanged.connect(self._on_speed_changed)
        ctrl.addWidget(self._cmb_speed)

        ctrl.addSpacing(12)

        self._btn_finish = QPushButton(_PL_T["finish_btn"][self._lang])
        self._btn_finish.setStyleSheet(
            "QPushButton { background-color: #CB0017; color: white; font-weight: bold; "
            "padding: 4px 12px; border-radius: 4px; } "
            "QPushButton:hover { background-color: #a80013; }"
        )
        self._btn_finish.clicked.connect(self._on_finish)
        ctrl.addWidget(self._btn_finish)

        left_layout.addLayout(ctrl)
        main_layout.addWidget(left, stretch=7)

        # --- Columna derecha ---
        right = QWidget()
        right.setMaximumWidth(260)
        right.setMinimumWidth(180)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(6)

        self._lbl_current_label = QLabel(_PL_T["no_label"][self._lang])
        self._lbl_current_label.setAlignment(Qt.AlignCenter)
        self._lbl_current_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; padding: 6px; "
            "background-color: #eee; border-radius: 4px;"
        )
        right_layout.addWidget(self._lbl_current_label)

        self._grp_keys = QGroupBox(_PL_T["grp_keys"][self._lang])
        keys_layout = QVBoxLayout(self._grp_keys)
        keys_layout.setSpacing(3)
        for qt_key, name, display, hex_c, bgr_c in PRELABEL_KEYS:
            btn = QPushButton(_KEY_DISPLAY[name][self._lang])
            btn.setCheckable(True)
            r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            fg = "#000" if lum > 128 else "#fff"
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {hex_c}; color: {fg}; "
                f"font-size: 11px; padding: 3px; border-radius: 3px; border: 2px solid transparent; }}"
                f"QPushButton:checked {{ border: 2px solid #000; }}"
            )
            btn.clicked.connect(lambda checked, n=name: self._select_label(n))
            self._label_buttons[name] = btn
            keys_layout.addWidget(btn)

        # Boton de ocluir hocico (estilo cambia segun estado del frame actual)
        self._btn_occlude = QPushButton("O: " + _PL_T["tt_occlude"][self._lang])
        self._btn_occlude.setToolTip(_PL_T["tt_occlude"][self._lang])
        self._btn_occlude.clicked.connect(self._on_toggle_occlude)
        keys_layout.addWidget(self._btn_occlude)
        self._refresh_occlude_btn(active=False)

        right_layout.addWidget(self._grp_keys)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #ccc;")
        right_layout.addWidget(sep)

        self._grp_stats = QGroupBox(_PL_T["grp_stats"][self._lang])
        stats_layout = QFormLayout(self._grp_stats)
        stats_layout.setSpacing(3)
        for _, name, display, _, _ in PRELABEL_KEYS:
            lbl_val = QLabel("0 bouts | 0.0 s")
            lbl_val.setStyleSheet("font-size: 10px;")
            stats_layout.addRow(_KEY_DISPLAY[name][self._lang] + ":", lbl_val)
            self._stat_labels[name] = lbl_val
        right_layout.addWidget(self._grp_stats)

        right_layout.addStretch()

        self._btn_edit_labels = QPushButton(_PL_T["edit_labels_btn"][self._lang])
        self._btn_edit_labels.clicked.connect(self._on_edit_labels)
        right_layout.addWidget(self._btn_edit_labels)

        main_layout.addWidget(right, stretch=3)
        return w

    # ------------------------------------------------------------------
    # Sub-pagina 0: logica
    # ------------------------------------------------------------------

    def _on_browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar video", "",
            "Videos (*.mp4 *.avi *.mov *.mkv);;Todos (*.*)"
        )
        if not path:
            return
        self._video_path = Path(path)
        self._lbl_video_path.setText(str(self._video_path))

        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self._pl_calib_frame    = frame
            self._pl_calib_exterior = []
            self._pl_calib_interior = []
            self._pl_calib_holes    = []
            self._pl_calib_center   = []
            QTimer.singleShot(120, self._pl_display_calib_frame)
            self._update_pl_calib_instruction()
        else:
            QMessageBox.warning(self, "Error", "No se pudo leer el primer frame del video.")

        self._update_step0_next()

    def _on_pl_zone_btn(self, zone: int, checked: bool) -> None:
        if checked:
            for i, btn in enumerate(self._pl_calib_zone_btns):
                if i != zone:
                    btn.setChecked(False)
            targets = [self._pl_calib_exterior, self._pl_calib_interior, self._pl_calib_holes, self._pl_calib_center]
            targets[zone].clear()
            self._pl_calib_edit_zone = zone
        else:
            self._pl_calib_edit_zone = None
        self._pl_display_calib_frame()
        self._update_pl_calib_instruction()
        self._update_step0_next()

    def _on_pl_import_coords(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Cargar coordenadas", "", "JSON Files (*.json);;Todos (*.*)"
        )
        if not path:
            return
        try:
            import json
            with open(path, "r") as f:
                data = json.load(f)
            ext    = data.get("exterior", [])
            intr   = data.get("interior", [])
            holes  = data.get("holes", [])
            center = data.get("center_zone", [])
            if len(ext) >= 2:
                self._pl_calib_exterior = [(int(p[0]), int(p[1])) for p in ext[:2]]
            if len(intr) >= 2:
                self._pl_calib_interior = [(int(p[0]), int(p[1])) for p in intr[:2]]
            if holes:
                self._pl_calib_holes = [(int(p[0]), int(p[1])) for p in holes[:4]]
            if len(center) >= 2:
                self._pl_calib_center = [(int(p[0]), int(p[1])) for p in center[:2]]
            if "hole_radius" in data:
                self._pl_hole_radius = int(data["hole_radius"])
                self._spin_hole_radius.blockSignals(True)
                self._spin_hole_radius.setValue(self._pl_hole_radius)
                self._spin_hole_radius.blockSignals(False)
            fname = Path(path).name
            self._lbl_pl_import_status.setText(f"✓ {fname}")
            self._lbl_pl_import_status.setStyleSheet("color: #27ae60; font-size: 10px;")
            self._pl_display_calib_frame()
            self._update_pl_calib_instruction()
            self._update_step0_next()
        except Exception as e:
            self._lbl_pl_import_status.setText(f"Error: {e}")
            self._lbl_pl_import_status.setStyleSheet("color: #e74c3c; font-size: 10px;")

    def _on_pl_select_output(self) -> None:
        default = str(DATASETS_DIR) if DATASETS_DIR.exists() else str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Carpeta de dataset", default)
        if folder:
            self._output_dir_override = Path(folder)
            self._edit_pl_output.setText(str(self._output_dir_override))
            self._edit_pl_output.setStyleSheet("color: #2c3e50; font-size: 10px;")

    def _pl_calib_phase(self) -> int:
        if len(self._pl_calib_exterior) < 2:
            return 0
        if len(self._pl_calib_interior) < 2:
            return 1
        if len(self._pl_calib_holes) < 4:
            return 2
        return 3

    def _pl_calib_done(self) -> bool:
        return (
            len(self._pl_calib_exterior) == 2
            and len(self._pl_calib_interior) == 2
            and len(self._pl_calib_holes) == 4
        )

    def _hole_corner_handles(self, hole_idx: int) -> list[tuple[float, float]]:
        """Devuelve las 4 esquinas (NW,NE,SW,SE) del cuadrado envolvente en coords originales."""
        hx, hy = self._pl_calib_holes[hole_idx]
        d = self._pl_hole_radius * 0.707  # diagonal
        return [
            (hx - d, hy - d),  # NW
            (hx + d, hy - d),  # NE
            (hx - d, hy + d),  # SW
            (hx + d, hy + d),  # SE
        ]

    def _orig_from_event(self, event) -> tuple[int, int]:
        """Convierte coordenadas del evento (display) a coordenadas de imagen original."""
        lx = event.x() - self._pl_calib_offset_x
        ly = event.y() - self._pl_calib_offset_y
        if lx < 0 or ly < 0:
            return -1, -1
        orig_x = int(lx * self._pl_calib_scale_x)
        orig_y = int(ly * self._pl_calib_scale_y)
        if self._pl_calib_frame is not None:
            oh, ow = self._pl_calib_frame.shape[:2]
            orig_x = max(0, min(orig_x, ow - 1))
            orig_y = max(0, min(orig_y, oh - 1))
        return orig_x, orig_y

    def _find_handle_hit(self, ox: int, oy: int) -> tuple[int, int]:
        """Devuelve (hole_idx, handle_idx) si el punto esta cerca de un handle, o (-1,-1)."""
        for hi, hole in enumerate(self._pl_calib_holes):
            for hndl_i, (hx, hy) in enumerate(self._hole_corner_handles(hi)):
                dist = math.hypot(ox - hx, oy - hy)
                if dist < _HANDLE_HIT_PX:
                    return hi, hndl_i
        return -1, -1

    def _on_pl_calib_mouse_press(self, event) -> None:
        if self._pl_calib_frame is None:
            return
        if self._pl_calib_edit_zone is None and self._pl_calib_done() and len(self._pl_calib_center) >= 2:
            return

        ox, oy = self._orig_from_event(event)
        if ox < 0:
            return

        # Comprobar si el clic esta sobre un handle (solo en fase 2+ con agujeros)
        if len(self._pl_calib_holes) > 0:
            hi, hndl_i = self._find_handle_hit(ox, oy)
            if hi >= 0:
                self._pl_drag_hole_idx = hi
                self._pl_drag_handle   = hndl_i
                return

        if self._pl_calib_edit_zone is not None:
            z = self._pl_calib_edit_zone
            targets  = [self._pl_calib_exterior, self._pl_calib_interior, self._pl_calib_holes, self._pl_calib_center]
            capacity = [2, 2, 4, 2]
            if len(targets[z]) < capacity[z]:
                targets[z].append((ox, oy))
            if len(targets[z]) >= capacity[z]:
                self._pl_calib_zone_btns[z].setChecked(False)
                self._pl_calib_edit_zone = None
        else:
            # Clic normal segun fase
            phase = self._pl_calib_phase()
            if phase == 0:
                self._pl_calib_exterior.append((ox, oy))
            elif phase == 1:
                self._pl_calib_interior.append((ox, oy))
            elif phase == 2:
                self._pl_calib_holes.append((ox, oy))
            elif phase == 3 and len(self._pl_calib_center) < 2:
                self._pl_calib_center.append((ox, oy))

        self._pl_display_calib_frame()
        self._update_pl_calib_instruction()
        self._update_step0_next()

    def _on_pl_calib_mouse_move(self, event) -> None:
        if self._pl_drag_hole_idx < 0:
            return
        ox, oy = self._orig_from_event(event)
        if ox < 0:
            return
        hx, hy = self._pl_calib_holes[self._pl_drag_hole_idx]
        new_r = max(5, int(math.hypot(ox - hx, oy - hy)))
        self._pl_hole_radius = new_r
        self._spin_hole_radius.blockSignals(True)
        self._spin_hole_radius.setValue(new_r)
        self._spin_hole_radius.blockSignals(False)
        self._pl_display_calib_frame()

    def _on_pl_calib_mouse_release(self, event) -> None:
        self._pl_drag_hole_idx = -1
        self._pl_drag_handle   = -1

    def _on_hole_radius_changed(self, value: int) -> None:
        self._pl_hole_radius = value
        self._pl_display_calib_frame()

    def _pl_display_calib_frame(self) -> None:
        if self._pl_calib_frame is None:
            return

        label = self._lbl_pl_frame
        lw, lh = label.width(), label.height()
        if lw < 10 or lh < 10:
            QTimer.singleShot(80, self._pl_display_calib_frame)
            return

        frame = self._pl_calib_frame.copy()

        # Exterior (rojo)
        for pt in self._pl_calib_exterior:
            cv2.circle(frame, pt, 8, (0, 0, 255), -1)
        if len(self._pl_calib_exterior) == 2:
            xs = [p[0] for p in self._pl_calib_exterior]
            ys = [p[1] for p in self._pl_calib_exterior]
            cv2.rectangle(frame, (min(xs), min(ys)), (max(xs), max(ys)), (0, 0, 255), 2)

        # Interior (azul)
        for pt in self._pl_calib_interior:
            cv2.circle(frame, pt, 8, (255, 0, 0), -1)
        if len(self._pl_calib_interior) == 2:
            xs = [p[0] for p in self._pl_calib_interior]
            ys = [p[1] for p in self._pl_calib_interior]
            cv2.rectangle(frame, (min(xs), min(ys)), (max(xs), max(ys)), (255, 0, 0), 2)

        # Agujeros con radio actual + handles de esquina
        r = self._pl_hole_radius
        for hi, pt in enumerate(self._pl_calib_holes):
            cv2.circle(frame, pt, r, (0, 255, 0), 2)
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)
            # Handles de esquina (cuadrados pequeños en NW/NE/SW/SE)
            for hx, hy in self._hole_corner_handles(hi):
                hxi, hyi = int(hx), int(hy)
                cv2.rectangle(frame,
                               (hxi - _HANDLE_SIZE, hyi - _HANDLE_SIZE),
                               (hxi + _HANDLE_SIZE, hyi + _HANDLE_SIZE),
                               (0, 200, 255), -1)
                cv2.rectangle(frame,
                               (hxi - _HANDLE_SIZE, hyi - _HANDLE_SIZE),
                               (hxi + _HANDLE_SIZE, hyi + _HANDLE_SIZE),
                               (0, 0, 0), 1)

        # Centro (amarillo opcional)
        for pt in self._pl_calib_center:
            cv2.circle(frame, pt, 8, (0, 220, 220), -1)
        if len(self._pl_calib_center) == 2:
            xs = [p[0] for p in self._pl_calib_center]
            ys = [p[1] for p in self._pl_calib_center]
            cv2.rectangle(frame, (min(xs), min(ys)), (max(xs), max(ys)), (0, 220, 220), 2)

        oh, ow = frame.shape[:2]
        scale = min(lw / ow, lh / oh)
        dw = int(ow * scale)
        dh = int(oh * scale)
        self._pl_calib_scale_x  = ow / dw
        self._pl_calib_scale_y  = oh / dh
        self._pl_calib_offset_x = (lw - dw) // 2
        self._pl_calib_offset_y = (lh - dh) // 2

        resized = cv2.resize(frame, (dw, dh))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        qimg    = QImage(rgb.data, dw, dh, dw * 3, QImage.Format_RGB888)
        pixmap  = QPixmap.fromImage(qimg)

        canvas = QPixmap(lw, lh)
        canvas.fill(Qt.black)
        painter = QPainter(canvas)
        painter.drawPixmap(self._pl_calib_offset_x, self._pl_calib_offset_y, pixmap)
        painter.end()
        label.setPixmap(canvas)

    def _update_pl_calib_instruction(self) -> None:
        if self._pl_calib_frame is None:
            self._lbl_pl_calib_instr.setText("")
            return
        t = _PL_T
        L = self._lang
        phase = self._pl_calib_phase()
        if len(self._pl_calib_center) == 2:
            txt = t["calib_done_c"][L]
        elif self._pl_calib_done():
            txt = t["calib_step4"][L].format(n=len(self._pl_calib_center))
        elif phase == 0:
            txt = t["calib_step1"][L].format(n=len(self._pl_calib_exterior))
        elif phase == 1:
            txt = t["calib_step2"][L].format(n=len(self._pl_calib_interior))
        else:
            txt = t["calib_step3"][L].format(n=len(self._pl_calib_holes))
        self._lbl_pl_calib_instr.setText(txt)

    def _update_step0_next(self) -> None:
        video_ok = self._video_path is not None
        calib_ok = self._pl_calib_done() or self._chk_skip_calib.isChecked()
        self._btn_step0_next.setEnabled(video_ok and calib_ok)

    def _on_step0_next(self) -> None:
        if self._video_path is None:
            return
        self._inner_stack.setCurrentIndex(1)
        self._start_preprocess()

    # ------------------------------------------------------------------
    # Sub-pagina 1: pre-procesado
    # ------------------------------------------------------------------

    def _start_preprocess(self) -> None:
        self._progress_bar.setValue(0)
        self._lbl_preprocess_status.setText("Iniciando YOLO...")
        self._preproc_worker = PreprocessWorker(self._video_path, parent=self)
        self._preproc_worker.progress.connect(self._on_preprocess_progress)
        self._preproc_worker.finished.connect(self._on_preprocess_finished)
        self._preproc_worker.error.connect(self._on_preprocess_error)
        self._preproc_worker.start()

    def _on_preprocess_progress(self, current: int, total: int) -> None:
        if total > 0:
            self._progress_bar.setValue(int(current / total * 100))
            self._lbl_preprocess_status.setText(f"Procesando frame {current} / {total} con YOLO...")
        else:
            self._lbl_preprocess_status.setText(f"Procesando frame {current}...")

    def _on_preprocess_finished(self, detections: dict) -> None:
        self._detections = detections
        self._progress_bar.setValue(100)
        self._lbl_preprocess_status.setText("Procesado completo. Cargando interfaz de etiquetado...")
        self._setup_labeling()
        self._inner_stack.setCurrentIndex(2)
        self.setFocus()

    def _on_cancel_preprocess(self) -> None:
        if hasattr(self, "_preproc_worker") and self._preproc_worker.isRunning():
            self._preproc_worker.terminate()
            self._preproc_worker.wait()
        self._inner_stack.setCurrentIndex(0)

    def _on_preprocess_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Error de pre-procesado", msg)
        self._inner_stack.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Sub-pagina 2: etiquetado — configuracion inicial
    # ------------------------------------------------------------------

    def _setup_labeling(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(str(self._video_path))
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._frame_idx = 0
        self._playing = False
        self._direction = 1
        self._speed = 1.0
        self._label_map = {}
        self._current_label = None
        self._occluded_snout = set()

        for btn in self._label_buttons.values():
            btn.setChecked(False)
        self._lbl_current_label.setText(_PL_T["no_label"][self._lang])
        self._lbl_current_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; padding: 6px; "
            "background-color: #eee; border-radius: 4px;"
        )

        self._show_frame(0)
        self._update_timeline()
        self._update_stats()
        self._update_frame_counter()

    # ------------------------------------------------------------------
    # Reproduccion
    # ------------------------------------------------------------------

    def _timer_interval_ms(self) -> int:
        return max(1, int(1000 / self._fps / self._speed))

    def _on_play_fwd(self) -> None:
        self._direction = 1
        self._playing = True
        self._timer.start(self._timer_interval_ms())

    def _on_play_rev(self) -> None:
        self._direction = -1
        self._playing = True
        self._timer.start(self._timer_interval_ms())

    def _on_toggle_pause(self) -> None:
        if self._playing:
            self._playing = False
            self._timer.stop()
            self._btn_pause.setText("> ")
        else:
            self._playing = True
            self._btn_pause.setText("||")
            self._timer.start(self._timer_interval_ms())

    def _on_step_prev(self) -> None:
        self._playing = False
        self._timer.stop()
        self._frame_idx = max(0, self._frame_idx - 1)
        self._show_frame(self._frame_idx)
        self._update_timeline()
        self._update_frame_counter()

    def _on_step_next(self) -> None:
        self._playing = False
        self._timer.stop()
        self._frame_idx = min(self._total_frames - 1, self._frame_idx + 1)
        self._show_frame(self._frame_idx)
        self._update_timeline()
        self._update_frame_counter()

    def _on_speed_changed(self, text: str) -> None:
        try:
            self._speed = float(text.replace("x", ""))
        except ValueError:
            self._speed = 1.0
        if self._playing:
            self._timer.start(self._timer_interval_ms())

    def _on_timer_tick(self) -> None:
        if self._current_label is not None:
            self._label_map[self._frame_idx] = self._current_label
        new_idx = self._frame_idx + self._direction
        if new_idx < 0 or new_idx >= self._total_frames:
            self._playing = False
            self._timer.stop()
            self._btn_pause.setText("> ")
            return
        self._frame_idx = new_idx
        self._show_frame(self._frame_idx)
        self._update_timeline()
        self._update_stats()
        self._update_frame_counter()

    # ------------------------------------------------------------------
    # Frame display
    # ------------------------------------------------------------------

    def _show_frame(self, frame_idx: int) -> None:
        if self._cap is None:
            return
        try:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, raw = self._cap.read()
        except Exception:
            return
        if not ret:
            return

        # Recortar al borde exterior calibrado
        img = raw.copy()
        x_off, y_off = 0, 0
        if len(self._pl_calib_exterior) == 2:
            xs = [p[0] for p in self._pl_calib_exterior]
            ys = [p[1] for p in self._pl_calib_exterior]
            H, W = img.shape[:2]
            x1c = max(0, min(xs))
            y1c = max(0, min(ys))
            x2c = min(W, max(xs))
            y2c = min(H, max(ys))
            if x2c > x1c and y2c > y1c:
                img = img[y1c:y2c, x1c:x2c]
                x_off, y_off = x1c, y1c

        # Dibujar detecciones
        det = self._detections.get(frame_idx)
        if det is not None:
            box, kps_xy, kps_conf, best = det
            if box is not None:
                rx1, ry1, rx2, ry2 = (int(box[0]) - x_off, int(box[1]) - y_off,
                                       int(box[2]) - x_off, int(box[3]) - y_off)
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (80, 160, 80), 1)

                if kps_xy is not None and len(kps_xy) > best:
                    try:
                        kps = kps_xy[best].numpy() if hasattr(kps_xy[best], "numpy") else kps_xy[best]
                        conf = (kps_conf[best].numpy()
                                if (kps_conf is not None and len(kps_conf) > best) else None)
                        kp_colors = [(0, 0, 200), (0, 200, 0), (200, 60, 0)]
                        visible = []
                        for i, (kp, color) in enumerate(zip(kps, kp_colors)):
                            c = float(conf[i]) if conf is not None else 1.0
                            kpx = int(kp[0]) - x_off
                            kpy = int(kp[1]) - y_off
                            is_vis = c >= 0.3 and not (kp[0] < 1 and kp[1] < 1)
                            # Keypoint 0 = snout: puede estar oculto
                            if i == 0 and frame_idx in self._occluded_snout:
                                visible.append(True)
                                cv2.circle(img, (kpx, kpy), 9, (255, 255, 255), -1)
                                cv2.circle(img, (kpx, kpy), 9, (0, 80, 220), 2)
                                cv2.line(img, (kpx - 6, kpy - 6), (kpx + 6, kpy + 6),
                                         (0, 60, 200), 2, cv2.LINE_AA)
                                cv2.line(img, (kpx + 6, kpy - 6), (kpx - 6, kpy + 6),
                                         (0, 60, 200), 2, cv2.LINE_AA)
                            elif is_vis:
                                visible.append(True)
                                cv2.circle(img, (kpx, kpy), 3, color, -1)
                            else:
                                visible.append(False)
                        for a, b in [(0, 1), (1, 2)]:
                            if (a < len(kps) and b < len(kps)
                                    and len(visible) > max(a, b)
                                    and visible[a] and visible[b]):
                                kpax = int(kps[a][0]) - x_off
                                kpay = int(kps[a][1]) - y_off
                                kpbx = int(kps[b][0]) - x_off
                                kpby = int(kps[b][1]) - y_off
                                cv2.line(img, (kpax, kpay), (kpbx, kpby), (120, 120, 120), 1)
                    except Exception:
                        pass

        # Escalar y mostrar
        label = self._lbl_pl_video
        lw, lh = label.width(), label.height()
        if lw < 10 or lh < 10:
            return
        h, w = img.shape[:2]
        scale = min(lw / w, lh / h)
        dw, dh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (dw, dh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, dw, dh, dw * 3, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))

        self._refresh_occlude_btn(frame_idx in self._occluded_snout)

    # ------------------------------------------------------------------
    # Timeline, stats y contador
    # ------------------------------------------------------------------

    def _update_timeline(self) -> None:
        hex_colors = {name: hex_c for _, name, _, hex_c, _ in PRELABEL_KEYS}
        self._timeline.update_state(
            self._total_frames, self._frame_idx, self._label_map, hex_colors, self._fps
        )

    def _update_stats(self) -> None:
        fps = self._fps if self._fps > 0 else 25.0
        bouts_word = _PL_T["bouts_unit"][self._lang]
        for _, name, _, _, _ in PRELABEL_KEYS:
            frames_with_label = [f for f, n in self._label_map.items() if n == name]
            if not frames_with_label:
                self._stat_labels[name].setText(f"0 {bouts_word} | 0.0 s")
                continue
            total_sec = len(frames_with_label) / fps
            sorted_frames = sorted(frames_with_label)
            bouts = 1
            for i in range(1, len(sorted_frames)):
                if sorted_frames[i] != sorted_frames[i - 1] + 1:
                    bouts += 1
            self._stat_labels[name].setText(f"{bouts} {bouts_word} | {total_sec:.1f} s")

    def _update_frame_counter(self) -> None:
        fps = self._fps if self._fps > 0 else 25.0
        total = max(self._total_frames - 1, 0)
        t_cur = self._frame_idx / fps
        t_tot = total / fps
        self._lbl_frame_counter.setText(
            f"{self._frame_idx} / {total}   {_fmt_time(t_cur)} / {_fmt_time(t_tot)}"
        )

    # ------------------------------------------------------------------
    # Seleccion de etiqueta + oclusion
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        for qt_key, name, _, _, _ in PRELABEL_KEYS:
            if event.key() == qt_key:
                self._select_label(name)
                return
        if event.key() == Qt.Key_O:
            self._on_toggle_occlude()
            return
        super().keyPressEvent(event)

    def _select_label(self, name: str) -> None:
        self._current_label = name
        for n, btn in self._label_buttons.items():
            btn.setChecked(n == name)
        hex_c   = next((h for _, n, _, h, _ in PRELABEL_KEYS if n == name), "#eee")
        display = _KEY_DISPLAY[name][self._lang]
        r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        fg = "#000" if lum > 128 else "#fff"
        self._lbl_current_label.setText(display)
        self._lbl_current_label.setStyleSheet(
            f"font-size: 13px; font-weight: bold; padding: 6px; "
            f"background-color: {hex_c}; color: {fg}; border-radius: 4px;"
        )

    def _refresh_occlude_btn(self, active: bool) -> None:
        if active:
            self._btn_occlude.setStyleSheet(
                "font-size: 10px; padding: 2px; border-radius: 3px; font-weight: bold; "
                "background-color: #e74c3c; color: white; border: 2px solid #c0392b;"
            )
        else:
            self._btn_occlude.setStyleSheet(
                "font-size: 10px; padding: 2px; border-radius: 3px; "
                "background-color: transparent; color: #555; border: 1px solid #999;"
            )

    def _on_toggle_occlude(self) -> None:
        """Alterna el estado de hocico oculto para el frame actual."""
        if self._frame_idx in self._occluded_snout:
            self._occluded_snout.discard(self._frame_idx)
        else:
            self._occluded_snout.add(self._frame_idx)
        self._refresh_occlude_btn(self._frame_idx in self._occluded_snout)
        self._show_frame(self._frame_idx)

    def _on_edit_labels(self) -> None:
        dlg = EditLabelsDialog(self)
        dlg.exec_()

    # ------------------------------------------------------------------
    # Finalizar — generar dataset
    # ------------------------------------------------------------------

    def _on_finish(self) -> None:
        self._playing = False
        self._timer.stop()
        self._btn_pause.setText("> ")

        valid_frames = [
            f for f in self._label_map
            if f in self._detections
            and self._detections[f] is not None
            and self._detections[f][0] is not None
        ]

        if not valid_frames:
            QMessageBox.warning(self, "Sin datos",
                                "No hay frames con etiqueta y deteccion valida.")
            return

        dlg = QProgressDialog("Generando dataset...", None, 0, len(valid_frames), self)
        dlg.setWindowTitle("Generando dataset")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)

        try:
            output_path = self._generate_dataset(valid_frames, dlg)
        except Exception as exc:
            dlg.close()
            QMessageBox.critical(self, "Error generando dataset", str(exc))
            return

        dlg.close()
        QMessageBox.information(
            self, "Dataset generado",
            f"Dataset guardado en:\n{output_path}\n\n"
            f"Frames totales: {len(valid_frames)}\n"
            "Ahora puedes ir a la pestana Entrenar para crear el modelo.",
        )

    def _generate_dataset(self, valid_frames: list, progress_dlg: QProgressDialog) -> Path:
        random.shuffle(valid_frames)
        n = len(valid_frames)
        if n >= 3:
            n_train = max(1, int(n * 0.70))
            n_valid = max(1, int(n * 0.20))
            n_test  = max(1, n - n_train - n_valid)
        else:
            n_train, n_valid, n_test = n, 0, 0

        splits = {
            "train": valid_frames[:n_train],
            "valid": valid_frames[n_train:n_train + n_valid],
            "test":  valid_frames[n_train + n_valid:],
        }
        out_dir = self._output_dir_override or DATASETS_DIR
        for split in ["train", "valid", "test"]:
            (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        step = 0
        for split, frame_list in splits.items():
            for fidx in frame_list:
                self._save_frame_annotation(fidx, split, out_dir)
                step += 1
                progress_dlg.setValue(step)

        _write_data_yaml(CLASS_NAMES, out_dir / "data.yaml")
        return out_dir

    def _save_frame_annotation(self, frame_idx: int, split: str, out_dir: Path | None = None) -> None:
        if self._cap is None:
            return
        try:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()
        except Exception:
            return
        if not ret:
            return

        base = out_dir or DATASETS_DIR
        H, W = frame.shape[:2]
        img_name = f"frame_{frame_idx:06d}.jpg"
        img_path = base / "images" / split / img_name
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        det = self._detections.get(frame_idx)
        if det is None:
            return
        box, kps_xy, kps_conf, best = det
        if box is None:
            return

        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2) / W
        cy = ((y1 + y2) / 2) / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H

        label_name = self._label_map.get(frame_idx, CLASS_NAMES[0])
        try:
            class_id = CLASS_NAMES.index(label_name)
        except ValueError:
            class_id = 0

        kp_parts = []
        if kps_xy is not None and len(kps_xy) > best:
            try:
                kps = kps_xy[best].numpy() if hasattr(kps_xy[best], "numpy") else kps_xy[best]
                conf_arr = (kps_conf[best].numpy()
                            if (kps_conf is not None and len(kps_conf) > best) else None)
                for i in range(3):
                    if i < len(kps):
                        kpx = float(kps[i][0]) / W
                        kpy = float(kps[i][1]) / H
                        c = float(conf_arr[i]) if conf_arr is not None else 1.0
                        if i == 0 and frame_idx in self._occluded_snout:
                            vis = 1   # occluded
                        elif c >= 0.3 and not (kps[i][0] < 1 and kps[i][1] < 1):
                            vis = 2   # visible
                        else:
                            vis = 0   # not labeled
                        kp_parts.extend([f"{kpx:.6f}", f"{kpy:.6f}", str(vis)])
                    else:
                        kp_parts.extend(["0.000000", "0.000000", "0"])
            except Exception:
                kp_parts = ["0.000000", "0.000000", "0"] * 3
        else:
            kp_parts = ["0.000000", "0.000000", "0"] * 3

        line = f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(kp_parts)
        lbl_path = base / "labels" / split / f"frame_{frame_idx:06d}.txt"
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write(line + "\n")

    # ------------------------------------------------------------------
    # Traduccion
    # ------------------------------------------------------------------

    def apply_language(self, lang: str) -> None:
        """Actualiza todos los textos de la UI al idioma indicado."""
        self._lang = lang
        t = _PL_T

        # Step 0
        self._lbl_step0_title.setText(t["step0_title"][lang])
        self._btn_browse_vid.setText(t["browse_video"][lang])
        self._chk_skip_calib.setText(t["skip_calib"][lang])
        self._btn_step0_next.setText(t["next_btn"][lang])
        self._grp_hole_size.setTitle(t["hole_size_grp"][lang])
        self._lbl_hole_radius.setText(t["radius_lbl"][lang])
        if self._pl_calib_frame is None:
            self._lbl_pl_frame.setText(t["no_video"][lang])

        # Step 1
        self._lbl_step1_title.setText(t["step1_title"][lang])
        self._btn_cancel_preprocess.setText(t["cancel_btn"][lang])

        # Step 2
        self._grp_keys.setTitle(t["grp_keys"][lang])
        self._grp_stats.setTitle(t["grp_stats"][lang])
        self._btn_finish.setText(t["finish_btn"][lang])
        self._btn_edit_labels.setText(t["edit_labels_btn"][lang])
        self._lbl_speed.setText(t["vel_lbl"][lang])
        self._btn_occlude.setText("O: " + t["tt_occlude"][lang])
        self._refresh_occlude_btn(self._frame_idx in self._occluded_snout)
        self._btn_rev.setToolTip(t["tt_rev"][lang])
        self._btn_prev.setToolTip(t["tt_prev"][lang])
        self._btn_pause.setToolTip(t["tt_pause"][lang])
        self._btn_next.setToolTip(t["tt_next"][lang])
        self._btn_fwd.setToolTip(t["tt_fwd"][lang])

        # Actualizar botones de etiqueta y stats
        for _, name, _, hex_c, _ in PRELABEL_KEYS:
            display = _KEY_DISPLAY[name][lang]
            if name in self._label_buttons:
                self._label_buttons[name].setText(display)
            # Stat labels: actualizar row labels en el QFormLayout
        # Reconstruir filas del stats layout con nuevo idioma
        stats_layout = self._grp_stats.layout()
        if stats_layout is not None:
            for i, (_, name, _, _, _) in enumerate(PRELABEL_KEYS):
                item = stats_layout.itemAt(i * 2)  # QFormLayout: label+field pairs
                if item and item.widget():
                    item.widget().setText(_KEY_DISPLAY[name][lang] + ":")

        # Panel derecho: importar coords + carpeta salida
        self._grp_pl_import.setTitle(t["import_coords_grp"][lang])
        self._btn_pl_import_coords.setText(t["import_btn"][lang])
        self._grp_pl_output.setTitle(t["output_folder_grp"][lang])
        self._btn_pl_select_output.setText(t["browse_btn"][lang])
        if self._output_dir_override is None:
            self._edit_pl_output.setText(t["output_default"][lang])

        # Botones de zona
        if self._pl_calib_zone_btns:
            _ZONE_TEXTS_PL = [
                ("Borde Exterior", "Exterior Border"),
                ("Borde Interior", "Interior Border"),
                ("Agujeros",       "Holes"),
                ("Borde Central",  "Central Border"),
            ]
            for btn, (es, en) in zip(self._pl_calib_zone_btns, _ZONE_TEXTS_PL):
                btn.setText(es if lang == "es" else en)

        # Instruccion de calibracion
        self._update_pl_calib_instruction()

        # Si hay label activo, actualizar su display
        if self._current_label:
            self._select_label(self._current_label)

    # ------------------------------------------------------------------
    # Limpieza
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
        super().closeEvent(event)
