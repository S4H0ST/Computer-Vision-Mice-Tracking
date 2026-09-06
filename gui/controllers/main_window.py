"""
Controlador de la ventana principal. Carga main_window.ui y conecta todas las paginas:
  Inicio -> Calibracion -> Deteccion -> Resultados

Navegacion (indice del QStackedWidget):
  0 = page_home
  1 = page_calibration
  2 = page_detection
  3 = page_results

Flujo de calibracion (8 clics, equivalente a ImageCalibrator):
  Fase 0: 2 clics -> esquinas del borde exterior
  Fase 1: 2 clics -> esquinas del borde interior
  Fase 2: 4 clics -> centros de los agujeros
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QMessageBox, QListWidgetItem,
                             QSizePolicy, QDialog, QVBoxLayout, QTabWidget,
                             QTextBrowser, QDialogButtonBox, QStyle)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from config.config import paths
from gui.controllers.detect_worker import DetectionWorker


UI_PATH = Path(__file__).parent.parent / "main_window.ui"

HOLE_RADIUS: int = 15

# (widget_name, method, text_es, text_en)
_TRANSLATIONS: list[tuple] = [
    ("nav_home",              "setText",  "  Inicio",                         "  Home"),
    ("nav_results",           "setText",  "  Resultados",                     "  Results"),
    ("btn_back_calib",        "setText",  "← Volver",                         "← Back"),
    ("btn_back_detection",    "setText",  "← Volver",                         "← Back"),
    ("lbl_welcome",           "setText",  "Sistema de Seguimiento de Ratones","Mouse Behaviour Tracking System"),
    ("lbl_subtitle",          "setText",  "Analisis de comportamiento mediante vision por computador",
                                          "Behaviour analysis using computer vision"),
    ("grp_video",             "setTitle", "Deteccion desde Video",            "Detection from Video"),
    ("lbl_video_desc",        "setText",  "Carga un fichero de video (.mp4, .avi, .mov) para analizar el comportamiento del raton fuera de linea. El modelo procesara cada frame automaticamente.",
                                          "Load a video file (.mp4, .avi, .mov) to analyse mouse behaviour offline. The model will process each frame automatically."),
    ("btn_video",             "setText",  "Seleccionar Video",                "Select Video"),
    ("grp_camera",            "setTitle", "Deteccion en Vivo (Camara)",       "Live Detection (Camera)"),
    ("lbl_camera_desc",       "setText",  "Conecta una camara para analizar el comportamiento del raton en tiempo real. Requiere calibracion previa de la arena.",
                                          "Connect a camera to analyse mouse behaviour in real time. Arena calibration is required."),
    ("btn_camera",            "setText",  "Iniciar Camara",                   "Start Camera"),
    ("lbl_calib_title",       "setText",  "Calibracion de la Arena",          "Arena Calibration"),
    ("grp_exterior",          "setTitle", "Borde Exterior (px)",              "Exterior Border (px)"),
    ("grp_interior",          "setTitle", "Borde Interior (px)",              "Interior Border (px)"),
    ("grp_holes",             "setTitle", "Centros de Agujeros (px)",         "Hole Centers (px)"),
    ("grp_dimensions",        "setTitle", "Medidas Reales de la Caja",        "Real Box Dimensions"),
    ("s_lbl_width",           "setText",  "Ancho (cm):",                      "Width (cm):"),
    ("s_lbl_height",          "setText",  "Alto (cm):",                       "Height (cm):"),
    ("grp_import_coords",     "setTitle", "Importar Coordenadas",             "Import Coordinates"),
    ("btn_import_coords",     "setText",  "Examinar...",                      "Browse..."),
    ("lbl_coords_hint",       "setText",  "Guardado en: outputs/calibration/coords_camera.json o coords_video.json",
                                          "Saved to: outputs/calibration/coords_camera.json or coords_video.json"),
    ("grp_output_folder",     "setTitle", "Carpeta de Salida",                "Output Folder"),
    ("btn_select_output",     "setText",  "Examinar...",                      "Browse..."),
    ("btn_clear_calib",       "setText",  "Limpiar",                          "Clear"),
    ("btn_confirm_calib",     "setText",  "Siguiente →",                      "Next →"),
    ("lbl_stats_title",       "setText",  "Estadisticas en Vivo",             "Live Statistics"),
    ("grp_progress",          "setTitle", "Progreso",                         "Progress"),
    ("grp_behaviors",         "setTitle", "Comportamientos",                  "Behaviours"),
    ("s_lbl_frames",          "setText",  "Frames:",                          "Frames:"),
    ("s_lbl_fps",             "setText",  "FPS:",                             "FPS:"),
    ("s_lbl_time",            "setText",  "Tiempo de video:",                 "Video time:"),
    ("s_lbl_beh_idle",        "setText",  "Inactivo:",                        "Idle:"),
    ("s_lbl_beh_walking",     "setText",  "Caminando:",                       "Walking:"),
    ("s_lbl_beh_sniffing",    "setText",  "Olisqueando:",                     "Sniffing:"),
    ("s_lbl_beh_climbing",    "setText",  "Escalando:",                       "Climbing:"),
    ("s_lbl_beh_rearing",     "setText",  "Erguido:",                         "Rearing:"),
    ("s_lbl_beh_dipping",     "setText",  "Asomando:",                        "Head-dip:"),
    ("s_lbl_beh_grooming",    "setText",  "Aseo:",                            "Grooming:"),
    ("btn_stop",              "setText",  "Cancelar",                         "Cancel"),
    ("lbl_results_title",     "setText",  "Resultados de la Deteccion",       "Detection Results"),
    ("btn_browse_results",    "setText",  "Seleccionar Carpeta",              "Select Folder"),
    ("grp_files",             "setTitle", "Archivos Generados",               "Generated Files"),
    ("lbl_file_excel",        "setText",  "Estadisticas (.xlsx)",             "Statistics (.xlsx)"),
    ("lbl_desc_excel",        "setText",  "Tabla por frame con comportamientos, posicion y velocidad",
                                          "Per-frame table with behaviours, position and speed"),
    ("lbl_file_video1",       "setText",  "Video anotado",                    "Annotated video"),
    ("lbl_desc_video1",       "setText",  "Video con etiquetas y zonas calibradas superpuestas",
                                          "Video with behaviour labels and calibrated zones overlaid"),
    ("lbl_file_video2",       "setText",  "Video de recorrido",               "Clean video"),
    ("lbl_desc_video2",       "setText",  "Video solo con etiquetas, sin zonas de fondo",
                                          "Video with labels only, no background zones"),
    ("lbl_file_folder",       "setText",  "Carpeta de salida",                "Output folder"),
    ("lbl_desc_folder",       "setText",  "Contiene todos los archivos de esta ejecucion",
                                          "Contains all files from this run"),
    ("grp_summary",           "setTitle", "Resumen",                          "Summary"),
    ("s_lbl_duration",        "setText",  "Duracion:",                        "Duration:"),
    ("s_lbl_res_frames",      "setText",  "Frames totales:",                  "Total frames:"),
    ("s_lbl_res_model",       "setText",  "Modelo:",                          "Model:"),
    ("btn_open_excel",        "setText",  "Abrir",                            "Open"),
    ("btn_open_video1",       "setText",  "Abrir",                            "Open"),
    ("btn_open_video2",       "setText",  "Abrir",                            "Open"),
    ("btn_open_folder",       "setText",  "Abrir",                            "Open"),
    ("btn_new_detection",     "setText",  "Nueva Deteccion",                  "New Detection"),
    ("tab_images",            None,       None,                               None),
    # Avisos de validacion (el texto lo gestiona _update_confirm_state directamente)
    ("lbl_warn_output",       None,       None,                               None),
    ("lbl_warn_coords",       None,       None,                               None),
]

# Textos de aviso de validacion por idioma
_WARN_OUTPUT = {
    "es": "Selecciona una carpeta de salida para continuar.",
    "en": "Select an output folder to continue.",
}
_WARN_COORDS = {
    "es": "Completa los {done}/8 puntos de calibracion para continuar.",
    "en": "Set all {done}/8 calibration points to continue.",
}

_TAB_LABELS = {
    "es": ["Recorrido", "Mapa de Calor"],
    "en": ["Trajectory", "Heatmap"],
}

_CALIB_INSTRUCTIONS = {
    "es": [
        "Paso 1/3 — Clic en BORDE EXTERIOR: 2 esquinas opuestas de la pared exterior ({n}/2)",
        "Paso 2/3 — Clic en BORDE INTERIOR: 2 esquinas opuestas del suelo interior ({n}/2)",
        "Paso 3/3 — Clic en los 4 AGUJEROS (centra el clic en cada uno) ({n}/4)",
        "Calibracion completa — pulsa Confirmar para guardar",
    ],
    "en": [
        "Step 1/3 — Click EXTERIOR border: 2 opposite corners of the outer wall ({n}/2)",
        "Step 2/3 — Click INTERIOR border: 2 opposite corners of the inner floor ({n}/2)",
        "Step 3/3 — Click 4 HOLE centers (click the center of each hole) ({n}/4)",
        "Calibration complete — press Confirm to save",
    ],
}


_HELP_CSS = """
<style>
  body { font-family: Segoe UI, Arial, sans-serif; font-size: 13px; color: #2c3e50; margin: 12px; }
  h2   { color: #CB0017; margin-top: 18px; margin-bottom: 4px; }
  h3   { color: #34495e; margin-top: 14px; margin-bottom: 2px; }
  p    { margin: 4px 0 10px 0; line-height: 1.5; }
  ul   { margin: 4px 0 10px 16px; line-height: 1.6; }
  .step { background: #f4f6f7; border-left: 4px solid #CB0017;
          padding: 8px 12px; margin: 8px 0; border-radius: 2px; }
  .tip  { background: #eafaf1; border-left: 4px solid #27ae60;
          padding: 6px 12px; margin: 6px 0; border-radius: 2px; font-size: 12px; }
  kbd  { background: #ecf0f1; border: 1px solid #bdc3c7; border-radius: 3px;
         padding: 1px 5px; font-family: Consolas, monospace; font-size: 11px; }
</style>
"""

_HELP_NAV_HTML = _HELP_CSS + """
<h2>Application Overview</h2>
<p>Mouse Tracker analyses mouse behaviour in a Barnes maze arena using computer vision.
The workflow has four pages: <b>Home → Calibration → Detection → Results</b>.</p>

<h2>Home</h2>
<div class="step">
  <b>Select Video</b> — load an offline video file (.mp4, .avi, .mov, .mkv).<br>
  <b>Start Camera</b> — open the default webcam for real-time analysis.
</div>
<p>Either button proceeds to the <b>Calibration</b> page automatically.</p>

<h2>Calibration</h2>
<p>Click 8 points on the arena image to define its geometry:</p>
<ul>
  <li><b style="color:#cc0000">Step 1</b> — 2 clicks on opposite corners of the <b>exterior wall</b> (red rectangle).</li>
  <li><b style="color:#0000cc">Step 2</b> — 2 clicks on opposite corners of the <b>interior floor</b> (blue rectangle).</li>
  <li><b style="color:#007700">Step 3</b> — 4 clicks on the <b>centre of each hole</b> (green circles).</li>
</ul>
<p>Set the real-world box dimensions (cm) and choose an <b>Output Folder</b> before clicking <b>Next →</b>.</p>
<div class="tip">Tip: you can reuse a previous calibration via <b>Import Coordinates → Browse…</b>
and selecting a <code>coords_*.json</code> file.</div>
<div class="tip">Tip: <b>Clear</b> resets all 8 points so you can start over.</div>

<h2>Detection</h2>
<p>The model processes each frame and shows:</p>
<ul>
  <li>Live video feed with behaviour label overlaid.</li>
  <li>Frame count, FPS and video time elapsed.</li>
  <li>Running totals (seconds) per behaviour: Idle, Walking, Sniffing, Climbing, Rearing, Head-dip, Grooming.</li>
</ul>
<p>Press <b>Cancel</b> to stop early — partial results will still be saved.<br>
Use <b>← Back</b> (visible after cancelling) to correct the calibration and re-run.</p>

<h2>Results</h2>
<p>Displayed automatically after detection finishes, or navigate here via the sidebar to
browse a previous run folder.</p>
<ul>
  <li><b>Trajectory</b> tab — colour-coded path of the mouse across the arena.</li>
  <li><b>Heatmap</b> tab — density map showing where the mouse spent most time.</li>
  <li><b>Generated Files</b> panel — open the stats spreadsheet (.xlsx), annotated video,
      clean trajectory video, or the output folder directly.</li>
</ul>
<div class="tip">Tip: click <b>Select Folder</b> in the top-right to load results from any previous run.</div>

<h2>Keyboard Shortcuts</h2>
<ul>
  <li><kbd>Ctrl+N</kbd> — New Detection (from any page)</li>
  <li><kbd>Ctrl+Q</kbd> — Exit the application</li>
  <li><kbd>F1</kbd> — Open this guide</li>
</ul>
"""

_HELP_FAQ_HTML = _HELP_CSS + """
<h2>Frequently Asked Questions</h2>

<h3>What video formats are supported?</h3>
<p>MP4, AVI, MOV and MKV. Any format that OpenCV can decode on your system will work.</p>

<h3>Do I need to calibrate every time?</h3>
<p>No. Save the generated <code>coords_*.json</code> file from the output folder and
re-import it next session via <b>Import Coordinates → Browse…</b>.</p>

<h3>Where are the output files saved?</h3>
<p>In the folder you selected during calibration, inside a timestamped sub-folder
(e.g. <code>myvideo_20250101_120000/</code>). A copy of the calibration JSON is also
written to <code>outputs/calibration/</code> for convenience.</p>

<h3>What behaviours does the model detect?</h3>
<ul>
  <li><b>Idle</b> — mouse is stationary.</li>
  <li><b>Walking</b> — moving across the arena floor.</li>
  <li><b>Sniffing</b> — nose-down exploration.</li>
  <li><b>Climbing</b> — moving along the arena wall.</li>
  <li><b>Rearing</b> — standing on hind legs.</li>
  <li><b>Head-dip</b> — head extended into a hole.</li>
  <li><b>Grooming</b> — self-grooming posture.</li>
</ul>

<h3>Can I analyse a video without a connected camera?</h3>
<p>Yes — use <b>Select Video</b> on the Home page to load any recorded video file.</p>

<h3>The model is shown as "NOT FOUND" in the sidebar. What do I do?</h3>
<p>Place the YOLO weights file (<code>.pt</code>) in the path shown in
<code>scripts/config/config.py</code> under <code>yolo_model</code>.
Detection will not work until the model file is present.</p>

<h3>Can I stop detection mid-way and still get results?</h3>
<p>Yes. Press <b>Cancel</b>, confirm the prompt, and the app will save whatever has
been processed so far — trajectory image, heatmap, annotated video and stats spreadsheet
will all reflect the partial run.</p>

<h3>How do I switch the interface language?</h3>
<p>Click the <b>ES / EN</b> button at the bottom of the left sidebar to toggle between
Spanish and English.</p>
"""


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        uic.loadUi(str(UI_PATH), self)

        self._lang: str = "en"

        # Calibration state: 3-phase (exterior 2pts, interior 2pts, holes 4pts)
        self._calib_exterior: list[tuple[int, int]] = []
        self._calib_interior: list[tuple[int, int]] = []
        self._calib_holes:    list[tuple[int, int]] = []
        self._calib_frame: np.ndarray | None = None
        self._calib_scale_x: float = 1.0
        self._calib_scale_y: float = 1.0
        self._calib_offset_x: int = 0
        self._calib_offset_y: int = 0

        # Runtime state
        self._video_source = None
        self._custom_output_dir: Path | None = None
        self._output_dir: Path | None = None
        self._coords_json: Path | None = None
        self._worker: DetectionWorker | None = None
        self._output_paths: dict = {}
        self._result_runs: dict[str, Path] = {}
        self._detect_fps: float = 0.0
        self._detect_total_s: float = 0.0
        self._result_img_paths: dict[str, Path | None] = {}  # label name -> image path

        # Timer for elapsed time display
        self._timer = QTimer(self)
        self._elapsed_s: int = 0
        self._timer.timeout.connect(self._tick_timer)

        self._setup_model_status()
        self._connect_signals()
        # Evitar bucle de retroalimentacion donde el pixmap aumenta el sizeHint del label
        self.lbl_frame_display.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_trajectory_img.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_heatmap_img.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.statusbar.setVisible(False)
        self._apply_language()  # idioma por defecto: ingles
        self._update_confirm_state()

    # ------------------------------------------------------------------
    # Eventos del sistema
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.stackedWidget.currentIndex() == 1 and self._calib_frame is not None:
            self._display_calib_frame()
        elif self.stackedWidget.currentIndex() == 3:
            self._rescale_result_image(self.lbl_trajectory_img)
            self._rescale_result_image(self.lbl_heatmap_img)

    # ------------------------------------------------------------------
    # Configuracion inicial
    # ------------------------------------------------------------------

    def _setup_model_status(self) -> None:
        ok = paths.yolo_model.exists()
        status = paths.yolo_model.name if ok else "NOT FOUND"
        color  = "#2ecc71" if ok else "#e74c3c"
        self.lbl_model_status.setText(f"Model: {status}")
        self.lbl_model_status.setStyleSheet(
            f"font-size: 10px; color: {color}; padding: 0px 8px 14px 12px;"
        )

    def _connect_signals(self) -> None:
        # Sidebar navigation
        self.nav_home.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
        self.nav_results.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(3))

        # Home
        self.btn_video.clicked.connect(self._on_select_video)
        self.btn_camera.clicked.connect(self._on_select_camera)

        # Calibration
        self.lbl_frame_display.mousePressEvent = self._on_calib_click
        self.btn_back_calib.clicked.connect(self._on_back_calib_to_home)
        self.btn_clear_calib.clicked.connect(self._on_clear_calib)
        self.btn_confirm_calib.clicked.connect(self._on_confirm_calib)
        self.btn_import_coords.clicked.connect(self._on_import_coords)
        self.btn_select_output.clicked.connect(self._on_select_output_folder)
        self.spin_width_cm.valueChanged.connect(self._update_ratio_label)
        self.spin_height_cm.valueChanged.connect(self._update_ratio_label)

        # Deteccion: volver oculto hasta cancelar; cancelar detiene y desbloquea volver
        self.btn_back_detection.clicked.connect(self._on_back_detection_to_calib)
        self.btn_stop.clicked.connect(self._on_cancel_detection)

        # Results
        self.btn_browse_results.clicked.connect(self._on_browse_results)
        self.list_result_runs.currentItemChanged.connect(self._on_result_run_selected)
        self.btn_open_excel.clicked.connect(lambda: self._open_path(self._output_paths.get("excel")))
        self.btn_open_video1.clicked.connect(lambda: self._open_path(self._output_paths.get("video_annotated")))
        self.btn_open_video2.clicked.connect(lambda: self._open_path(self._output_paths.get("video_clean")))
        self.btn_open_folder.clicked.connect(lambda: self._open_path(self._output_paths.get("folder")))
        self.btn_new_detection.clicked.connect(self._on_new_detection)
        self.tab_images.currentChanged.connect(self._on_result_tab_changed)

        # Language
        self.btn_lang.clicked.connect(self._toggle_language)

        # Menu
        self.action_salir.triggered.connect(self.close)
        self.action_nueva_deteccion.triggered.connect(self._on_new_detection)
        self.action_help.triggered.connect(self._show_help_dialog)
        self.action_acerca_de.triggered.connect(self._show_about_dialog)
        self._setup_menu_icons()

    def _setup_menu_icons(self) -> None:
        s = self.style()
        self.action_nueva_deteccion.setIcon(s.standardIcon(QStyle.SP_MediaPlay))
        self.action_salir.setIcon(s.standardIcon(QStyle.SP_DialogCloseButton))
        self.action_help.setIcon(s.standardIcon(QStyle.SP_DialogHelpButton))
        self.action_acerca_de.setIcon(s.standardIcon(QStyle.SP_MessageBoxInformation))

    def _show_help_dialog(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dlg.setWindowTitle("User Guide — Mouse Tracker")
        dlg.setMinimumSize(680, 520)

        tabs = QTabWidget()

        nav = QTextBrowser()
        nav.setHtml(_HELP_NAV_HTML)
        nav.setOpenExternalLinks(False)
        tabs.addTab(nav, self.style().standardIcon(QStyle.SP_DialogHelpButton), "Navigation")

        faq = QTextBrowser()
        faq.setHtml(_HELP_FAQ_HTML)
        faq.setOpenExternalLinks(False)
        tabs.addTab(faq, self.style().standardIcon(QStyle.SP_MessageBoxQuestion), "FAQ")

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(dlg.accept)

        layout = QVBoxLayout(dlg)
        layout.addWidget(tabs)
        layout.addWidget(btn_box)
        dlg.exec_()

    def _show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "About Mouse Tracker",
            "<b>Mouse Tracker v1.0</b><br><br>"
            "Automated mouse behaviour analysis using computer vision.<br><br>"
            "<small style='color:#7f8c8d;'>Universidad Rey Juan Carlos, 2026</small>",
        )

    # ------------------------------------------------------------------
    # Pagina inicio
    # ------------------------------------------------------------------

    def _on_select_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video", "",
            "Videos (*.mp4 *.avi *.mov *.mkv);;All files (*.*)"
        )
        if not path:
            return
        self._video_source = Path(path)
        self._reset_calib_state()
        self._load_calib_frame_from_video()
        self.stackedWidget.setCurrentIndex(1)
        # Renderizar DESPUES de que el layout asigne el tamano final al label
        # (evita el drift del primer clic cuando el widget aun no esta pintado)
        QTimer.singleShot(60, self._display_calib_frame)

    def _on_select_camera(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", "Could not open camera.")
            return
        ret, frame = cap.read()
        cap.release()
        if not ret:
            QMessageBox.warning(self, "Error", "Could not capture a frame from the camera.")
            return
        self._video_source = 0
        self._reset_calib_state()
        self._calib_frame = frame
        self.stackedWidget.setCurrentIndex(1)
        QTimer.singleShot(60, self._display_calib_frame)

    # ------------------------------------------------------------------
    # Calibracion
    # ------------------------------------------------------------------

    def _reset_calib_state(self) -> None:
        self._calib_exterior = []
        self._calib_interior = []
        self._calib_holes = []
        self._update_calib_fields()
        self._update_calib_instruction()

    def _calib_phase(self) -> int:
        if len(self._calib_exterior) < 2:
            return 0
        if len(self._calib_interior) < 2:
            return 1
        return 2

    def _calib_done(self) -> bool:
        return (
            len(self._calib_exterior) == 2
            and len(self._calib_interior) == 2
            and len(self._calib_holes) == 4
        )

    def _load_calib_frame_from_video(self) -> None:
        cap = cv2.VideoCapture(str(self._video_source))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            QMessageBox.warning(self, "Error", "Could not read the first frame of the video.")
            return
        self._calib_frame = frame

    def _draw_grid(self, frame: np.ndarray, divisions: int = 12) -> None:
        """Cuadricula semitransparente sobre el frame para facilitar la alineacion de puntos."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        gray = (210, 210, 210)
        for i in range(1, divisions):
            x = int(w * i / divisions)
            cv2.line(overlay, (x, 0), (x, h), gray, 1)
        for j in range(1, divisions):
            y = int(h * j / divisions)
            cv2.line(overlay, (0, y), (w, y), gray, 1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    def _display_calib_frame(self) -> None:
        if self._calib_frame is None:
            return
        frame = self._calib_frame.copy()

        # 1. Dibujar puntos sobre el frame en coordenadas originales
        # Exterior (rojo)
        for pt in self._calib_exterior:
            cv2.circle(frame, pt, 8, (0, 0, 255), -1)
        if len(self._calib_exterior) == 2:
            xs = [p[0] for p in self._calib_exterior]
            ys = [p[1] for p in self._calib_exterior]
            cv2.rectangle(frame, (min(xs), min(ys)), (max(xs), max(ys)), (0, 0, 255), 2)
        # Interior (azul)
        for pt in self._calib_interior:
            cv2.circle(frame, pt, 8, (255, 0, 0), -1)
        if len(self._calib_interior) == 2:
            xs = [p[0] for p in self._calib_interior]
            ys = [p[1] for p in self._calib_interior]
            cv2.rectangle(frame, (min(xs), min(ys)), (max(xs), max(ys)), (255, 0, 0), 2)
        # Agujeros (verde)
        for pt in self._calib_holes:
            cv2.circle(frame, pt, HOLE_RADIUS, (0, 255, 0), 2)
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)

        # 2. Escalar al tamano del label
        label = self.lbl_frame_display
        lw, lh = label.width(), label.height()
        oh, ow = frame.shape[:2]
        scale = min(lw / ow, lh / oh)
        dw = int(ow * scale)
        dh = int(oh * scale)
        self._calib_scale_x  = ow / dw
        self._calib_scale_y  = oh / dh
        self._calib_offset_x = (lw - dw) // 2
        self._calib_offset_y = (lh - dh) // 2

        resized = cv2.resize(frame, (dw, dh))

        # 3. Cuadricula sobre el frame ya escalado (lineas de 1px siempre)
        self._draw_grid(resized)

        rgb    = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        qimg   = QImage(rgb.data, dw, dh, dw * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        canvas = QPixmap(lw, lh)
        canvas.fill(Qt.black)
        from PyQt5.QtGui import QPainter
        painter = QPainter(canvas)
        painter.drawPixmap(self._calib_offset_x, self._calib_offset_y, pixmap)
        painter.end()

        label.setPixmap(canvas)

    def _on_calib_click(self, event) -> None:
        if self._calib_done():
            return
        if self._calib_frame is None:
            return

        lx = event.x() - self._calib_offset_x
        ly = event.y() - self._calib_offset_y
        if lx < 0 or ly < 0:
            return

        orig_x = int(lx * self._calib_scale_x)
        orig_y = int(ly * self._calib_scale_y)
        oh, ow = self._calib_frame.shape[:2]
        orig_x = max(0, min(orig_x, ow - 1))
        orig_y = max(0, min(orig_y, oh - 1))

        phase = self._calib_phase()
        if phase == 0:
            self._calib_exterior.append((orig_x, orig_y))
        elif phase == 1:
            self._calib_interior.append((orig_x, orig_y))
        else:
            self._calib_holes.append((orig_x, orig_y))

        self._update_calib_fields()
        self._update_calib_instruction()
        self._display_calib_frame()
        self._update_confirm_state()

    def _update_calib_fields(self) -> None:
        def fmt(pts, idx):
            if idx < len(pts):
                x, y = pts[idx]
                return f"({x}, {y})"
            return ""

        self.edit_ext_p1.setText(fmt(self._calib_exterior, 0))
        self.edit_ext_p2.setText(fmt(self._calib_exterior, 1))
        self.edit_int_p1.setText(fmt(self._calib_interior, 0))
        self.edit_int_p2.setText(fmt(self._calib_interior, 1))
        self.edit_hole_1.setText(fmt(self._calib_holes, 0))
        self.edit_hole_2.setText(fmt(self._calib_holes, 1))
        self.edit_hole_3.setText(fmt(self._calib_holes, 2))
        self.edit_hole_4.setText(fmt(self._calib_holes, 3))
        self._update_ratio_label()

    def _update_calib_instruction(self) -> None:
        msgs = _CALIB_INSTRUCTIONS[self._lang]
        phase = self._calib_phase()
        if self._calib_done():
            txt = msgs[3]
        elif phase == 0:
            txt = msgs[0].format(n=len(self._calib_exterior))
        elif phase == 1:
            txt = msgs[1].format(n=len(self._calib_interior))
        else:
            txt = msgs[2].format(n=len(self._calib_holes))
        self.lbl_calib_instructions.setText(txt)

    def _update_ratio_label(self) -> None:
        if len(self._calib_exterior) < 2:
            lbl = "Relacion: — px/cm" if self._lang == "es" else "Ratio: — px/cm"
            self.lbl_px_cm_ratio.setText(lbl)
            return
        xs = [p[0] for p in self._calib_exterior]
        ys = [p[1] for p in self._calib_exterior]
        px_w = max(xs) - min(xs)
        px_h = max(ys) - min(ys)
        cm_w = self.spin_width_cm.value()
        cm_h = self.spin_height_cm.value()
        if cm_w > 0 and cm_h > 0:
            ratio = ((px_w / cm_w) + (px_h / cm_h)) / 2
            prefix = "Relacion" if self._lang == "es" else "Ratio"
            self.lbl_px_cm_ratio.setText(f"{prefix}: {ratio:.1f} px/cm")

    def _update_confirm_state(self) -> None:
        """Habilita/deshabilita Siguiente y muestra avisos de lo que falta."""
        coords_ok = self._calib_done()
        folder_ok = self._custom_output_dir is not None

        done = len(self._calib_exterior) + len(self._calib_interior) + len(self._calib_holes)
        self.lbl_warn_coords.setText(_WARN_COORDS[self._lang].format(done=done))
        self.lbl_warn_coords.setVisible(not coords_ok)

        self.lbl_warn_output.setText(_WARN_OUTPUT[self._lang])
        self.lbl_warn_output.setVisible(not folder_ok)

        self.btn_confirm_calib.setEnabled(coords_ok and folder_ok)

    def _on_clear_calib(self) -> None:
        self._reset_calib_state()
        self._display_calib_frame()
        self._update_confirm_state()

    def _on_import_coords(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Coordinates", "", "JSON Files (*.json);;All Files (*.*)"
        )
        if not path:
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)

            ext = data.get("exterior", [])
            intr = data.get("interior", [])
            holes = data.get("holes", [])

            if len(ext) >= 2:
                self._calib_exterior = [(int(p[0]), int(p[1])) for p in ext[:2]]
            if len(intr) >= 2:
                self._calib_interior = [(int(p[0]), int(p[1])) for p in intr[:2]]
            if holes:
                self._calib_holes = [(int(p[0]), int(p[1])) for p in holes[:4]]

            if "box_width_cm" in data:
                self.spin_width_cm.setValue(float(data["box_width_cm"]))
            if "box_height_cm" in data:
                self.spin_height_cm.setValue(float(data["box_height_cm"]))

            self._update_calib_fields()
            self._update_calib_instruction()
            if self._calib_frame is not None:
                self._display_calib_frame()

            fname = Path(path).name
            self.lbl_import_status.setText(f"✓ {fname}")
            self.lbl_import_status.setStyleSheet("color: #27ae60; font-size: 11px;")
            self._update_confirm_state()
        except Exception as e:
            self.lbl_import_status.setText(f"Error: {e}")
            self.lbl_import_status.setStyleSheet("color: #e74c3c; font-size: 11px;")

    def _on_select_output_folder(self) -> None:
        default = str(paths.detect_dir) if paths.detect_dir.exists() else str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", default)
        if folder:
            self._custom_output_dir = Path(folder)
            self.edit_output_folder.setText(str(self._custom_output_dir))
            self._update_confirm_state()

    def _on_confirm_calib(self) -> None:
        e = [[p[0], p[1]] for p in self._calib_exterior]
        i = [[p[0], p[1]] for p in self._calib_interior]
        h = [[p[0], p[1]] for p in self._calib_holes]
        cm_w = self.spin_width_cm.value()
        cm_h = self.spin_height_cm.value()

        coords_data = {
            "exterior":   e,
            "interior":   i,
            "holes":      h,
            "hole_radius": HOLE_RADIUS,
            "limits_inner": {
                "x_min": min(i[0][0], i[1][0]), "x_max": max(i[0][0], i[1][0]),
                "y_min": min(i[0][1], i[1][1]), "y_max": max(i[0][1], i[1][1]),
            },
            "limits_outer": {
                "x_min": min(e[0][0], e[1][0]), "x_max": max(e[0][0], e[1][0]),
                "y_min": min(e[0][1], e[1][1]), "y_max": max(e[0][1], e[1][1]),
            },
            "box_width_cm":  cm_w,
            "box_height_cm": cm_h,
        }

        is_camera = isinstance(self._video_source, int)
        stem = "camara" if is_camera else Path(self._video_source).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        coords_name = "coords_camera.json" if is_camera else "coords_video.json"

        base = self._custom_output_dir if self._custom_output_dir else paths.detect_dir
        self._output_dir = base / f"{stem}_{timestamp}"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Copia por ejecucion (dentro de la carpeta de salida del run)
        self._coords_json = self._output_dir / coords_name
        with open(self._coords_json, "w") as f:
            json.dump(coords_data, f, indent=4)

        # Copias de referencia en outputs/calibration/
        calib_dir = paths.coords_json.parent
        calib_dir.mkdir(parents=True, exist_ok=True)
        # Archivo con nombre especifico segun fuente (camara o video)
        with open(calib_dir / coords_name, "w") as f:
            json.dump(coords_data, f, indent=4)
        # Alias generico coords.json para compatibilidad con scripts CLI
        with open(paths.coords_json, "w") as f:
            json.dump(coords_data, f, indent=4)

        self._start_detection()

    # ------------------------------------------------------------------
    # Navegacion entre paginas
    # ------------------------------------------------------------------

    def _on_back_calib_to_home(self) -> None:
        """Calibracion ← Volver → Inicio. Resetea todo el estado."""
        self._timer.stop()
        self._reset_calib_state()
        self._calib_frame = None
        self._video_source = None
        self.stackedWidget.setCurrentIndex(0)

    def _on_back_detection_to_calib(self) -> None:
        """Deteccion ← Volver → Calibracion. Solo activo tras cancelar."""
        self.stackedWidget.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # Deteccion
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        s = int(seconds)
        return f"{s // 60}:{s % 60:02d}"

    def _start_detection(self) -> None:
        is_camera = isinstance(self._video_source, int)
        source_name = f"Camera [{self._video_source}]" if is_camera else Path(self._video_source).name
        self.lbl_source.setText(f"Source: {source_name}")

        # Calcula duracion total del video antes de iniciar el worker
        if not is_camera:
            cap = cv2.VideoCapture(str(self._video_source))
            self._detect_fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n_frames             = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self._detect_total_s = n_frames / self._detect_fps if self._detect_fps > 0 else 0.0
            cap.release()
        else:
            self._detect_fps     = 30.0
            self._detect_total_s = 0.0

        for lbl in (self.lbl_beh_idle, self.lbl_beh_walking, self.lbl_beh_sniffing,
                    self.lbl_beh_climbing, self.lbl_beh_rearing, self.lbl_beh_dipping,
                    self.lbl_beh_grooming):
            lbl.setText("0 s")
        self.lbl_frames_count.setText("0")
        self.lbl_fps_count.setText("—")
        self.lbl_time_count.setText("—")
        self.lbl_elapsed_time.setText("00:00")
        self.log_detection.clear()
        self.lbl_video_feed.setText("Starting..." if self._lang == "en" else "Iniciando...")

        # Bloquea el boton volver mientras la deteccion esta en curso
        self.btn_back_detection.setVisible(False)
        self.btn_stop.setEnabled(True)

        self.stackedWidget.setCurrentIndex(2)

        self._elapsed_s = 0
        self._timer.start(1000)

        self._worker = DetectionWorker(self._video_source, self._output_dir, self._coords_json)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.log_msg.connect(self._on_log_msg)
        self._worker.finished.connect(self._on_detection_finished)
        self._worker.error.connect(self._on_detection_error)
        self._worker.start()

    @pyqtSlot(object, dict, int)
    def _on_frame_ready(self, frame: np.ndarray, stats: dict, frame_idx: int) -> None:
        h, w = frame.shape[:2]
        label = self.lbl_video_feed
        lw, lh = label.width(), label.height()
        scale = min(lw / w, lh / h)
        dw, dh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (dw, dh))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        qimg    = QImage(rgb.data, dw, dh, dw * 3, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))

        fps_est = frame_idx / max(self._elapsed_s, 1)
        self.lbl_frames_count.setText(str(frame_idx))
        self.lbl_fps_count.setText(f"{fps_est:.1f}")

        # Tiempo de video procesado / duracion total
        if self._detect_fps > 0:
            elapsed_video = frame_idx / self._detect_fps
            elapsed_str = self._fmt_time(elapsed_video)
            if self._detect_total_s > 0:
                total_str = self._fmt_time(self._detect_total_s)
                self.lbl_time_count.setText(f"{elapsed_str} / {total_str}")
            else:
                self.lbl_time_count.setText(elapsed_str)

        fps_approx = self._detect_fps if self._detect_fps > 0 else 30.0
        self.lbl_beh_idle.setText(f"{stats.get('immobile', 0) / fps_approx:.1f} s")
        self.lbl_beh_walking.setText(f"{stats.get('walking', 0) / fps_approx:.1f} s")
        self.lbl_beh_sniffing.setText(f"{stats.get('sniffing', 0) / fps_approx:.1f} s")
        self.lbl_beh_climbing.setText(f"{stats.get('climbing', 0) / fps_approx:.1f} s")
        self.lbl_beh_rearing.setText(f"{stats.get('rearing', 0) / fps_approx:.1f} s")
        self.lbl_beh_dipping.setText(f"{stats.get('dipping', 0) / fps_approx:.1f} s")
        self.lbl_beh_grooming.setText(f"{stats.get('grooming', 0) / fps_approx:.1f} s")

    @pyqtSlot(str)
    def _on_log_msg(self, msg: str) -> None:
        self.log_detection.append(msg)

    @pyqtSlot(dict)
    def _on_detection_finished(self, output_paths: dict) -> None:
        self._timer.stop()
        self._output_paths = output_paths
        self._show_results(output_paths)

    @pyqtSlot(str)
    def _on_detection_error(self, msg: str) -> None:
        self._timer.stop()
        # Desbloquea el boton volver para que pueda corregir la calibracion
        self.btn_back_detection.setVisible(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.critical(self, "Detection error", msg)

    def _on_cancel_detection(self) -> None:
        """Pide confirmacion antes de detener la deteccion en curso."""
        if self._worker and self._worker.isRunning():
            if self._lang == "es":
                title = "¿Cancelar deteccion?"
                msg = (
                    "Si cancelas ahora, el analisis quedara incompleto.\n\n"
                    "• El mapa de calor y el recorrido solo mostraran los frames ya procesados.\n"
                    "• Las estadisticas no reflejaran el comportamiento completo del animal.\n\n"
                    "¿Seguro que quieres cancelar?"
                )
            else:
                title = "Cancel detection?"
                msg = (
                    "If you cancel now, the analysis will be incomplete.\n\n"
                    "• The heatmap and trajectory will only show frames processed so far.\n"
                    "• Statistics will not reflect the animal's full behaviour.\n\n"
                    "Are you sure you want to cancel?"
                )
            reply = QMessageBox.question(
                self, title, msg,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            self._worker.request_stop()
        self._timer.stop()
        self.btn_back_detection.setVisible(True)
        self.btn_stop.setEnabled(False)

    def _tick_timer(self) -> None:
        self._elapsed_s += 1
        m, s = divmod(self._elapsed_s, 60)
        self.lbl_elapsed_time.setText(f"{m:02d}:{s:02d}")

    # ------------------------------------------------------------------
    # Resultados
    # ------------------------------------------------------------------

    def _on_browse_results(self) -> None:
        default = str(paths.detect_dir) if paths.detect_dir.exists() else str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select Results Folder", default)
        if not folder:
            return
        root = Path(folder)
        self.lbl_results_folder_path.setText(str(root))
        self.list_result_runs.clear()
        self._result_runs = {}

        candidates: list[Path] = []
        # Check if root itself is a run folder
        if self._looks_like_run(root):
            candidates.append(root)
        else:
            for sub in sorted(root.iterdir()):
                if sub.is_dir() and self._looks_like_run(sub):
                    candidates.append(sub)

        if not candidates:
            no_runs = "No se encontraron resultados" if self._lang == "es" else "No detection runs found"
            self.list_result_runs.addItem(no_runs)
            return

        for run in candidates:
            self._result_runs[run.name] = run
            self.list_result_runs.addItem(run.name)

    @staticmethod
    def _looks_like_run(folder: Path) -> bool:
        return bool(
            list(folder.glob("*.csv"))
            or list(folder.glob("*.mp4"))
            or list(folder.glob("stats/*.xlsx"))
        )

    def _on_result_run_selected(self, current: QListWidgetItem, previous) -> None:
        if current is None:
            return
        run_name = current.text()
        if run_name not in self._result_runs:
            return
        run_dir = self._result_runs[run_name]
        stats_dir = run_dir / "stats"

        def first(it):
            lst = list(it)
            return lst[0] if lst else None

        sd = stats_dir if stats_dir.exists() else run_dir
        paths_dict = {
            "excel":           first(sd.glob("stats_*.xlsx")) or first(run_dir.glob("stats_*.xlsx")),
            "trajectory":      first(sd.glob("trajectory_*.png")) or first(run_dir.glob("trajectory_*.png")),
            "heatmap":         first(sd.glob("heatmap_*.png")) or first(run_dir.glob("heatmap_*.png")),
            "video_annotated": first(run_dir.glob("*_anotado.mp4")),
            "video_clean":     first(run_dir.glob("*_clean.mp4")),
            "folder":          run_dir,
            "frame_count":     0,
            "duration_s":      0,
        }
        self._output_paths = paths_dict
        self._show_results(paths_dict)

    def _show_results(self, paths_dict: dict) -> None:
        self.stackedWidget.setCurrentIndex(3)

        dur  = paths_dict.get("duration_s", 0)
        frm  = paths_dict.get("frame_count", 0)
        m, s = divmod(int(dur), 60)
        self.lbl_res_duration.setText(f"{m:02d}:{s:02d}" if dur else "—")
        self.lbl_res_frames.setText(str(frm) if frm else "—")
        from config.config import paths as cfg_paths
        self.lbl_res_model.setText(cfg_paths.yolo_model.name)

        def short_name(p):
            return Path(p).name if p else "—"

        self.lbl_file_excel.setText(short_name(paths_dict.get("excel")))
        self.lbl_file_video1.setText(short_name(paths_dict.get("video_annotated")))
        self.lbl_file_video2.setText(short_name(paths_dict.get("video_clean")))

        for btn, key in [
            (self.btn_open_excel,   "excel"),
            (self.btn_open_video1,  "video_annotated"),
            (self.btn_open_video2,  "video_clean"),
        ]:
            p = paths_dict.get(key)
            btn.setEnabled(bool(p and Path(p).exists()))

        self._load_result_image(self.lbl_trajectory_img, paths_dict.get("trajectory"))
        self._load_result_image(self.lbl_heatmap_img,    paths_dict.get("heatmap"))

    def _load_result_image(self, label, img_path) -> None:
        na = "No disponible" if self._lang == "es" else "Not available"
        key = label.objectName()
        if not img_path or not Path(img_path).exists():
            label.setText(na)
            self._result_img_paths.pop(key, None)
            return
        self._result_img_paths[key] = Path(img_path)
        self._rescale_result_image(label)

    def _rescale_result_image(self, label) -> None:
        path = self._result_img_paths.get(label.objectName())
        if not path:
            return
        img = cv2.imread(str(path))
        if img is None:
            return
        lw, lh = label.width(), label.height()
        if lw < 10 or lh < 10:
            return
        h, w  = img.shape[:2]
        scale = min(lw / w, lh / h)
        dw, dh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (dw, dh))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        qimg    = QImage(rgb.data, dw, dh, dw * 3, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))

    def _on_result_tab_changed(self) -> None:
        idx = self.tab_images.currentIndex()
        lbl = self.lbl_trajectory_img if idx == 0 else self.lbl_heatmap_img
        QTimer.singleShot(30, lambda: self._rescale_result_image(lbl))

    def _open_path(self, path) -> None:
        if not path:
            return
        p = Path(path)
        if p.exists():
            os.startfile(str(p))

    def _on_new_detection(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait()
        self._timer.stop()
        self._reset_calib_state()
        self._calib_frame = None
        self._video_source = None
        self._output_paths = {}
        # Restaura el estado de los botones de la pagina de deteccion para la siguiente ejecucion
        self.btn_back_detection.setVisible(False)
        self.btn_stop.setEnabled(True)
        self.stackedWidget.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Idioma
    # ------------------------------------------------------------------

    def _toggle_language(self) -> None:
        self._lang = "es" if self._lang == "en" else "en"
        self._apply_language()

    def _apply_language(self) -> None:
        lang_idx = 2 if self._lang == "es" else 3

        for entry in _TRANSLATIONS:
            widget_name, method = entry[0], entry[1]
            text = entry[lang_idx]
            if method is None or text is None:
                continue
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                getattr(widget, method)(text)
            except Exception:
                pass

        tabs = _TAB_LABELS[self._lang]
        for i, title in enumerate(tabs):
            self.tab_images.setTabText(i, title)

        self.btn_lang.setText("EN" if self._lang == "es" else "ES")

        # Actualiza etiquetas dinamicas segun idioma activo
        self._update_ratio_label()
        self._update_calib_instruction()
        self._update_confirm_state()
        self._update_calib_legend()
        self._display_calib_frame()

        ph = "Predeterminada: outputs/detections/" if self._lang == "es" else "Default: outputs/detections/"
        self.edit_output_folder.setPlaceholderText(ph)

        self._update_behavior_legend()

    def _update_calib_legend(self) -> None:
        """Leyenda de colores de calibracion debajo de la imagen (fuera del frame)."""
        if self._lang == "es":
            items = [
                ("#ff0000", "Borde exterior"),
                ("#0000ff", "Borde interior"),
                ("#00ff00", "Agujeros"),
            ]
        else:
            items = [
                ("#ff0000", "Exterior border"),
                ("#0000ff", "Interior border"),
                ("#00ff00", "Holes"),
            ]
        parts = [f'<font color="{c}">■</font> {t}' for c, t in items]
        self.lbl_calib_legend.setText(" &nbsp;&nbsp; ".join(parts))

    def _update_behavior_legend(self) -> None:
        """Construye la leyenda de colores de comportamiento en el idioma activo."""
        # Colores en formato RGB (despues de conversion BGR->RGB al mostrar en Qt)
        if self._lang == "es":
            items = [
                ("#b4b4b4", "Inmovil"),
                ("#ffff00", "Caminando"),
                ("#ffc800", "Olfateando"),
                ("#ff00ff", "Escalando"),
                ("#00ff00", "Erguido"),
                ("#ffa500", "Asomando"),
                ("#b4ffb4", "Aseo"),
            ]
        else:
            items = [
                ("#b4b4b4", "Idle"),
                ("#ffff00", "Walking"),
                ("#ffc800", "Sniffing"),
                ("#ff00ff", "Climbing"),
                ("#00ff00", "Rearing"),
                ("#ffa500", "Head-dip"),
                ("#b4ffb4", "Grooming"),
            ]
        lines = []
        for i in range(0, len(items), 2):
            c1, t1 = items[i]
            c2, t2 = items[i + 1] if i + 1 < len(items) else (None, None)
            row = f'<font color="{c1}">■</font> {t1}'
            if c2:
                row += f' &nbsp;&nbsp; <font color="{c2}">■</font> {t2}'
            lines.append(row)
        self.lbl_behavior_legend.setText("<br>".join(lines))

    # ------------------------------------------------------------------
    # Cierre de ventana
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(3000)
        self._timer.stop()
        event.accept()
