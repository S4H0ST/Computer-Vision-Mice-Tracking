"""
Pagina "Comparar Grupos" del GUI.
Permite cargar multiples archivos stats_*.xlsx (uno por rata/sesion),
asignarles grupo (Control / Tratamiento) y sesion, y generar graficas
comparativas PNG + Excel de resumen.
"""

from __future__ import annotations
from pathlib import Path
import sys

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QAbstractItemView,
    QFileDialog, QLineEdit, QTextEdit, QGroupBox,
    QHeaderView, QMessageBox, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ------------------------------------------------------------------
# Worker
# ------------------------------------------------------------------

class _Worker(QThread):
    log_msg  = pyqtSignal(str)
    finished = pyqtSignal(str)   # carpeta de salida
    error    = pyqtSignal(str)

    def __init__(self, entries: list[dict], output_dir: Path,
                 group_names: tuple[str, str]) -> None:
        super().__init__()
        self._entries     = entries
        self._output_dir  = Path(output_dir)
        self._group_names = group_names

    def run(self) -> None:
        try:
            from utils.group_stats import GroupStatsGenerator
            gen = GroupStatsGenerator(self._entries, self._output_dir, self._group_names)
            out = gen.generate(log_fn=self.log_msg.emit)
            self.finished.emit(str(out))
        except Exception as exc:
            import traceback
            self.error.emit(traceback.format_exc())


# ------------------------------------------------------------------
# Pagina principal
# ------------------------------------------------------------------

class GroupStatsPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._worker: _Worker | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    # Construccion del interfaz
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 14)
        root.setSpacing(10)

        # Titulo
        lbl_title = QLabel("Comparar Grupos")
        lbl_title.setStyleSheet("font-size:18px; font-weight:bold; color:#2c3e50;")
        root.addWidget(lbl_title)

        lbl_sub = QLabel(
            "Carga archivos Excel (stats_*.xlsx) de detecciones individuales, "
            "asigna grupo y sesion, y genera graficas de barras con media ± SEM."
        )
        lbl_sub.setWordWrap(True)
        lbl_sub.setStyleSheet("color:#555; font-size:11px;")
        root.addWidget(lbl_sub)

        # ---- Tabla de archivos ----
        grp_files = QGroupBox("Archivos de sesion")
        grp_files.setStyleSheet("QGroupBox{font-weight:bold;}")
        ly_files  = QVBoxLayout(grp_files)
        ly_files.setSpacing(6)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Archivo", "Grupo", "Sesion"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self._table.setColumnWidth(1, 130)
        self._table.setColumnWidth(2, 130)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setMinimumHeight(160)
        self._table.setAlternatingRowColors(True)
        ly_files.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("+ Añadir archivos")
        self._btn_add.setStyleSheet(
            "background:#2980b9; color:white; padding:5px 16px; border-radius:3px;"
        )
        self._btn_rem = QPushButton("Eliminar seleccion")
        self._btn_rem.setStyleSheet(
            "background:#c0392b; color:white; padding:5px 16px; border-radius:3px;"
        )
        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_rem)
        btn_row.addStretch()
        ly_files.addLayout(btn_row)
        root.addWidget(grp_files)

        # ---- Nombres de grupo ----
        grp_names = QGroupBox("Nombres de grupo")
        grp_names.setStyleSheet("QGroupBox{font-weight:bold;}")
        ly_names  = QHBoxLayout(grp_names)
        ly_names.addWidget(QLabel("Grupo control:"))
        self._ctrl_name = QLineEdit("Control")
        self._ctrl_name.setMaximumWidth(150)
        self._ctrl_name.setPlaceholderText("Control")
        ly_names.addWidget(self._ctrl_name)
        ly_names.addSpacing(24)
        ly_names.addWidget(QLabel("Grupo tratamiento:"))
        self._treat_name = QLineEdit("FOLFOX")
        self._treat_name.setMaximumWidth(150)
        self._treat_name.setPlaceholderText("FOLFOX")
        ly_names.addWidget(self._treat_name)
        ly_names.addStretch()
        root.addWidget(grp_names)

        # ---- Carpeta de salida ----
        grp_out = QGroupBox("Carpeta de salida")
        grp_out.setStyleSheet("QGroupBox{font-weight:bold;}")
        ly_out  = QHBoxLayout(grp_out)
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Por defecto: carpeta del primer archivo / comparacion_grupos")
        self._btn_browse = QPushButton("Examinar...")
        ly_out.addWidget(self._out_edit)
        ly_out.addWidget(self._btn_browse)
        root.addWidget(grp_out)

        # ---- Boton generar ----
        gen_row = QHBoxLayout()
        self._btn_gen = QPushButton("Generar Graficas y Excel")
        self._btn_gen.setStyleSheet(
            "QPushButton{"
            "  background:#27ae60; color:white; font-size:14px; font-weight:bold;"
            "  padding:10px 36px; border-radius:4px;"
            "}"
            "QPushButton:disabled{background:#95a5a6;}"
        )
        self._btn_gen.setMinimumHeight(44)
        gen_row.addStretch()
        gen_row.addWidget(self._btn_gen)
        gen_row.addStretch()
        root.addLayout(gen_row)

        # ---- Log ----
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(80)
        self._log.setMaximumHeight(130)
        self._log.setStyleSheet(
            "background:#1e1e1e; color:#d4d4d4; font-family:Consolas,monospace; font-size:11px;"
        )
        root.addWidget(self._log)

        # ---- Conexiones ----
        self._btn_add.clicked.connect(self._on_add)
        self._btn_rem.clicked.connect(self._on_remove)
        self._btn_browse.clicked.connect(self._on_browse_out)
        self._btn_gen.clicked.connect(self._on_generate)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_add(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Seleccionar archivos Excel de estadisticas", "",
            "Excel (*.xlsx);;Todos (*.*)"
        )
        if not paths:
            return
        ctrl = self._ctrl_name.text().strip() or "Control"
        for p in paths:
            row = self._table.rowCount()
            self._table.insertRow(row)

            item_file = QTableWidgetItem(Path(p).name)
            item_file.setData(Qt.UserRole, str(p))
            item_file.setFlags(item_file.flags() & ~Qt.ItemIsEditable)
            item_file.setToolTip(str(p))
            self._table.setItem(row, 0, item_file)

            self._table.setItem(row, 1, QTableWidgetItem(ctrl))
            self._table.setItem(row, 2, QTableWidgetItem("Semana 1"))

    def _on_remove(self) -> None:
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()}, reverse=True)
        for r in rows:
            self._table.removeRow(r)

    def _on_browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Carpeta de salida", "")
        if path:
            self._out_edit.setText(path)

    def _on_generate(self) -> None:
        if self._table.rowCount() == 0:
            QMessageBox.warning(self, "Sin archivos", "Añade al menos un archivo Excel.")
            return

        entries: list[dict] = []
        for row in range(self._table.rowCount()):
            path  = self._table.item(row, 0).data(Qt.UserRole)
            group = (self._table.item(row, 1).text() if self._table.item(row, 1) else "Control")
            sess  = (self._table.item(row, 2).text() if self._table.item(row, 2) else "Sesion 1")
            entries.append({"path": path, "group": group, "session": sess})

        out_str = self._out_edit.text().strip()
        if out_str:
            output_dir = Path(out_str)
        else:
            output_dir = Path(entries[0]["path"]).parent / "comparacion_grupos"

        group_names = (
            self._ctrl_name.text().strip()  or "Control",
            self._treat_name.text().strip() or "Tratamiento",
        )

        self._log.clear()
        self._btn_gen.setEnabled(False)
        self._btn_gen.setText("Generando...")

        self._worker = _Worker(entries, output_dir, group_names)
        self._worker.log_msg.connect(self._log.append)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, folder: str) -> None:
        self._btn_gen.setEnabled(True)
        self._btn_gen.setText("Generar Graficas y Excel")
        self._log.append(f"\n Completado. Carpeta: {folder}")
        QMessageBox.information(
            self, "Completado",
            f"Graficas y Excel generados en:\n{folder}"
        )

    def _on_error(self, err: str) -> None:
        self._btn_gen.setEnabled(True)
        self._btn_gen.setText("Generar Graficas y Excel")
        self._log.append(f"[ERROR] {err}")
        QMessageBox.critical(self, "Error al generar", err[:800])

    # ------------------------------------------------------------------
    # Soporte de idioma (preparado para bilingue)
    # ------------------------------------------------------------------

    def apply_language(self, lang: str) -> None:
        pass
