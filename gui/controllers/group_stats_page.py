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

_TRANS_PATH = PROJECT_ROOT / "scripts" / "config" / "translations.json"

def _load_t() -> dict:
    try:
        import json as _j
        with open(_TRANS_PATH, "r", encoding="utf-8") as f:
            return _j.load(f).get("group_stats_page", {})
    except Exception:
        return {}

_T: dict = _load_t()


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
        self._lang: str = "es"
        self._build_ui()

    # ------------------------------------------------------------------
    # Construccion del interfaz
    # ------------------------------------------------------------------

    def _t(self, key: str) -> str:
        entry = _T.get(key, {})
        return entry.get(self._lang, entry.get("es", key))

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 14)
        root.setSpacing(10)

        self._lbl_title = QLabel(self._t("title"))
        self._lbl_title.setStyleSheet("font-size:18px; font-weight:bold; color:#2c3e50;")
        root.addWidget(self._lbl_title)

        self._lbl_sub = QLabel(self._t("subtitle"))
        self._lbl_sub.setWordWrap(True)
        self._lbl_sub.setStyleSheet("color:#555; font-size:11px;")
        root.addWidget(self._lbl_sub)

        # ---- Tabla de archivos ----
        self._grp_files = QGroupBox(self._t("grp_files"))
        self._grp_files.setStyleSheet("QGroupBox{font-weight:bold;}")
        ly_files = QVBoxLayout(self._grp_files)
        ly_files.setSpacing(6)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(
            [self._t("col_file"), self._t("col_group"), self._t("col_session")]
        )
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
        self._btn_add = QPushButton(self._t("btn_add"))
        self._btn_add.setStyleSheet(
            "background:#2980b9; color:white; padding:5px 16px; border-radius:3px;"
        )
        self._btn_rem = QPushButton(self._t("btn_remove"))
        self._btn_rem.setStyleSheet(
            "background:#c0392b; color:white; padding:5px 16px; border-radius:3px;"
        )
        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_rem)
        btn_row.addStretch()
        ly_files.addLayout(btn_row)
        root.addWidget(self._grp_files)

        # ---- Nombres de grupo ----
        self._grp_names = QGroupBox(self._t("grp_names"))
        self._grp_names.setStyleSheet("QGroupBox{font-weight:bold;}")
        ly_names = QHBoxLayout(self._grp_names)
        self._lbl_ctrl = QLabel(self._t("lbl_ctrl"))
        ly_names.addWidget(self._lbl_ctrl)
        self._ctrl_name = QLineEdit("Control")
        self._ctrl_name.setMaximumWidth(150)
        self._ctrl_name.setPlaceholderText("Control")
        ly_names.addWidget(self._ctrl_name)
        ly_names.addSpacing(24)
        self._lbl_treat = QLabel(self._t("lbl_treat"))
        ly_names.addWidget(self._lbl_treat)
        self._treat_name = QLineEdit("FOLFOX")
        self._treat_name.setMaximumWidth(150)
        self._treat_name.setPlaceholderText("FOLFOX")
        ly_names.addWidget(self._treat_name)
        ly_names.addStretch()
        root.addWidget(self._grp_names)

        # ---- Carpeta de salida ----
        self._grp_out = QGroupBox(self._t("grp_out"))
        self._grp_out.setStyleSheet("QGroupBox{font-weight:bold;}")
        ly_out = QHBoxLayout(self._grp_out)
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText(self._t("ph_out"))
        self._btn_browse = QPushButton(self._t("browse_btn"))
        ly_out.addWidget(self._out_edit)
        ly_out.addWidget(self._btn_browse)
        root.addWidget(self._grp_out)

        # ---- Boton generar ----
        gen_row = QHBoxLayout()
        self._btn_gen = QPushButton(self._t("gen_btn"))
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
            self, self._t("dlg_add"), "",
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
            self._table.setItem(row, 2, QTableWidgetItem(self._t("week_default")))

    def _on_remove(self) -> None:
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()}, reverse=True)
        for r in rows:
            self._table.removeRow(r)

    def _on_browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, self._t("dlg_out"), "")
        if path:
            self._out_edit.setText(path)

    def _on_generate(self) -> None:
        if self._table.rowCount() == 0:
            QMessageBox.warning(self, self._t("err_no_files"), self._t("err_no_files_m"))
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
        self._btn_gen.setText(self._t("gen_btn_busy"))

        self._worker = _Worker(entries, output_dir, group_names)
        self._worker.log_msg.connect(self._log.append)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, folder: str) -> None:
        self._btn_gen.setEnabled(True)
        self._btn_gen.setText(self._t("gen_btn"))
        self._log.append(f"\n {self._t('done_log')} {folder}")
        QMessageBox.information(
            self, self._t("done_title"),
            f"{self._t('done_msg')}\n{folder}"
        )

    def _on_error(self, err: str) -> None:
        self._btn_gen.setEnabled(True)
        self._btn_gen.setText(self._t("gen_btn"))
        self._log.append(f"[ERROR] {err}")
        QMessageBox.critical(self, self._t("err_title"), err[:800])

    def apply_language(self, lang: str) -> None:
        self._lang = lang
        self._lbl_title.setText(self._t("title"))
        self._lbl_sub.setText(self._t("subtitle"))
        self._grp_files.setTitle(self._t("grp_files"))
        self._table.setHorizontalHeaderLabels(
            [self._t("col_file"), self._t("col_group"), self._t("col_session")]
        )
        self._btn_add.setText(self._t("btn_add"))
        self._btn_rem.setText(self._t("btn_remove"))
        self._grp_names.setTitle(self._t("grp_names"))
        self._lbl_ctrl.setText(self._t("lbl_ctrl"))
        self._lbl_treat.setText(self._t("lbl_treat"))
        self._grp_out.setTitle(self._t("grp_out"))
        self._out_edit.setPlaceholderText(self._t("ph_out"))
        self._btn_browse.setText(self._t("browse_btn"))
        if not (self._worker and self._worker.isRunning()):
            self._btn_gen.setText(self._t("gen_btn"))
