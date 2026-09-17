# -*- coding: utf-8 -*-
"""
Pagina de entrenamiento (indice 5 del stackedWidget principal).

Clases:
    TrainWorker  — QThread que ejecuta model.train() y copia el mejor .pt.
    TrainPage    — QWidget con formulario + consola para lanzar el entrenamiento.
"""

import os
import platform
import shutil
import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QFileDialog, QMessageBox, QSizePolicy,
    QFormLayout, QGroupBox, QFrame,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

_DEV_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_DEV_ROOT / "scripts"))

from config.config import paths, train_cfg

# paths.root is always correct: exe dir when frozen, project root in dev.
# _DEV_ROOT is only used for the translation file at import time (safe).
_TRANS_PATH = _DEV_ROOT / "scripts" / "config" / "translations.json"

def _load_train_t() -> dict:
    try:
        import json as _j
        with open(_TRANS_PATH, "r", encoding="utf-8") as f:
            return _j.load(f).get("train_page", {})
    except Exception:
        return {}

_T: dict = _load_train_t()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class TrainWorker(QThread):
    """Ejecuta YOLO .train() en segundo plano."""

    log_msg  = pyqtSignal(str)
    finished = pyqtSignal(str, dict)   # ruta del .pt guardado, metricas
    error    = pyqtSignal(str)

    def __init__(self, yaml_path: Path, model_name: str, parent=None) -> None:
        super().__init__(parent)
        self._yaml_path  = yaml_path
        self._model_name = model_name

    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            self._train()
        except Exception as exc:
            self.error.emit(str(exc))

    def _train(self) -> None:
        import time
        from ultralytics import YOLO

        base = str(paths.yolo_model) if paths.yolo_model.exists() else train_cfg.base_model
        self.log_msg.emit(f"Cargando modelo base: {base}")
        self.log_msg.emit("Calibrando batch optimo en GPU... (puede tardar 1-2 min)")
        model = YOLO(base)

        runs_dir = paths.root / "runs" / "train"

        # --- Callbacks de progreso ---
        _t = [0.0]      # tiempo inicio de epoca
        _best = [0.0]   # mejor mAP50(B) visto

        def _on_train_start(trainer):
            nb = getattr(trainer, "nb", "?")
            self.log_msg.emit(
                f"Entrenamiento iniciado — epochs={trainer.epochs}, "
                f"batch={trainer.batch_size}, lotes/epoca={nb}"
            )

        def _on_epoch_start(trainer):
            _t[0] = time.time()
            epoch = trainer.epoch + 1
            self.log_msg.emit(f"-- Epoca {epoch}/{trainer.epochs} --")

        def _on_epoch_end(trainer):
            elapsed = time.time() - _t[0]
            epoch   = trainer.epoch + 1
            total   = trainer.epochs
            m       = trainer.metrics
            map50b  = m.get("metrics/mAP50(B)", 0.0)
            map50p  = m.get("metrics/mAP50(P)", 0.0)
            if map50b > _best[0]:
                _best[0] = map50b
            remaining_s = (total - epoch) * elapsed
            eta = f"{int(remaining_s // 3600):02d}h{int((remaining_s % 3600) // 60):02d}m"
            self.log_msg.emit(
                f"  [{elapsed:.0f}s] mAP50(B)={map50b:.3f}  mAP50(P)={map50p:.3f}"
                f"  mejor={_best[0]:.3f}  ETA {eta}"
            )

        model.add_callback("on_train_start",     _on_train_start)
        model.add_callback("on_train_epoch_start", _on_epoch_start)
        model.add_callback("on_train_epoch_end",   _on_epoch_end)

        # En Windows, DataLoader con workers>0 da problemas de multiprocessing
        workers = 0 if platform.system() == "Windows" else 8

        model.train(
            data=str(self._yaml_path),
            epochs=train_cfg.epochs,
            imgsz=train_cfg.imgsz,
            batch=train_cfg.batch_size,
            device=train_cfg.device,
            workers=workers,
            project=str(runs_dir),
            name="train_exp",
            exist_ok=True,
            patience=30,
            dropout=0.2,
            weight_decay=0.0008,
            cos_lr=True,
            degrees=180,
            translate=0.05,
            scale=0.4,
            fliplr=0.5,
            flipud=0.5,
            mosaic=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            erasing=0.0,
        )

        self.log_msg.emit("Entrenamiento finalizado. Buscando best.pt...")

        # Extraer metricas antes de limpiar
        metrics: dict = {}
        try:
            for k, v in model.trainer.metrics.items():
                try:
                    metrics[k] = round(float(v), 4)
                except Exception:
                    pass
        except Exception:
            pass

        # Localizar best.pt
        best_pt: Path | None = None
        try:
            save_dir = Path(model.trainer.save_dir)
            candidate = save_dir / "weights" / "best.pt"
            if candidate.exists():
                best_pt = candidate
        except Exception:
            save_dir = None

        if best_pt is None:
            fallback = runs_dir / "train_exp" / "weights" / "best.pt"
            if fallback.exists():
                best_pt = fallback
                save_dir = fallback.parent.parent

        if best_pt is None:
            raise FileNotFoundError("No se encontro best.pt tras el entrenamiento.")

        # Copiar al directorio de modelos con nombre elegido
        paths.models_dir.mkdir(parents=True, exist_ok=True)
        dest = paths.models_dir / f"{self._model_name}.pt"
        suffix = 2
        while dest.exists():
            dest = paths.models_dir / f"{self._model_name}_{suffix}.pt"
            suffix += 1

        shutil.copy2(str(best_pt), str(dest))
        self.log_msg.emit(f"Modelo guardado en: {dest}")

        # Limpiar directorio temporal de runs
        if save_dir is not None:
            try:
                shutil.rmtree(str(save_dir), ignore_errors=True)
                self.log_msg.emit("Carpeta temporal de entrenamiento eliminada.")
            except Exception as exc:
                self.log_msg.emit(f"[!] No se pudo eliminar la carpeta temporal: {exc}")

        self.finished.emit(str(dest), metrics)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

class TrainPage(QWidget):
    """Formulario de entrenamiento + consola de salida."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._worker: TrainWorker | None = None
        self._yaml_path: Path | None = None
        self._dataset_dir: Path | None = None
        self._lang: str = "es"
        self._build_ui()

    # ------------------------------------------------------------------
    # Construccion de la UI
    # ------------------------------------------------------------------

    def _t(self, key: str) -> str:
        entry = _T.get(key, {})
        return entry.get(self._lang, entry.get("es", key))

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        self._lbl_title = QLabel(self._t("title"))
        self._lbl_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #CB0017;")
        root.addWidget(self._lbl_title)

        self._lbl_subtitle = QLabel(self._t("subtitle"))
        self._lbl_subtitle.setWordWrap(True)
        self._lbl_subtitle.setStyleSheet("font-size: 12px; color: #555;")
        root.addWidget(self._lbl_subtitle)

        self._lbl_dataset_status = QLabel()
        self._lbl_dataset_status.setWordWrap(True)
        self._lbl_dataset_status.setStyleSheet("font-size: 12px; padding: 6px 10px; border-radius: 4px;")
        self._lbl_dataset_status.setVisible(False)
        root.addWidget(self._lbl_dataset_status)

        self._form_group = QGroupBox(self._t("form_group"))
        self._form_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
        form_layout = QFormLayout(self._form_group)
        form_layout.setSpacing(10)
        form_layout.setContentsMargins(16, 16, 16, 16)

        # Dataset folder
        ds_row = QHBoxLayout()
        self._edit_dataset = QLineEdit()
        self._edit_dataset.setReadOnly(True)
        self._edit_dataset.setPlaceholderText(self._t("ph_dataset"))
        ds_row.addWidget(self._edit_dataset)
        self._btn_browse_ds = QPushButton(self._t("browse_btn"))
        self._btn_browse_ds.setFixedWidth(100)
        self._btn_browse_ds.clicked.connect(self._on_browse_dataset)
        ds_row.addWidget(self._btn_browse_ds)
        self._lbl_row_dataset = QLabel(self._t("lbl_dataset"))
        form_layout.addRow(self._lbl_row_dataset, ds_row)

        # data.yaml
        yaml_row = QHBoxLayout()
        self._edit_yaml = QLineEdit()
        self._edit_yaml.setReadOnly(True)
        self._edit_yaml.setPlaceholderText(self._t("ph_yaml"))
        yaml_row.addWidget(self._edit_yaml)
        self._btn_browse_yaml = QPushButton(self._t("browse_btn"))
        self._btn_browse_yaml.setFixedWidth(100)
        self._btn_browse_yaml.clicked.connect(self._on_browse_yaml)
        yaml_row.addWidget(self._btn_browse_yaml)
        self._lbl_row_yaml = QLabel(self._t("lbl_yaml"))
        form_layout.addRow(self._lbl_row_yaml, yaml_row)

        # Nombre del modelo
        self._edit_model_name = QLineEdit()
        self._edit_model_name.setPlaceholderText(self._t("ph_model_name"))
        self._edit_model_name.setText("rata_model")
        self._edit_model_name.textChanged.connect(self._update_train_btn)
        self._lbl_row_model = QLabel(self._t("lbl_model_name"))
        form_layout.addRow(self._lbl_row_model, self._edit_model_name)

        # Ruta de salida del modelo (solo lectura + boton abrir carpeta)
        out_row = QHBoxLayout()
        self._edit_output_path = QLineEdit()
        self._edit_output_path.setReadOnly(True)
        self._edit_output_path.setStyleSheet("color: #555; font-size: 11px;")
        out_row.addWidget(self._edit_output_path)
        self._btn_open_folder = QPushButton("Abrir carpeta")
        self._btn_open_folder.setFixedWidth(110)
        self._btn_open_folder.clicked.connect(self._on_open_output_folder)
        out_row.addWidget(self._btn_open_folder)
        self._lbl_row_output = QLabel("Guardará en:")
        form_layout.addRow(self._lbl_row_output, out_row)

        root.addWidget(self._form_group)

        # Fila Train + Cancelar
        btn_row = QHBoxLayout()
        self._btn_train = QPushButton(self._t("train_btn"))
        self._btn_train.setEnabled(False)
        self._btn_train.setFixedHeight(40)
        self._btn_train.setStyleSheet(
            "QPushButton { background-color: #CB0017; color: white; font-weight: bold; "
            "font-size: 14px; border-radius: 6px; } "
            "QPushButton:disabled { background-color: #aaa; } "
            "QPushButton:hover:!disabled { background-color: #a80013; }"
        )
        self._btn_train.clicked.connect(self._on_train)
        btn_row.addWidget(self._btn_train)

        self._btn_cancel = QPushButton("Cancelar")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.setFixedHeight(40)
        self._btn_cancel.setFixedWidth(110)
        self._btn_cancel.setStyleSheet(
            "QPushButton { background-color: #555; color: white; font-weight: bold; "
            "font-size: 13px; border-radius: 6px; } "
            "QPushButton:disabled { background-color: #ccc; color: #999; } "
            "QPushButton:hover:!disabled { background-color: #333; }"
        )
        self._btn_cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._btn_cancel)
        root.addLayout(btn_row)

        # Panel de metricas (oculto hasta que termina el entrenamiento)
        self._grp_metrics = QGroupBox(self._t("metrics_grp"))
        self._grp_metrics.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
        self._grp_metrics.setVisible(False)
        metrics_form = QFormLayout(self._grp_metrics)
        metrics_form.setSpacing(6)
        metrics_form.setContentsMargins(16, 12, 16, 12)

        def _metric_lbl() -> QLabel:
            l = QLabel("—")
            l.setStyleSheet("font-size: 13px; font-weight: bold; color: #1a5276;")
            return l

        self._lbl_m_precision = _metric_lbl()
        self._lbl_m_recall    = _metric_lbl()
        self._lbl_m_map50     = _metric_lbl()
        self._lbl_m_map5095   = _metric_lbl()
        self._lbl_m_prec_row  = QLabel(self._t("m_precision"))
        self._lbl_m_rec_row   = QLabel(self._t("m_recall"))
        self._lbl_m_map50_row = QLabel(self._t("m_map50"))
        self._lbl_m_map5095_row = QLabel(self._t("m_map5095"))
        metrics_form.addRow(self._lbl_m_prec_row,    self._lbl_m_precision)
        metrics_form.addRow(self._lbl_m_rec_row,     self._lbl_m_recall)
        metrics_form.addRow(self._lbl_m_map50_row,   self._lbl_m_map50)
        metrics_form.addRow(self._lbl_m_map5095_row, self._lbl_m_map5095)
        self._lbl_m_note = QLabel(self._t("m_note"))
        self._lbl_m_note.setStyleSheet("font-size: 10px; color: #888;")
        metrics_form.addRow(self._lbl_m_note)
        root.addWidget(self._grp_metrics)

        self._lbl_console = QLabel(self._t("console_lbl"))
        self._lbl_console.setStyleSheet("font-size: 11px; font-weight: bold; color: #333;")
        root.addWidget(self._lbl_console)

        self._console = QTextEdit()
        self._console.setReadOnly(True)
        font = QFont("Consolas", 9)
        self._console.setFont(font)
        self._console.setStyleSheet(
            "background-color: #1e1e1e; color: #d4d4d4; border: 1px solid #444; border-radius: 4px;"
        )
        self._console.setMinimumHeight(200)
        root.addWidget(self._console, stretch=1)

    # ------------------------------------------------------------------
    # Evento showEvent — comprueba dataset al entrar a la pagina
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._check_dataset_auto()

    # ------------------------------------------------------------------
    # Logica interna
    # ------------------------------------------------------------------

    def _check_dataset_auto(self) -> None:
        """Comprueba si existe el dataset y rellena los campos automaticamente."""
        train_images = paths.root / "datasets" / "unified" / "train" / "images"
        if train_images.exists() and any(train_images.iterdir()):
            self._lbl_dataset_status.setText(self._t("ds_found"))
            self._lbl_dataset_status.setStyleSheet(
                "font-size: 12px; padding: 6px 10px; border-radius: 4px; "
                "background-color: #e6f9ee; color: #1a7a3a; border: 1px solid #a8d8b8;"
            )
            self._lbl_dataset_status.setVisible(True)

            # Auto-rellenar carpeta si no tiene ya una
            if self._dataset_dir is None:
                datasets_root = paths.root / "datasets"
                self._dataset_dir = datasets_root
                self._edit_dataset.setText(str(datasets_root))

            # Auto-buscar yaml
            if self._yaml_path is None:
                self._find_yaml_auto()
        else:
            self._lbl_dataset_status.setText(self._t("ds_missing"))
            self._lbl_dataset_status.setStyleSheet(
                "font-size: 12px; padding: 6px 10px; border-radius: 4px; "
                "background-color: #fff8e1; color: #a06000; border: 1px solid #f0c040;"
            )
            self._lbl_dataset_status.setVisible(True)

        self._update_train_btn()

    def _find_yaml_auto(self) -> None:
        """Busca data.yaml en datasets/ automaticamente."""
        datasets_root = paths.root / "datasets"
        candidates = list(datasets_root.glob("*.yaml"))
        if not candidates:
            candidates = list(datasets_root.rglob("*.yaml"))
        if candidates:
            self._yaml_path = candidates[0]
            self._edit_yaml.setText(str(self._yaml_path))
        elif paths.data_yaml.exists():
            self._yaml_path = paths.data_yaml
            self._edit_yaml.setText(str(self._yaml_path))

    def _update_train_btn(self) -> None:
        has_dataset = self._dataset_dir is not None
        has_yaml    = self._yaml_path is not None
        has_name    = bool(self._edit_model_name.text().strip())

        _ok  = "font-size: 12px; color: #333;"
        _err = "font-size: 12px; color: #CB0017; font-weight: bold;"
        self._lbl_row_dataset.setStyleSheet(_err if not has_dataset else _ok)
        self._lbl_row_yaml.setStyleSheet(   _err if not has_yaml    else _ok)
        self._lbl_row_model.setStyleSheet(  _err if not has_name    else _ok)

        name = self._edit_model_name.text().strip() or "rata_model"
        dest = paths.models_dir / f"{name}.pt"
        self._edit_output_path.setText(str(dest))

        training = self._worker is not None and self._worker.isRunning()
        ok = has_dataset and has_yaml and has_name
        self._btn_train.setEnabled(ok and not training)
        self._btn_cancel.setEnabled(training)

    # ------------------------------------------------------------------
    # Slots de botones
    # ------------------------------------------------------------------

    def _on_browse_dataset(self) -> None:
        default = str(self._dataset_dir or Path.home())
        folder = QFileDialog.getExistingDirectory(self, self._t("dlg_dataset"), default)
        if folder:
            self._dataset_dir = Path(folder)
            self._edit_dataset.setText(folder)
            # Buscar yaml dentro de la carpeta seleccionada — siempre actualiza
            yamls = list(self._dataset_dir.glob("*.yaml"))
            if yamls:
                self._yaml_path = yamls[0]
                self._edit_yaml.setText(str(self._yaml_path))
            self._update_train_btn()

    def _on_browse_yaml(self) -> None:
        default = str(self._dataset_dir or Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, self._t("dlg_yaml"), default, "YAML Files (*.yaml *.yml);;All Files (*.*)"
        )
        if path:
            self._yaml_path = Path(path)
            self._edit_yaml.setText(path)
            self._update_train_btn()

    def _on_cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(3000)
            self._console.append("[!] Entrenamiento cancelado por el usuario.")
            self._btn_train.setText(self._t("train_btn"))
            self._update_train_btn()

    def _on_open_output_folder(self) -> None:
        folder = paths.models_dir
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))

    def _on_train(self) -> None:
        if self._worker and self._worker.isRunning():
            return

        # Guarda explícita: no continuar si falta algún campo
        missing = []
        if self._dataset_dir is None:
            missing.append("• Carpeta del dataset")
        if self._yaml_path is None or not self._yaml_path.exists():
            missing.append("• Archivo data.yaml")
        if not self._edit_model_name.text().strip():
            missing.append("• Nombre del modelo")
        if missing:
            QMessageBox.warning(
                self,
                "Faltan campos obligatorios",
                "No se puede entrenar sin completar:\n" + "\n".join(missing),
            )
            self._update_train_btn()
            return

        model_name = self._edit_model_name.text().strip()

        self._console.clear()
        self._console.append("=" * 55)
        self._console.append(f"  Dataset:   {self._dataset_dir}")
        self._console.append(f"  YAML:      {self._yaml_path}")
        self._console.append(f"  Modelo:    {paths.models_dir / (model_name + '.pt')}")
        self._console.append("=" * 55)
        self._btn_train.setEnabled(False)
        self._btn_train.setText(self._t("training_btn"))
        self._btn_cancel.setEnabled(True)
        self._grp_metrics.setVisible(False)

        self._worker = TrainWorker(self._yaml_path, model_name, parent=self)
        self._worker.log_msg.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_log(self, msg: str) -> None:
        self._console.append(msg)
        sb = self._console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_finished(self, model_path: str, metrics: dict) -> None:
        self._btn_train.setText(self._t("train_btn"))
        self._btn_cancel.setEnabled(False)
        self._update_train_btn()

        def _get(*keys) -> str:
            for k in keys:
                if k in metrics:
                    return f"{metrics[k]:.3f}"
            return "—"

        self._lbl_m_precision.setText(_get("metrics/precision(P)", "metrics/precision(B)"))
        self._lbl_m_recall.setText(   _get("metrics/recall(P)",    "metrics/recall(B)"))
        self._lbl_m_map50.setText(    _get("metrics/mAP50(P)",     "metrics/mAP50(B)"))
        self._lbl_m_map5095.setText(  _get("metrics/mAP50-95(P)",  "metrics/mAP50-95(B)"))
        self._grp_metrics.setVisible(True)

        QMessageBox.information(
            self,
            self._t("done_title"),
            f"{self._t('done_msg')}\n{model_path}",
        )

    def _on_error(self, msg: str) -> None:
        self._btn_train.setText(self._t("train_btn"))
        self._btn_cancel.setEnabled(False)
        self._update_train_btn()
        self._console.append(f"[ERROR] {msg}")
        QMessageBox.critical(self, self._t("error_title"), msg)

    def apply_language(self, lang: str) -> None:
        self._lang = lang
        self._lbl_title.setText(self._t("title"))
        self._lbl_subtitle.setText(self._t("subtitle"))
        self._form_group.setTitle(self._t("form_group"))
        self._lbl_row_dataset.setText(self._t("lbl_dataset"))
        self._lbl_row_yaml.setText(self._t("lbl_yaml"))
        self._lbl_row_model.setText(self._t("lbl_model_name"))
        self._edit_dataset.setPlaceholderText(self._t("ph_dataset"))
        self._edit_yaml.setPlaceholderText(self._t("ph_yaml"))
        self._edit_model_name.setPlaceholderText(self._t("ph_model_name"))
        self._btn_browse_ds.setText(self._t("browse_btn"))
        self._btn_browse_yaml.setText(self._t("browse_btn"))
        self._lbl_console.setText(self._t("console_lbl"))
        self._grp_metrics.setTitle(self._t("metrics_grp"))
        self._lbl_m_prec_row.setText(self._t("m_precision"))
        self._lbl_m_rec_row.setText(self._t("m_recall"))
        self._lbl_m_map50_row.setText(self._t("m_map50"))
        self._lbl_m_map5095_row.setText(self._t("m_map5095"))
        self._lbl_m_note.setText(self._t("m_note"))
        if not (self._worker and self._worker.isRunning()):
            self._btn_train.setText(self._t("train_btn"))
        self._check_dataset_auto()
