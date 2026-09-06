"""
Configuracion centralizada del proyecto: rutas, parametros de entrenamiento y deteccion.

Clases:
    Paths        — rutas de archivos y directorios del proyecto.
    TrainParams  — hiperparametros para el entrenamiento YOLO.
    DetectParams — parametros para la inferencia/deteccion.
    DatasetParams — parametros para la construccion del dataset.

Variables de modulo:
    paths      : Paths
    train_cfg  : TrainParams
    detect_cfg : DetectParams
    data_cfg   : DatasetParams
"""

import sys
from pathlib import Path
from dataclasses import dataclass

try:
    import torch
    _DEFAULT_DEVICE: str = "0" if torch.cuda.is_available() else "cpu"
except ImportError:
    _DEFAULT_DEVICE: str = "cpu"

# Raiz del proyecto: scripts/config/config.py -> scripts/config -> scripts -> raiz
FILE_PATH: Path = Path(__file__).resolve()
PROJECT_ROOT: Path = FILE_PATH.parent.parent.parent


@dataclass
class Paths:
    root: Path = PROJECT_ROOT

    # Rutas de modelos
    models_dir: Path = root / "models"

    base_yolo_model: Path = models_dir / "yolov8s-pose.pt"
    yolo_model:      Path = models_dir / "yolo_ratas.pt"
    rnn_model:       Path = models_dir / "best_rnn.pth"

    # Rutas de salida
    output_dir:  Path = root / "outputs"
    detect_dir:  Path = output_dir / "detections"
    reports_dir: Path = output_dir / "reports"

    data_yaml:   Path = root / "datasets" / "data.yaml"
    coords_json: Path = root / "outputs" / "calibration" / "coords.json"

    # Rutas de video
    video_dir:    Path = root / "media_original" / "videos"
    video_source: Path = video_dir / "testRata5.mp4"
    output_video: Path = detect_dir / "testRata5_resultado.mp4"

    def check_dirs(self) -> None:
        """Crea los directorios esenciales y avisa si falta el video fuente."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.detect_dir.mkdir(parents=True, exist_ok=True)

        if not self.video_source.exists():
            print(f"[ADVERTENCIA] No encuentro el video en: {self.video_source}")


paths: Paths = Paths()


@dataclass
class TrainParams:
    epochs: int = 100
    imgsz: int = 640
    batch_size: int = -1
    device: str = _DEFAULT_DEVICE
    base_model: str = str(paths.base_yolo_model)


@dataclass
class DetectParams:
    conf_threshold: float = 0.18
    iou_threshold: float = 0.5
    model_path: Path = paths.yolo_model
    device: str = _DEFAULT_DEVICE


@dataclass
class DatasetParams:
    split_ratio: float = 0.8   # 80% train, 20% valid
    base_name: str = "rat"


train_cfg: TrainParams = TrainParams()
detect_cfg: DetectParams = DetectParams()
data_cfg: DatasetParams = DatasetParams()
