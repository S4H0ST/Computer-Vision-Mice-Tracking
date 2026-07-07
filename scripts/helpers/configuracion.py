import sys
from pathlib import Path
from dataclasses import dataclass

# --- 1. DEFINICIÓN DE LA RAÍZ DEL PROYECTO ---
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent.parent


@dataclass
class Paths:
    root: Path = PROJECT_ROOT

    # --- RUTAS DE CÓDIGO Y MODELOS ---
    scripts: Path = root / "scripts"
    models_dir: Path = root / "models"

    # NOMBRES DE ARCHIVOS CLAVE
    base_yolo_model: Path = models_dir / "yolov8s.pt"

    # Definimos dónde vivirá el modelo final de las ratas
    yolo_model: Path = models_dir / "yolo_ratas.pt"

    # El cerebro RNN
    rnn_model: Path = models_dir / "best_rnn.pth"

    # --- RUTAS DE DATOS ---
    data_dir: Path = root / "data"
    output_dir: Path = data_dir / "output"
    data_yaml: Path = root / "datasets" / "data.yaml"
    coords_json: Path = root / "datasets" / "coords.json"

    # --- RUTAS DE VIDEO ---
    video_dir: Path = root / "media" / "videos"

    # VIDEO DE PRUEBA (Asegúrate de que este nombre es correcto)
    video_source: Path = video_dir / "testRata1.mp4"

    # Salida automática
    output_video: Path = output_dir / "resultado_final.mp4"

    # --- RUTAS PARA EXTRACCIÓN DE DATASET ---
    raw_videos: Path = root / "media_original" / "videos"
    raw_images: Path = root / "media_original" / "frames_original"
    temp_pool: Path = root / "media_original" / "TEMP_POOL"
    final_dataset: Path = root / "datasets" / "DataSet_Full"
    dataset_valid: Path = root / "datasets" / "valid"

    def check_dirs(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.video_source.exists():
            print(f"[ADVERTENCIA] No encuentro el video en: {self.video_source}")


paths = Paths()


# --- PARÁMETROS DE ENTRENAMIENTO ---
@dataclass
class TrainParams:
    epochs: int = 50
    imgsz: int = 640
    batch_size: int = -1
    device: str = "0"
    # Usamos la ruta del modelo base para evitar descargas
    base_model: str = str(paths.base_yolo_model)
    augment: bool = False


# --- PARÁMETROS DE DETECCIÓN ---
@dataclass
class DetectParams:
    conf_threshold: float = 0.4
    iou_threshold: float = 0.5
    # Ahora usamos la variable correcta que acabamos de crear arriba
    model_path: Path = paths.yolo_model
    device: str = "0"


# --- PARÁMETROS DE EXTRACCIÓN DE DATASET ---
@dataclass
class DatasetParams:
    split_ratio: float = 0.8    # 80% train, 20% valid
    base_name: str = "rat"


train_cfg = TrainParams()
detect_cfg = DetectParams()
data_cfg = DatasetParams()