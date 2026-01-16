from pydantic import BaseModel
from pathlib import Path
import torch


class ProjectPaths(BaseModel):
    """Gestor de rutas del proyecto."""
    # ---------------------------------------------------------
    # CORRECCIÓN DE RUTA RAÍZ
    # config.py está en: rat_detector_project / scripts / helpers
    # parents[0] = helpers
    # parents[1] = scripts
    # parents[2] = rat_detector_project (LA RAÍZ REAL)
    # ---------------------------------------------------------
    root: Path = Path(__file__).resolve().parents[2]

    # --- RUTAS DE MODELO (Main 1) ---
    # Ahora 'root' es rat_detector_project, así que buscará en las carpetas correctas
    data_yaml: Path = root / "data" / "rats" / "data.yaml"
    video_source: Path = root / "videos" / "testRata1.mp4"
    coords_json: Path = root / "config" / "coords.json"
    models_dir: Path = root / "models"
    output_video: Path = root / "output" / "analizado.mp4"

    # --- RUTAS DE DATASET (Main 2) ---
    raw_videos: Path = root / "videos" / "raw"
    raw_images: Path = root / "videos" / "imagenesRata"
    temp_pool: Path = root / "videos" / "TEMP_POOL"
    final_dataset: Path = root / "videos" / "DataSet_Full"

    def check_dirs(self) -> None:
        """Crea directorios necesarios si no existen."""
        # Solo creará las carpetas si realmente faltan en la raíz
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.output_video.parent.mkdir(parents=True, exist_ok=True)
        self.coords_json.parent.mkdir(parents=True, exist_ok=True)

        # Directorios dataset
        self.raw_videos.mkdir(parents=True, exist_ok=True)
        self.temp_pool.mkdir(parents=True, exist_ok=True)
        self.final_dataset.mkdir(parents=True, exist_ok=True)


class TrainParams(BaseModel):
    base_model: str = "yolov8s.pt"
    epochs: int = 50
    imgsz: int = 640
    batch_size: int = -1
    device: str = "0" if torch.cuda.is_available() else "cpu"
    augment: bool = True


class DetectParams(BaseModel):
    # Aseguramos que busque el nombre correcto del modelo sin la 's' extra
    model_name: str = "yolov8_ratas_best.pt"
    conf_threshold: float = 0.5
    device: str = "0" if torch.cuda.is_available() else "cpu"


class DatasetParams(BaseModel):
    fps_extract: int = 5
    split_ratio: float = 0.8
    base_name: str = "rata"


# Instancias globales
paths = ProjectPaths()
train_cfg = TrainParams()
detect_cfg = DetectParams()
data_cfg = DatasetParams()