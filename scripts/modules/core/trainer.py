import torch
from ultralytics import YOLO
from helpers.interfaces import BaseModule
from helpers.config import paths, TrainParams


class YOLOTrainer(BaseModule):
    def __init__(self, config: TrainParams):
        self.cfg = config
        self.model = YOLO(self.cfg.base_model)

    def _estimate_batch(self) -> int:
        if self.cfg.device == "cpu": return 1
        try:
            # Estimación simple basada en memoria VRAM disponible
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_mem >= 8: return 8
            if gpu_mem >= 4: return 4
            return 2
        except:
            return 2

    def run(self) -> None:
        print(f"[>] Iniciando entrenamiento YOLO en {self.cfg.device}...")

        batch_size = self.cfg.batch_size
        if batch_size == -1:
            batch_size = self._estimate_batch()

        # Entrenar
        self.model.train(
            data=str(paths.data_yaml),
            epochs=self.cfg.epochs,
            imgsz=self.cfg.imgsz,
            batch=batch_size,
            device=self.cfg.device,
            project=str(paths.root / "runs" / "train"),
            name="rat_experiment",
            exist_ok=True,
            augment=self.cfg.augment
        )
        print("[+] Entrenamiento YOLO completado.")