import torch
import shutil
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, Any
from helpers.interfaces import BaseModule
from helpers.config import paths, TrainParams

class YOLOTrainer(BaseModule):
    def __init__(self, config: TrainParams):
        self.cfg = config
        self.model = YOLO(self.cfg.base_model)

    def _estimate_batch(self) -> int:
        if self.cfg.device == "cpu": return 1
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_mem >= 12: return 8
            if gpu_mem >= 8: return 6
            if gpu_mem >= 6: return 4
            return 2
        except:
            return 2

    def _get_augmentation_args(self) -> Dict[str, float]:
        return {
            "degrees": 180.0, "translate": 0.2, "scale": 0.4,
            "shear": 0.0, "flipud": 0.5, "fliplr": 0.5,
            "hsv_h": 0.015, "hsv_s": 0.2, "hsv_v": 0.2,
            "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0
        }

    def run(self) -> None:
        print(f"[>] Entrenando en {self.cfg.device}...")
        batch_size = self.cfg.batch_size
        if batch_size == -1:
            batch_size = self._estimate_batch()
            print(f"[i] Batch automático: {batch_size}")

        args = {
            "data": str(paths.data_yaml),
            "epochs": self.cfg.epochs,
            "imgsz": self.cfg.imgsz,
            "batch": batch_size,
            "device": self.cfg.device,
            "project": str(paths.root / "runs" / "train"),
            "name": "exp",
            "exist_ok": True,
            "augment": self.cfg.augment
        }

        if self.cfg.augment:
            args.update(self._get_augmentation_args())

        self.model.train(**args)
        self._save_best_model()

    def _save_best_model(self):
        src = paths.root / "runs" / "train" / "exp" / "weights" / "best.pt"
        dst = paths.models_dir / "yolov8_ratas_best.pt"
        if src.exists():
            shutil.copy(src, dst)
            print(f"[OK] Modelo guardado en: {dst}")
        else:
            print("[!] No se encontró best.pt")