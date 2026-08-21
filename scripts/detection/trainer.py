"""
Entrenamiento del modelo YOLO Pose para deteccion de comportamientos del raton.

Clases:
    YOLOTrainer — entrena el modelo YOLO, estima el batch optimo y copia automaticamente
                  el mejor peso al directorio models/.
"""

import torch
import shutil
import platform
from pathlib import Path
from ultralytics import YOLO
from config.interfaces import BaseModule
from config.config import paths, TrainParams


class YOLOTrainer(BaseModule):
    """Entrena un modelo YOLO Pose y copia el mejor checkpoint a models/yolo_ratas.pt."""

    def __init__(self, config: TrainParams) -> None:
        self.cfg: TrainParams = config
        self.model: YOLO = YOLO(self.cfg.base_model)

    def _estimate_batch(self) -> int:
        """Estima el batch size optimo segun la VRAM disponible en la GPU."""
        if self.cfg.device == "cpu":
            return 1
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_mem >= 8:
                return 8
            if gpu_mem >= 4:
                return 4
            return 2
        except Exception:
            return 2

    def run(self) -> None:
        """Lanza el entrenamiento y, al finalizar, copia best.pt a models/yolo_ratas.pt."""
        print(f"[Core] Iniciando entrenamiento YOLO en {self.cfg.device}...")

        batch_size = self.cfg.batch_size
        if batch_size == -1:
            batch_size = self._estimate_batch()

        # En Windows, DataLoader con workers > 0 puede dar problemas de multiprocessing
        workers = 0 if platform.system() == "Windows" else 8

        # Augmentation geometrico puro — apropiado para camara cenital fija:
        #   degrees=180  -> rotaciones +-180 (cualquier orientacion es valida)
        #   flipud/fliplr -> espejos validos en vista top-down
        #   translate=0.05, scale=0.4 -> variaciones pequenas (rata siempre en caja)
        #   mosaic=0.0   -> desactivado (mezcla fondos irreales)
        #   hsv_*=0.0    -> sin cambios de color (iluminacion controlada, blanco/negro)
        #   erasing=0.0  -> sin borrado sintetico (fondo negro es parte del dominio)
        self.model.train(
            data=str(paths.data_yaml),
            epochs=self.cfg.epochs,
            imgsz=self.cfg.imgsz,
            batch=batch_size,
            device=self.cfg.device,
            workers=workers,
            project=str(paths.root / "runs" / "train"),
            name="exp",
            exist_ok=False,
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

        print("[+] Entrenamiento finalizado.")

        try:
            save_dir: Path = Path(self.model.trainer.save_dir)
            best_weight: Path = save_dir / "weights" / "best.pt"

            if best_weight.exists():
                print(f"[Auto] Encontrado mejor modelo en: {best_weight}")
                target_path: Path = paths.yolo_model
                shutil.copy(str(best_weight), str(target_path))

                print("=" * 60)
                print(f" [OK] MODELO COPIADO AUTOMATICAMENTE")
                print(f" -> Origen: {best_weight}")
                print(f" -> Destino: {target_path}")
                print(" -> Ya puedes ejecutar la Opcion 3 directamente.")
                print("=" * 60)
            else:
                print(f"[!] Error: No encuentro el archivo 'best.pt' en {save_dir}")

        except Exception as e:
            print(f"[X] Fallo la copia automatica: {e}")
            print("    Tendras que copiar 'best.pt' manualmente a 'models/yolo_ratas.pt'")
