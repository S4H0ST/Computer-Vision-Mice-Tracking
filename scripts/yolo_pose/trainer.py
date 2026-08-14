import torch
import shutil
import platform
from pathlib import Path
from ultralytics import YOLO
from config.interfaces import BaseModule
from config.config import paths, TrainParams


class YOLOTrainer(BaseModule):
    def __init__(self, config: TrainParams):
        self.cfg = config
        self.model = YOLO(self.cfg.base_model)

    def _estimate_batch(self) -> int:
        if self.cfg.device == "cpu": return 1
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_mem >= 8: return 8
            if gpu_mem >= 4: return 4
            return 2
        except:
            return 2

    def run(self) -> None:
        print(f"[Core] Iniciando entrenamiento YOLO en {self.cfg.device}...")

        batch_size = self.cfg.batch_size
        if batch_size == -1:
            batch_size = self._estimate_batch()

        workers = 0 if platform.system() == "Windows" else 8

        # 1. ENTRENAR
        # Augmentation geométrico puro — apropiado para cámara cenital fija:
        #   · degrees=180  → rotaciones ±180° (cualquier orientación es válida)
        #   · flipud/fliplr → espejos válidos en vista top-down
        #   · translate=0.05, scale=0.4 → variaciones pequeñas (rata siempre en caja)
        #   · mosaic=0.0   → desactivado (mezcla fondos irreales)
        #   · hsv_*=0.0    → sin cambios de color (iluminación controlada, blanco/negro)
        #   · erasing=0.0  → sin borrado sintético (el fondo negro es parte del dominio)
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
            # Regularización (dataset pequeño, ~176 imágenes)
            dropout=0.2,          # reduce overfitting en dataset pequeño
            weight_decay=0.0008,  # ligeramente más que el default (0.0005)
            cos_lr=True,          # cosine LR: convergencia más suave al final
            # Geometría (cámara cenital fija, todas las orientaciones son válidas)
            degrees=180,
            translate=0.05,
            scale=0.4,
            fliplr=0.5,
            flipud=0.5,
            # Sin augmentation de color/textura (iluminación controlada, fondo negro)
            mosaic=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            erasing=0.0,
        )

        print("[+] Entrenamiento finalizado.")

        # 2. AUTOMATIZACIÓN: COPIAR EL MODELO
        try:
            save_dir = Path(self.model.trainer.save_dir)
            best_weight = save_dir / "weights" / "best.pt"

            if best_weight.exists():
                print(f"[Auto] Encontrado mejor modelo en: {best_weight}")
                target_path = paths.yolo_model
                shutil.copy(str(best_weight), str(target_path))

                print("=" * 60)
                print(f" [EXITO] MODELO COPIADO AUTOMÁTICAMENTE")
                print(f" -> Origen: {best_weight}")
                print(f" -> Destino: {target_path}")
                print(" -> Ya puedes ejecutar la Opción 3 directamente.")
                print("=" * 60)
            else:
                print(f"[!] Error: No encuentro el archivo 'best.pt' en {save_dir}")

        except Exception as e:
            print(f"[X] Falló la copia automática: {e}")
            print("    Tendrás que copiar 'best.pt' manualmente a 'models/yolo_ratas.pt'")
