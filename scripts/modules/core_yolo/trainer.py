import torch
import shutil  # Librería para mover archivos
from pathlib import Path
from ultralytics import YOLO
from helpers.base import BaseModule
from helpers.configuracion import paths, TrainParams


class YOLOTrainer(BaseModule):
    def __init__(self, config: TrainParams):
        self.cfg = config
        # Cargar modelo base
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

        # 1. ENTRENAR
        self.model.train(
            data=str(paths.data_yaml),
            epochs=self.cfg.epochs,
            imgsz=self.cfg.imgsz,
            batch=batch_size,
            device=self.cfg.device,
            project=str(paths.root / "runs" / "train"),
            name="exp",
            exist_ok=False,  # Crea exp, exp2, exp3... para no borrar historial
            augment=self.cfg.augment
        )

        print("[+] Entrenamiento finalizado.")

        # 2. AUTOMATIZACIÓN: COPIAR EL MODELO
        try:
            # Buscamos dónde guardó YOLO el resultado
            save_dir = Path(self.model.trainer.save_dir)
            best_weight = save_dir / "weights" / "best.pt"

            if best_weight.exists():
                print(f"[Auto] Encontrado mejor modelo en: {best_weight}")

                # Destino: models/yolo_ratas.pt
                target_path = paths.yolo_model

                # Copiar archivo
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