import shutil
import random
import os
from pathlib import Path
from typing import List, Tuple
from helpers.interfaces import BaseModule
from helpers.config import paths, DatasetParams

class DatasetBuilder(BaseModule):
    """La 'Aspiradora': Junta, renombra, baraja y divide (Train/Valid)."""

    def __init__(self, config: DatasetParams):
        self.cfg = config

    def run(self) -> None:
        print("\n🏗️ CONSTRUYENDO DATASET YOLO...")

        # 1. Limpiar temporales
        if paths.temp_pool.exists(): shutil.rmtree(paths.temp_pool)
        paths.temp_pool.mkdir()
        (paths.temp_pool / "images").mkdir()
        (paths.temp_pool / "labels").mkdir()

        # 2. Recopilar (Aspirar)
        print(f"[1/3] Recopilando desde: {paths.raw_images}")
        files = self._gather_files()
        if not files: return

        # 3. Organizar Final
        print(f"[2/3] Organizando en: {paths.final_dataset}")
        self._create_yolo_structure(files)

        # 4. Limpieza final
        shutil.rmtree(paths.temp_pool)
        print("✅ Dataset generado exitosamente.")

    def _gather_files(self) -> List[Tuple[Path, Path]]:
        pairs = []
        for root, _, files in os.walk(paths.raw_images):
            # Evitar carpetas de salida
            if "DataSet_Full" in root or "TEMP_POOL" in root: continue

            for f in files:
                if f.lower().endswith(('.jpg', '.png')):
                    img_path = Path(root) / f
                    txt_path = img_path.with_suffix(".txt")

                    # Si no está al lado, buscar en ../labels/
                    if not txt_path.exists():
                        txt_path = Path(root).parent / "labels" / txt_path.name

                    if txt_path.exists():
                        # Renombrar para evitar duplicados: FolderName_FileName
                        prefix = Path(root).name
                        new_name = f"{prefix}_{img_path.stem}"

                        dst_img = paths.temp_pool / "images" / f"{new_name}{img_path.suffix}"
                        dst_txt = paths.temp_pool / "labels" / f"{new_name}.txt"

                        shutil.copy(img_path, dst_img)
                        shutil.copy(txt_path, dst_txt)
                        pairs.append((dst_img, dst_txt))

        print(f"   -> Encontrados {len(pairs)} pares (Imagen + Etiqueta).")
        return pairs

    def _create_yolo_structure(self, pairs: List[Tuple[Path, Path]]) -> None:
        random.shuffle(pairs)
        n_train = int(len(pairs) * self.cfg.split_ratio)

        for split in ['train', 'valid']:
            (paths.final_dataset / split / 'images').mkdir(parents=True, exist_ok=True)
            (paths.final_dataset / split / 'labels').mkdir(parents=True, exist_ok=True)

        for i, (src_img, src_txt) in enumerate(pairs):
            split = 'train' if i < n_train else 'valid'
            final_name = f"{self.cfg.base_name}_{i:05d}"

            shutil.copy(src_img, paths.final_dataset / split / 'images' / f"{final_name}{src_img.suffix}")
            shutil.copy(src_txt, paths.final_dataset / split / 'labels' / f"{final_name}.txt")