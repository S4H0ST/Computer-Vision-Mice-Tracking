# scripts/dataset_tools/sharpness_splitter.py
"""
Separa un set de validación YOLO-pose en nítido/borroso según la varianza
del Laplaciano, para poder medir el mAP50 por separado en cada subconjunto.

Esto es útil para la memoria del TFG: te permite argumentar cuantitativamente
"el modelo pierde X puntos de mAP50 en frames borrosos" en vez de una
afirmación cualitativa.

Uso típico: ver scripts/evaluation/evaluate_sharpness_gap.py
"""

import cv2
import shutil
import yaml
from pathlib import Path


class SharpnessSplitter:
    """Separa un set de validación YOLO-pose en nítido/borroso según Laplaciano."""

    def __init__(self, valid_dir: Path, threshold: float = 100.0):
        self.valid_dir = Path(valid_dir)
        self.threshold = threshold  # ajustar según tus pruebas

    def _laplacian_variance(self, img_path: Path) -> float:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return cv2.Laplacian(img, cv2.CV_64F).var()

    def split(self, output_root: Path):
        sharp_dir = output_root / "valid_sharp"
        blurry_dir = output_root / "valid_blurry"
        for d in (sharp_dir, blurry_dir):
            (d / "images").mkdir(parents=True, exist_ok=True)
            (d / "labels").mkdir(parents=True, exist_ok=True)

        images = list((self.valid_dir / "images").glob("*.*"))
        counts = {"sharp": 0, "blurry": 0}

        for img_path in images:
            var = self._laplacian_variance(img_path)
            bucket = "sharp" if var >= self.threshold else "blurry"
            target = sharp_dir if bucket == "sharp" else blurry_dir
            counts[bucket] += 1

            label_path = self.valid_dir / "labels" / f"{img_path.stem}.txt"
            shutil.copy(img_path, target / "images" / img_path.name)
            if label_path.exists():
                shutil.copy(label_path, target / "labels" / f"{img_path.stem}.txt")

        print(f"[OK] Nítidas: {counts['sharp']} | Borrosas: {counts['blurry']}")
        return sharp_dir, blurry_dir

    def make_temp_yaml(self, base_yaml: Path, valid_path: Path, out_yaml: Path):
        """Clona tu data.yaml pero apuntando val a la subcarpeta filtrada."""
        with open(base_yaml) as f:
            data = yaml.safe_load(f)
        data["val"] = str((valid_path / "images").resolve())
        with open(out_yaml, "w") as f:
            yaml.dump(data, f)
        return out_yaml