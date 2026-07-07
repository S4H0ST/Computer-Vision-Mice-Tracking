# scripts/evaluation/evaluate_sharpness_gap.py
"""
Evalúa el modelo YOLO-Pose entrenado por separado en el subconjunto de
validación nítido vs el borroso, y reporta el gap de mAP50 entre ambos.

Requisitos previos:
  - Un modelo YOLO ya entrenado en paths.yolo_model.
  - Un dataset de validación poblado en paths.dataset_valid (images/ + labels/).

Uso:
    cd scripts
    python evaluation/evaluate_sharpness_gap.py
"""

import sys
from pathlib import Path

# Aseguramos que 'scripts/' esté en el path aunque se ejecute desde otra carpeta
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from helpers.configuracion import paths
from dataset_tools.sharpness_splitter import SharpnessSplitter


def main():
    if not paths.yolo_model.exists():
        print(f"[X] No encuentro el modelo YOLO en: {paths.yolo_model}")
        print("    -> Entrena primero (Opción 2 del menú principal).")
        return

    if not (paths.dataset_valid / "images").exists():
        print(f"[X] No encuentro imágenes de validación en: {paths.dataset_valid}")
        return

    model = YOLO(str(paths.yolo_model))
    splitter = SharpnessSplitter(paths.dataset_valid, threshold=100.0)

    output_root = paths.data_dir / "sharpness_eval"
    sharp_dir, blurry_dir = splitter.split(output_root)

    yaml_sharp = splitter.make_temp_yaml(paths.data_yaml, sharp_dir, output_root / "sharp.yaml")
    yaml_blurry = splitter.make_temp_yaml(paths.data_yaml, blurry_dir, output_root / "blurry.yaml")

    print("\n=== EVALUACIÓN: SOLO IMÁGENES NÍTIDAS ===")
    metrics_sharp = model.val(data=str(yaml_sharp))
    print(f"mAP50 (nítidas): {metrics_sharp.pose.map50:.4f}")

    print("\n=== EVALUACIÓN: SOLO IMÁGENES BORROSAS ===")
    metrics_blurry = model.val(data=str(yaml_blurry))
    print(f"mAP50 (borrosas): {metrics_blurry.pose.map50:.4f}")

    gap = metrics_sharp.pose.map50 - metrics_blurry.pose.map50
    print(f"\n[GAP] Diferencia de rendimiento: {gap:.4f}")
    if gap > 0.05:
        print("[i] El gap es notable — el motion blur está afectando la precisión.")
        print("    Considera reforzar el bucket 'borrosas' en tu próxima ronda de etiquetado.")
    else:
        print("[i] El gap es pequeño — el modelo generaliza razonablemente bien al blur.")


if __name__ == "__main__":
    main()