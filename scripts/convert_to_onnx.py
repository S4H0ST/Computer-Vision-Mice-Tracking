"""
Convierte un modelo YOLO .pt a formato ONNX.

Uso:
    python convert_to_onnx.py                          # convierte models/yolo_ratas.pt
    python convert_to_onnx.py models/yolo_ratas_v1.pt  # ruta personalizada
    python convert_to_onnx.py --no-simplify             # sin onnx-simplifier
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Exportar modelo YOLO a ONNX")
    parser.add_argument(
        "model_path",
        nargs="?",
        default=str(_ROOT / "models" / "yolo_ratas.pt"),
        help="Ruta al archivo .pt (por defecto: models/yolo_ratas.pt)",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Desactivar onnx-simplifier",
    )
    args = parser.parse_args()

    pt_path = Path(args.model_path)
    if not pt_path.exists():
        print(f"[ERROR] No se encontro el modelo: {pt_path}")
        sys.exit(1)

    simplify = not args.no_simplify
    print(f"Modelo   : {pt_path}")
    print(f"Simplify : {simplify}")
    print("Exportando a ONNX...")

    t0 = time.time()

    from utils.model_loader import export_to_onnx
    onnx_path = export_to_onnx(pt_path, simplify=simplify)

    elapsed = time.time() - t0
    print(f"ONNX guardado en: {onnx_path}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
