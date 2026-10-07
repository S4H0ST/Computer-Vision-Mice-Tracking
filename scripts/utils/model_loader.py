"""
Utilidades para cargar modelos YOLO en distintos formatos de inferencia.

Functions:
    export_to_tensorrt   — exporta un modelo .pt a TensorRT (.engine) para GPU NVIDIA.
    load_model           — carga un modelo .pt y devuelve una instancia YOLO.
    pick_model_path      — devuelve la ruta al modelo .pt por defecto.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def export_to_tensorrt(model_path: Path) -> Path:
    model = YOLO(str(model_path))
    export_path = model.export(format="engine")
    return Path(export_path)


def pick_model_path() -> Path:
    """Devuelve la ruta al modelo .pt por defecto."""
    from app_config.config import paths

    if paths.yolo_model.exists():
        return paths.yolo_model
    raise FileNotFoundError("No se encontro ningun modelo YOLO en models/gpu/")


def load_model(model_path: Path) -> YOLO:
    return YOLO(str(model_path))
