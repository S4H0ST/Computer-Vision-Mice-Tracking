"""
Utilidades para cargar modelos YOLO.

Functions:
    pick_model_path — devuelve la ruta al modelo .pt por defecto.
    load_model      — carga un modelo .pt y devuelve una instancia YOLO.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def pick_model_path() -> Path:
    """Devuelve la ruta al modelo .pt por defecto."""
    from app_config.config import paths

    if paths.yolo_model.exists():
        return paths.yolo_model
    raise FileNotFoundError("No se encontro ningun modelo YOLO en models/gpu/")


def load_model(model_path: Path) -> YOLO:
    return YOLO(str(model_path))
