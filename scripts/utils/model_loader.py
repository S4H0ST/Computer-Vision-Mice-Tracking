"""
Utilidades para exportar y cargar modelos YOLO en distintos formatos de inferencia.

Logica de seleccion de modelo:
  Por defecto se prefiere .onnx (optimo para CPU, que es lo mas habitual en usuarios finales).
  Si se detecta GPU CUDA y el modelo activo es .onnx, la GUI puede ofrecer al usuario
  cambiar a .pt para aprovechar la aceleracion hardware.

Functions:
    export_to_onnx       — exporta un modelo .pt a ONNX (con simplificacion opcional).
    export_to_tensorrt   — exporta un modelo .pt a TensorRT (.engine) para GPU NVIDIA.
    load_model           — carga un modelo .pt o .onnx y devuelve una instancia YOLO.
    pick_model_path      — devuelve la ruta al modelo por defecto (.onnx preferido).
    gpu_can_upgrade      — True si hay GPU y existe .pt, pero se esta usando .onnx.
    needs_cpu_pt_warning — True si es CPU pero solo hay .pt (sin .onnx disponible).
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def export_to_onnx(model_path: Path, simplify: bool = True) -> Path:
    model = YOLO(str(model_path))
    export_path = model.export(format="onnx", simplify=simplify)
    return Path(export_path)


def export_to_tensorrt(model_path: Path) -> Path:
    model = YOLO(str(model_path))
    export_path = model.export(format="engine")
    return Path(export_path)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def pick_model_path() -> Path:
    """
    Devuelve la ruta al modelo por defecto.

    Politica: preferir .onnx siempre que exista, independientemente del hardware.
    Razon: el ejecutable distribuido esta pensado para usuarios sin GPU; .onnx es
    mas rapido en CPU y no tiene el bug de tensor binding de onnxruntime-gpu.
    Si el usuario tiene GPU, la GUI le ofrecera cambiar a .pt mediante un dialogo.

    Orden de preferencia:
      1. yolo_ratas.onnx  (si existe)
      2. yolo_ratas.pt    (fallback)
    """
    from app_config.config import paths

    if paths.yolo_model_onnx.exists():
        return paths.yolo_model_onnx
    if paths.yolo_model.exists():
        return paths.yolo_model
    raise FileNotFoundError("No se encontro ningun modelo YOLO en models/")


def gpu_can_upgrade(current_path: Path) -> bool:
    """
    True cuando el usuario podria mejorar el rendimiento cambiando a .pt con GPU:
      - Hay GPU CUDA disponible
      - El modelo activo elegido es .onnx
      - Existe tambien el .pt

    En ese caso la GUI debe ofrecer la opcion de cambiar a .pt.
    """
    from app_config.config import paths

    if current_path.suffix.lower() != ".onnx":
        return False
    if not paths.yolo_model.exists():
        return False
    return _cuda_available()


def needs_cpu_pt_warning() -> bool:
    """
    True cuando el usuario va a correr inferencia con .pt en CPU sin alternativa:
      - No hay GPU CUDA disponible
      - No existe .onnx
      - Existe .pt (es el unico modelo disponible)
    """
    from app_config.config import paths

    if _cuda_available():
        return False
    return paths.yolo_model.exists() and not paths.yolo_model_onnx.exists()


def load_model(model_path: Path) -> YOLO:
    return YOLO(str(model_path))
