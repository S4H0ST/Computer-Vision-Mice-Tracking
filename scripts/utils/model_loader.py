"""
Utilidades para exportar y cargar modelos YOLO en distintos formatos de inferencia.

Functions:
    export_to_onnx       — exporta un modelo .pt a ONNX (con simplificacion opcional).
    export_to_tensorrt   — exporta un modelo .pt a TensorRT (.engine) para GPU NVIDIA.
    load_model           — carga un modelo .pt o .onnx y devuelve una instancia YOLO.
    pick_model_path      — elige el mejor formato segun hardware disponible.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def export_to_onnx(model_path: Path, simplify: bool = True) -> Path:
    """
    Exporta un modelo .pt a formato ONNX.

    Args:
        model_path: Ruta al archivo .pt de Ultralytics.
        simplify:   Si True, aplica onnx-simplifier para reducir el grafo.

    Returns:
        Ruta al archivo .onnx generado (mismo directorio que model_path).
    """
    model = YOLO(str(model_path))
    export_path = model.export(format="onnx", simplify=simplify)
    return Path(export_path)


def export_to_tensorrt(model_path: Path) -> Path:
    """
    Exporta un modelo .pt a TensorRT (.engine) para inferencia en GPU NVIDIA.

    Requiere CUDA y TensorRT instalados. El archivo resultante solo es
    compatible con la GPU y el driver en los que se genero.

    Args:
        model_path: Ruta al archivo .pt de Ultralytics.

    Returns:
        Ruta al archivo .engine generado (mismo directorio que model_path).
    """
    model = YOLO(str(model_path))
    export_path = model.export(format="engine")
    return Path(export_path)


def pick_model_path() -> Path:
    """
    Devuelve la ruta al mejor modelo disponible segun el hardware:
      - GPU (CUDA): prefiere .pt (PyTorch + CUDA, sin problemas de provider)
      - CPU:        prefiere .onnx si existe (ONNX Runtime CPU es mas rapido)

    onnxruntime-gpu tiene un bug de tensor binding cuando se combina con
    CUDAExecutionProvider en esta version; usar .pt en GPU lo evita.
    """
    from app_config.config import paths

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False

    if cuda_ok:
        if paths.yolo_model.exists():
            return paths.yolo_model
    else:
        if paths.yolo_model_onnx.exists():
            return paths.yolo_model_onnx

    # fallback universal
    if paths.yolo_model.exists():
        return paths.yolo_model
    if paths.yolo_model_onnx.exists():
        return paths.yolo_model_onnx
    raise FileNotFoundError("No se encontro ningun modelo YOLO en models/")


def load_model(model_path: Path) -> YOLO:
    """
    Carga un modelo YOLO desde un archivo .pt o .onnx.

    Para ONNX con GPU: pasar device='0' al llamar a model.predict() activa
    CUDAExecutionProvider automaticamente si onnxruntime-gpu esta instalado.

    Args:
        model_path: Ruta al archivo .pt o .onnx.

    Returns:
        Instancia YOLO lista para inferencia.
    """
    return YOLO(str(model_path))
