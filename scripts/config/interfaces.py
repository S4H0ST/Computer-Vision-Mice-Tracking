"""
Interfaces abstractas compartidas por todos los módulos del pipeline.

Clases:
    BaseModule — clase abstracta que obliga a implementar run() y ofrece validate_file().
"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseModule(ABC):
    """Clase abstracta base para todos los módulos del pipeline."""

    @abstractmethod
    def run(self) -> None:
        """Metodo principal de ejecucion del modulo."""
        pass

    def validate_file(self, path: Path) -> bool:
        """Comprueba que el archivo existe; imprime un error si no."""
        if not path.exists():
            print(f"[X] Error: Archivo no encontrado -> {path}")
            return False
        return True
