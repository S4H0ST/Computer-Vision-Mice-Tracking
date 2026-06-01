from abc import ABC, abstractmethod
from pathlib import Path


class BaseModule(ABC):
    """Clase abstracta base para todos los módulos."""

    @abstractmethod
    def run(self) -> None:
        """ Metodo principal de ejecución."""
        pass

    def validate_file(self, path: Path) -> bool:
        if not path.exists():
            print(f"[X] Error: Archivo no encontrado -> {path}")
            return False
        return True