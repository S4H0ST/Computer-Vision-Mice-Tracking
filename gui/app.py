"""
Punto de entrada de la interfaz grafica.
Ejecutar desde la raiz del proyecto:  python -m gui.app
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from PyQt5.QtWidgets import QApplication
from gui.controllers.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
