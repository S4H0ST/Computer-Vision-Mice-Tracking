import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    _MEIPASS = Path(sys._MEIPASS)
    sys.path.insert(0, str(_MEIPASS / "scripts"))
    sys.path.insert(0, str(_MEIPASS))
else:
    _ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(_ROOT / "scripts"))
    sys.path.insert(0, str(_ROOT))

from gui.app import main
main()
