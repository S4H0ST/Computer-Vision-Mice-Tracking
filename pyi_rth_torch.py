"""
Runtime hook — ejecutado antes de run_gui.py en el ejecutable congelado.
Inicializa torch y escribe startup_diag.log junto al ejecutable.

Modo normal  : comprueba imports y archivos (rapido, <2 s).
Modo --diag  : ademas carga el modelo YOLO y hace inferencia de prueba.
"""
from __future__ import annotations

import sys
import os
import traceback
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# GUARD: abortar si somos un subprocess invocado por ultralytics/torch/pip
#
# En el exe congelado sys.executable == RatTracker.exe.
# Ultralytics puede llamar:
#   subprocess.run([sys.executable, "-m", "pip", "show", "ultralytics"])
# Eso relanza el exe con "-m" en argv[1].  freeze_support() solo atrapa
# "--multiprocessing-fork", asi que sin este guard abre una ventana nueva.
# --------------------------------------------------------------------------- #
if getattr(sys, "frozen", False) and len(sys.argv) > 1:
    _a1 = sys.argv[1]
    _known = {"--diag", "--multiprocessing-fork"}
    if _a1 not in _known:
        # Log para diagnostico y salir sin abrir GUI
        try:
            _log = Path(sys.executable).parent / "blocked_spawn.log"
            with open(str(_log), "a", encoding="utf-8") as _f:
                _f.write(f"[{datetime.now():%H:%M:%S}] PID={os.getpid()} argv={sys.argv}\n")
        except Exception:
            pass
        sys.exit(0)

# Evitar que ultralytics llame a uv/pip para verificar/instalar dependencias
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

# --------------------------------------------------------------------------- #
# FIX: PyTorch 2.7+ llama a inspect.getsource() en tiempo de import para
# parsear comentarios "compile_ignored" en torch.fx.experimental._config.
# En un bundle congelado (PyInstaller) no hay codigo fuente disponible y
# getsource() lanza OSError, abortando la carga de torch.__init__.
# Parcheamos getsource() para devolver "" en lugar de lanzar, de modo que
# install_config_module() obtenga un conjunto vacio de compile_ignored_keys
# — correcto para uso de inferencia donde no se usa torch.compile().
# --------------------------------------------------------------------------- #
if getattr(sys, "frozen", False):
    import inspect as _inspect

    def _safe_getsource(obj, _orig=_inspect.getsource):
        try:
            return _orig(obj)
        except OSError:
            return ""

    _inspect.getsource = _safe_getsource

# --------------------------------------------------------------------------- #
# Helpers de diagnóstico
# --------------------------------------------------------------------------- #

_lines: list[str] = []
_ok_n  = 0
_err_n = 0
_DIAG_MODE = "--diag" in sys.argv


def _add(s: str = "") -> None:
    _lines.append(s)


def _chk(label: str, fn) -> object:
    global _ok_n, _err_n
    try:
        result = fn()
        _add(f"  [OK]    {label}: {result}")
        _ok_n += 1
        return result
    except Exception as exc:
        _add(f"  [ERROR] {label}: {type(exc).__name__}: {exc}")
        _err_n += 1
        return None


def _file(label: str, *paths: Path) -> bool:
    global _ok_n, _err_n
    for p in paths:
        if p.exists():
            try:
                size = f"  ({p.stat().st_size / 1e6:.1f} MB)" if p.stat().st_size > 1e5 else ""
            except Exception:
                size = ""
            _add(f"  [OK]    {label}{size}")
            _ok_n += 1
            return True
    _add(f"  [MISS]  {label}")
    _err_n += 1
    return False


# --------------------------------------------------------------------------- #
# Encabezado
# --------------------------------------------------------------------------- #

_exe_dir = Path(sys.executable).parent
_meipass  = Path(getattr(sys, "_MEIPASS", str(_exe_dir / "_internal")))

_add(f"=== RatTracker Diagnostic Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
_add(f"  Python   : {sys.version.split()[0]}  ({sys.version})")
_add(f"  Platform : {sys.platform}")
_add(f"  EXE dir  : {_exe_dir}")
_add(f"  MEIPASS  : {_meipass}")
_add(f"  Diag mode: {'YES (--diag)' if _DIAG_MODE else 'NO'}")
_add()

# --------------------------------------------------------------------------- #
# Imports de paquetes Python
# --------------------------------------------------------------------------- #

_add("--- Python packages ---")

# shm.dll se incluye en el build CPU (torch_python.dll tiene dependencia hard en ella).
# En yolorat_cpu, shm.dll no tiene dependencias CUDA y carga sin problemas.
# Este patch es una red de seguridad por si shm.dll fallara en alguna maquina.
if getattr(sys, "frozen", False):
    import ctypes as _ctypes
    _orig_cdll_init = _ctypes.CDLL.__init__

    def _cdll_init_tolerant(self, name=None, *args, **kwargs):
        if isinstance(name, str) and "shm.dll" in name.lower():
            try:
                _orig_cdll_init(self, name, *args, **kwargs)
            except OSError:
                return
        else:
            _orig_cdll_init(self, name, *args, **kwargs)

    _ctypes.CDLL.__init__ = _cdll_init_tolerant

import torch  # noqa: E402 — también inicializa torch en el bundle (como el hook original)

_chk("torch",         lambda: torch.__version__)
_chk("torch CUDA",    lambda: (
    f"available={torch.cuda.is_available()}"
    + (f"  device={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")
))
_chk("cv2",           lambda: __import__("cv2").__version__)
_chk("numpy",         lambda: __import__("numpy").__version__)
_chk("ultralytics",   lambda: __import__("ultralytics").__version__)
_chk("PIL",           lambda: __import__("PIL").__version__)
_chk("scipy",         lambda: __import__("scipy").__version__)
_chk("pandas",        lambda: __import__("pandas").__version__)
_chk("openpyxl",      lambda: __import__("openpyxl").__version__)
_chk("PyQt5.QtCore",  lambda: __import__("PyQt5.QtCore", fromlist=["PYQT_VERSION_STR"]).PYQT_VERSION_STR)
_chk("jaraco.text",   lambda: "present" if __import__("jaraco.text") else "?")
_chk("pkg_resources", lambda: (
    getattr(__import__("pkg_resources"), "__version__", None) or "present"
))
_chk("app_config",    lambda: "present" if __import__("app_config") else "?")

_add()

# --------------------------------------------------------------------------- #
# Archivos de modelo
# --------------------------------------------------------------------------- #

_add("--- Model files ---")
_model_pt = _exe_dir / "models" / "gpu" / "yolo_ratas.pt"

_file("models/gpu/yolo_ratas.pt", _model_pt)

# Muestra qué modelo elegiría pick_model_path() — diagnóstico para el caso
# en que el video se muestra pero YOLO no detecta nada (modelo no encontrado).
try:
    from utils.model_loader import pick_model_path as _pick
    _picked = _pick()
    if _picked is not None:
        _add(f"  [OK]    pick_model_path() -> {_picked}")
        _ok_n += 1
    else:
        _add("  [ERROR] pick_model_path() -> None  (no model file found — detection will not work)")
        _err_n += 1
except Exception as _e:
    _add(f"  [WARN]  pick_model_path() check failed: {_e}")
_add()

# --------------------------------------------------------------------------- #
# Archivos de datos empaquetados
# --------------------------------------------------------------------------- #

_add("--- Data files ---")
_DATA: list[tuple[str, list[Path]]] = [
    ("datasets/data.yaml",                     [_exe_dir / "datasets" / "data.yaml"]),
    ("scripts/app_config/translations.json",   [_meipass / "scripts" / "app_config" / "translations.json",
                                                 _exe_dir / "scripts" / "app_config" / "translations.json"]),
    ("scripts/app_config/labels.json",         [_meipass / "scripts" / "app_config" / "labels.json",
                                                 _exe_dir / "scripts" / "app_config" / "labels.json"]),
    ("gui/main_window.ui",                     [_meipass / "gui" / "main_window.ui",
                                                 _exe_dir / "gui" / "main_window.ui"]),
    ("gui/assets/icons/app_icon.ico",          [_meipass / "gui" / "assets" / "icons" / "app_icon.ico"]),
    ("ultralytics/cfg/trackers/bytetrack.yaml",[_meipass / "ultralytics" / "cfg" / "trackers" / "bytetrack.yaml"]),
]
for _lbl, _paths in _DATA:
    _file(_lbl, *_paths)

_add()

# --------------------------------------------------------------------------- #
# Test de carga de modelo YOLO (solo con --diag)
# --------------------------------------------------------------------------- #

if _DIAG_MODE:
    _add("--- YOLO model load test (--diag) ---")
    _best = _model_pt if _model_pt.exists() else None
    if _best is None:
        _add("  [SKIP]  No model file found")
        _err_n += 1
    else:
        try:
            import numpy as np
            from ultralytics import YOLO as _YOLO
            _m = _YOLO(str(_best))
            _add(f"  [OK]    Loaded {_best.name}")
            _add(f"          task={_m.task}  classes={list(_m.names.values())}")
            _device = "0" if torch.cuda.is_available() else "cpu"
            _dummy  = np.zeros((320, 320, 3), dtype=np.uint8)
            _res    = _m.predict(_dummy, verbose=False, device=_device)
            _add(f"  [OK]    Inference on 320x320 blank frame: {len(_res)} result(s)")
            _ok_n += 1
        except Exception as _exc:
            _add(f"  [ERROR] {type(_exc).__name__}: {_exc}")
            _add(traceback.format_exc())
            _err_n += 1
    _add()

# --------------------------------------------------------------------------- #
# Resumen y escritura del log
# --------------------------------------------------------------------------- #

_add(f"--- Summary: {_ok_n} OK / {_err_n} problems ---")

_log_path = _exe_dir / "startup_diag.log"

# Solo escribir el log si hay problemas o si se pidio explicitamente con --diag.
# En arranque normal sin errores no se genera ningun archivo.
if _err_n or _DIAG_MODE:
    try:
        _log_path.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    except Exception:
        pass

# En modo --diag mostrar cuadro de dialogo con el resumen
if _DIAG_MODE:
    try:
        import ctypes
        icon = 0x30 if _err_n else 0x40   # MB_ICONWARNING / MB_ICONINFORMATION
        ctypes.windll.user32.MessageBoxW(
            0,
            f"Resultado: {_ok_n} OK / {_err_n} problemas\n\nLog guardado en:\n{_log_path}",
            "RatTracker — Diagnostico",
            icon,
        )
    except Exception:
        pass
