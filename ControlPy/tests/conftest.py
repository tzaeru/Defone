# Ensure ControlPy directory is on the path when running tests (e.g. pytest ControlPy/tests/ from repo root)
import sys
from pathlib import Path

_controlpy_root = Path(__file__).resolve().parent.parent
if str(_controlpy_root) not in sys.path:
    sys.path.insert(0, str(_controlpy_root))
