"""Deja los modulos de python/ importables desde los tests."""

import sys
from pathlib import Path

_PYTHON_DIR = Path(__file__).resolve().parent.parent / "python"

if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
