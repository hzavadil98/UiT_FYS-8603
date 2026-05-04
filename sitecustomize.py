"""Project-local Python startup hooks.

When this repository is executed directly, Python will import this module if
it is present on sys.path. We use it to ensure the vendored python_packages
directory is importable so both the notebook and standalone scripts can resolve
the bundled jive package.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_PACKAGES_DIR = PROJECT_ROOT / "python_packages"

if PYTHON_PACKAGES_DIR.exists():
    python_packages_path = str(PYTHON_PACKAGES_DIR)
    if python_packages_path not in sys.path:
        sys.path.insert(0, python_packages_path)
