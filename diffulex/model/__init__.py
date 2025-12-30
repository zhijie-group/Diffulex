"""Diffulex model package that imports built-in models to trigger registration."""
from __future__ import annotations
import importlib
from pathlib import Path

# Import built-in models so their registrations run at import time.
# Automatically import all Python files except auto_model and __init__
_excluded_modules = {"auto_model", "__init__"}
_model_modules = []

_current_dir = Path(__file__).parent
for py_file in _current_dir.glob("*.py"):
    module_name = py_file.stem
    if module_name not in _excluded_modules:
        try:
            importlib.import_module(f".{module_name}", __name__)
            _model_modules.append(module_name)
        except Exception as e:
            # Skip modules that fail to import
            import warnings
            # ImportWarning is ignored by default, which can hide real registration problems.
            warnings.warn(f"Failed to import {module_name}: {e!r}", RuntimeWarning)

__all__ = _model_modules.copy()

from .auto_model import AutoModelForDiffusionLM