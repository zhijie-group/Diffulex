"""Diffulex model package that imports built-in models to trigger registration."""
from __future__ import annotations

# Import built-in models so their registrations run at import time.
from . import dream  # noqa: F401
from . import llada  # noqa: F401
from . import fast_dllm_v2  # noqa: F401

__all__ = ["dream", "llada", "fast_dllm_v2"]

from .auto_model import AutoModelForDiffusionLM