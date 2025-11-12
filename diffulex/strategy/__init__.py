"""Diffulex strategy package that imports built-in strategies to trigger registration."""
from __future__ import annotations

# Import built-in strategies so their registrations run at import time.
from . import d2f  # noqa: F401

__all__ = ["d2f"]
