"""Diffulex package root.

Keep this module lightweight so that importing submodules like
`diffulex.utils.quantization` does not eagerly import the full engine/kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # These are available for type checkers; runtime import is lazy via __getattr__.
    from diffulex.diffulex import Diffulex as Diffulex  # noqa: F401
    from diffulex.sampling_params import SamplingParams as SamplingParams  # noqa: F401


def __getattr__(name: str):
    if name == "Diffulex":
        # Only trigger heavy side-effect imports when users actually construct the engine.
        # This keeps `import diffulex.utils.quantization` lightweight.
        from diffulex import strategy as _strategy  # noqa: F401
        from diffulex.diffulex import Diffulex

        return Diffulex
    if name == "SamplingParams":
        from diffulex.sampling_params import SamplingParams

        return SamplingParams
    raise AttributeError(name)


__all__ = ["Diffulex", "SamplingParams"]
