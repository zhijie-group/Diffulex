"""
Quantization registries for Diffulex.

This module provides lightweight registries similar in spirit to vLLM's
quantization directory: core runtime code should not hard-code concrete
strategy classes. Instead, it should dispatch by strategy capabilities
and let factories/registries decide which strategy to instantiate.
"""

from __future__ import annotations

from typing import Callable, Dict

from diffulex.utils.quantization.kv_cache_dtype import _normalize_kv_cache_dtype
from diffulex.utils.quantization.strategy import (
    KVCacheQuantizationStrategy,
    LinearQuantizationStrategy,
)

# A builder returns a fully constructed strategy instance.
KVCacheStrategyBuilder = Callable[[], KVCacheQuantizationStrategy]

_KV_CACHE_BUILDERS: Dict[str, KVCacheStrategyBuilder] = {}


def register_kv_cache_strategy(*dtype_aliases: str) -> Callable[[KVCacheStrategyBuilder], KVCacheStrategyBuilder]:
    """Register a KV-cache strategy builder for one or more dtype aliases."""

    def _decorator(builder: KVCacheStrategyBuilder) -> KVCacheStrategyBuilder:
        for alias in dtype_aliases:
            key = _normalize_kv_cache_dtype(alias)
            _KV_CACHE_BUILDERS[key] = builder
        return builder

    return _decorator


def create_kv_cache_strategy(kv_cache_dtype: str) -> KVCacheQuantizationStrategy:
    """Create a KV-cache quantization strategy from a dtype string."""
    key = _normalize_kv_cache_dtype(kv_cache_dtype)
    builder = _KV_CACHE_BUILDERS.get(key)
    if builder is None:
        raise ValueError(
            f"Unsupported kv_cache_dtype={kv_cache_dtype!r} (normalized={key!r}). "
            f"Registered: {sorted(_KV_CACHE_BUILDERS.keys())}"
        )
    return builder()


def registered_kv_cache_dtypes() -> list[str]:
    return sorted(_KV_CACHE_BUILDERS.keys())


# ---- Linear (weights + activations) registry ----
LinearStrategyBuilder = Callable[[], LinearQuantizationStrategy]
_LINEAR_BUILDERS: Dict[tuple[str, str], LinearStrategyBuilder] = {}


def _normalize_linear_dtype(dtype: str) -> str:
    """Normalize Linear quantization dtype/method strings.

    We intentionally keep this lightweight: the concrete semantics (weight-only,
    activation-only, etc.) live in the strategy implementations.
    """
    s = (dtype or "").strip().lower()
    # Reserved internal sentinel for generic fallback strategy registration.
    if s in {"__stub__", "__fallback__"}:
        return "__stub__"
    aliases = {
        "": "bf16",
        "none": "bf16",
        "bf16": "bf16",
        "bfloat16": "bf16",
        # Integer
        "int8": "int8",
        "i8": "int8",
        "int4": "int4",
        "i4": "int4",
        # FP8
        "fp8": "fp8_e4m3",
        "fp8_e4m3": "fp8_e4m3",
        "e4m3": "fp8_e4m3",
        "fp8_e5m2": "fp8_e5m2",
        "e5m2": "fp8_e5m2",
        # Weight-only methods (placeholders)
        "gptq": "gptq",
        "awq": "awq",
        "gptq_awq": "gptq_awq",
        # vLLM-style fused W8A16 path (Diffulex vendored): user-facing alias "marlin"
        # Normalized key is "marlin_int8" to avoid conflating with other quant methods.
        "marlin": "marlin_int8",
        "marlin_int8": "marlin_int8",
    }
    if s not in aliases:
        raise ValueError(
            f"Unsupported linear quant dtype={dtype!r}. "
            "Supported: bf16/int8/int4/fp8/fp8_e4m3/fp8_e5m2/gptq/awq/marlin"
        )
    return aliases[s]


def register_linear_strategy(
    *,
    weight_dtype: str,
    act_dtype: str,
) -> Callable[[LinearStrategyBuilder], LinearStrategyBuilder]:
    """Register a Linear strategy builder for a (weight_dtype, act_dtype) pair."""

    w = _normalize_linear_dtype(weight_dtype)
    a = _normalize_linear_dtype(act_dtype)

    def _decorator(builder: LinearStrategyBuilder) -> LinearStrategyBuilder:
        _LINEAR_BUILDERS[(w, a)] = builder
        return builder

    return _decorator


def create_linear_strategy(*, weight_dtype: str, act_dtype: str) -> LinearQuantizationStrategy:
    """Create a Linear quantization strategy from weight/activation dtype strings.

    If an exact pair is not registered, we fall back to:
    - bf16/bf16: a built-in BF16 strategy (registered by default)
    - otherwise: a generic stub strategy that raises NotImplementedError at runtime
      (registered by default).
    """
    w = _normalize_linear_dtype(weight_dtype)
    a = _normalize_linear_dtype(act_dtype)
    builder = _LINEAR_BUILDERS.get((w, a))
    if builder is not None:
        return builder()

    # Fall back to generic stub builder if present.
    stub = _LINEAR_BUILDERS.get(("__stub__", "__stub__"))
    if stub is None:
        raise ValueError(
            f"Unsupported linear strategy pair (weight_dtype={weight_dtype!r}, act_dtype={act_dtype!r}) "
            f"(normalized={(w, a)!r}). Registered pairs: {sorted(_LINEAR_BUILDERS.keys())}"
        )
    s = stub()
    # Attach requested formats for better error messages / future dispatch.
    try:
        setattr(s, "weight_dtype", w)
        setattr(s, "act_dtype", a)
    except Exception:
        pass
    return s


def registered_linear_dtypes() -> list[str]:
    """Return the normalized dtype/method names accepted by `_normalize_linear_dtype`."""
    # Keep this list stable for CLI/help messages.
    return ["bf16", "int8", "int4", "fp8_e4m3", "fp8_e5m2", "gptq", "awq", "gptq_awq", "marlin_int8"]


