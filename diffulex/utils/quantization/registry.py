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
    AttnQQuantizationStrategy,
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


# ---- Attention-Q (activation) registry ----
AttnQStrategyBuilder = Callable[[], AttnQQuantizationStrategy]
_ATTN_Q_BUILDERS: Dict[str, AttnQStrategyBuilder] = {}


def register_attn_q_strategy(*dtype_aliases: str) -> Callable[[AttnQStrategyBuilder], AttnQStrategyBuilder]:
    """Register an Attention-Q strategy builder for one or more dtype aliases."""

    def _decorator(builder: AttnQStrategyBuilder) -> AttnQStrategyBuilder:
        for alias in dtype_aliases:
            key = (alias or "").strip().lower()
            _ATTN_Q_BUILDERS[key] = builder
        return builder

    return _decorator


def create_attn_q_strategy(attn_q_dtype: str) -> AttnQQuantizationStrategy:
    key = (attn_q_dtype or "").strip().lower() or "bf16"
    builder = _ATTN_Q_BUILDERS.get(key)
    if builder is None:
        raise ValueError(
            f"Unsupported attn_q_dtype={attn_q_dtype!r} (normalized={key!r}). "
            f"Registered: {sorted(_ATTN_Q_BUILDERS.keys())}"
        )
    return builder()


def registered_attn_q_dtypes() -> list[str]:
    return sorted(_ATTN_Q_BUILDERS.keys())


