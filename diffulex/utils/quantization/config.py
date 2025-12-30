"""
Quantization configuration objects for Diffulex.

Diffulex currently exposes a single user-facing knob: `Config.kv_cache_dtype`.
This module introduces explicit config dataclasses so we can extend to
weights/activations quantization without growing ad-hoc fields everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVCacheQuantConfig:
    """KV-cache quantization configuration."""

    dtype: str = "bf16"
    # Future extension points:
    # - scale_mode: "running_max" | "static" | ...
    # - per_channel: bool


@dataclass(frozen=True)
class WeightQuantConfig:
    """Weight quantization configuration (placeholder)."""

    method: str = "none"


@dataclass(frozen=True)
class ActivationQuantConfig:
    """Activation quantization configuration (placeholder)."""

    # Currently used to control attention-Q quantization.
    # "bf16" (default) | "fp8" (placeholder; requires future kernel)
    attn_q_dtype: str = "bf16"


@dataclass(frozen=True)
class QuantizationConfig:
    """Top-level quantization configuration for Diffulex."""

    kv_cache: KVCacheQuantConfig = KVCacheQuantConfig()
    weights: WeightQuantConfig = WeightQuantConfig()
    activations: ActivationQuantConfig = ActivationQuantConfig()

    @classmethod
    def from_diffulex_config(cls, config) -> "QuantizationConfig":
        # Keep this tolerant: Diffulex's Config is a simple dataclass and may evolve.
        kv_cache_dtype = getattr(config, "kv_cache_dtype", "bf16") or "bf16"
        attn_q_dtype = getattr(config, "attn_q_dtype", "bf16") or "bf16"
        return cls(
            kv_cache=KVCacheQuantConfig(dtype=kv_cache_dtype),
            activations=ActivationQuantConfig(attn_q_dtype=attn_q_dtype),
        )


