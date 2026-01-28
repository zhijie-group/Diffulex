"""
Quantization strategy factory.

This module provides factory functions to create quantization strategies from configuration.
"""

from typing import Optional

from diffulex.utils.quantization.context import QuantizationContext
from diffulex.utils.quantization.config import QuantizationConfig
from diffulex.utils.quantization.registry import create_kv_cache_strategy as _create_kv_cache_strategy
from diffulex.utils.quantization.registry import create_linear_strategy as _create_linear_strategy
from diffulex.utils.quantization.strategy import KVCacheQuantizationStrategy

# Ensure built-in strategies are imported so they can register themselves.
from diffulex.utils.quantization import strategies as _builtin_strategies  # noqa: F401


class QuantizationStrategyFactory:
    """Quantization strategy factory."""
    
    @staticmethod
    def create_kv_cache_strategy(dtype: Optional[str] = None) -> KVCacheQuantizationStrategy:
        """
        Create KV Cache quantization strategy.
        
        Args:
            dtype: KV cache dtype string:
                - None or "bf16": BF16 (no quantization)
                - "fp16": FP16 (no quantization, future support)
                - "fp32": FP32 (no quantization, future support)
                - "fp8" or "fp8_e4m3": FP8 E4M3 with running max
                - "fp8_e5m2": FP8 E5M2 with running max
        
        Returns:
            KV Cache quantization strategy instance
        
        Raises:
            ValueError: If dtype is not supported
        """
        # NOTE: dtype normalization + compatibility handling lives in the registry.
        return _create_kv_cache_strategy(dtype or "bf16")
    
    @staticmethod
    def create_from_config(config) -> QuantizationContext:
        """
        Create and configure quantization context from config object.
        
        Args:
            config: Configuration object that may contain quantization-related fields:
                - kv_cache_dtype: KV cache dtype string
                - weight_dtype: Weight dtype string (future)
        
        Returns:
            Configured quantization context
        """
        ctx = QuantizationContext.current()
        
        quant_cfg = QuantizationConfig.from_diffulex_config(config)
        
        # KV Cache strategy
        strategy = QuantizationStrategyFactory.create_kv_cache_strategy(quant_cfg.kv_cache.dtype)
        strategy.configure(diffulex_config=config)
        ctx.set_strategy('kv_cache', strategy)

        # Linear strategies (weights + activations) by kind
        linear_attn = _create_linear_strategy(
            weight_dtype=quant_cfg.weights.linear_attn_dtype,
            act_dtype=quant_cfg.activations.linear_attn_dtype,
        )
        linear_attn.configure(diffulex_config=config)
        ctx.set_linear_strategy("attn", linear_attn)

        linear_mlp = _create_linear_strategy(
            weight_dtype=quant_cfg.weights.linear_mlp_dtype,
            act_dtype=quant_cfg.activations.linear_mlp_dtype,
        )
        linear_mlp.configure(diffulex_config=config)
        ctx.set_linear_strategy("mlp", linear_mlp)
        
        # Future: Weight strategy
        # weight_dtype = getattr(config, 'weight_dtype', None)
        # if weight_dtype:
        #     strategy = QuantizationStrategyFactory.create_weight_strategy(weight_dtype)
        #     ctx.set_strategy('weight', strategy)
        
        return ctx

