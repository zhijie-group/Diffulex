"""
Quantization module for diffulex.

This module provides a flexible, extensible quantization architecture that supports:
- KV Cache quantization
- Future: Weight quantization, Activation quantization, etc.

The architecture uses strategy pattern with context management to minimize coupling.
"""

from diffulex.utils.quantization.context import (
    QuantizationContext,
    get_quantization_context,
    set_kv_cache_strategy,
    get_kv_cache_strategy,
    set_attn_q_strategy,
    get_attn_q_strategy,
)
from diffulex.utils.quantization.factory import QuantizationStrategyFactory
from diffulex.utils.quantization.config import (
    QuantizationConfig,
    KVCacheQuantConfig,
    WeightQuantConfig,
    ActivationQuantConfig,
)
from diffulex.utils.quantization.registry import (
    create_kv_cache_strategy,
    registered_kv_cache_dtypes,
    create_attn_q_strategy,
    registered_attn_q_dtypes,
)
from diffulex.utils.quantization.strategy import (
    QuantizationStrategy,
    KVCacheQuantizationStrategy,
    AttnQQuantizationStrategy,
    WeightQuantizationStrategy,
)
# Re-export kv_cache_dtype utilities for backward compatibility
from diffulex.utils.quantization.kv_cache_dtype import (
    KvCacheDType,
    KvCacheDTypeSpec,
    parse_kv_cache_dtype,
    ensure_scale_tensor,
    view_fp8_cache,
)

__all__ = [
    # Context
    'QuantizationContext',
    'get_quantization_context',
    'set_kv_cache_strategy',
    'get_kv_cache_strategy',
    'set_attn_q_strategy',
    'get_attn_q_strategy',
    # Factory
    'QuantizationStrategyFactory',
    # Config
    'QuantizationConfig',
    'KVCacheQuantConfig',
    'WeightQuantConfig',
    'ActivationQuantConfig',
    # Registry
    'create_kv_cache_strategy',
    'registered_kv_cache_dtypes',
    'create_attn_q_strategy',
    'registered_attn_q_dtypes',
    # Strategy interfaces
    'QuantizationStrategy',
    'KVCacheQuantizationStrategy',
    'AttnQQuantizationStrategy',
    'WeightQuantizationStrategy',
    # KV Cache dtype utilities (for backward compatibility)
    'KvCacheDType',
    'KvCacheDTypeSpec',
    'parse_kv_cache_dtype',
    'ensure_scale_tensor',
    'view_fp8_cache',
]

