"""
Quantization context manager.

This module provides a thread-local context for managing quantization strategies,
avoiding the need to pass quantization parameters through multiple layers.
"""

from typing import Dict, Optional
from threading import local

from diffulex.utils.quantization.strategy import (
    QuantizationStrategy,
    KVCacheQuantizationStrategy,
    AttnQQuantizationStrategy,
    WeightQuantizationStrategy,
)


class QuantizationContext:
    """
    Quantization context manager.
    
    Uses thread-local storage to support multi-threaded/multi-process environments.
    Quantization strategies are registered and retrieved through context, avoiding parameter passing.
    """
    
    _thread_local = local()
    
    def __init__(self):
        self._strategies: Dict[str, QuantizationStrategy] = {}
    
    @classmethod
    def current(cls) -> 'QuantizationContext':
        """Get current thread's quantization context."""
        if not hasattr(cls._thread_local, 'context'):
            cls._thread_local.context = QuantizationContext()
        return cls._thread_local.context
    
    def set_strategy(self, key: str, strategy: QuantizationStrategy):
        """Set a quantization strategy."""
        self._strategies[key] = strategy
    
    def get_strategy(self, key: str, default: Optional[QuantizationStrategy] = None) -> Optional[QuantizationStrategy]:
        """Get a quantization strategy."""
        return self._strategies.get(key, default)
    
    def get_kv_cache_strategy(self) -> Optional[KVCacheQuantizationStrategy]:
        """Get KV Cache quantization strategy."""
        strategy = self._strategies.get('kv_cache')
        if strategy is None:
            return None
        if isinstance(strategy, KVCacheQuantizationStrategy):
            return strategy
        raise TypeError(
            f"KV cache strategy must be KVCacheQuantizationStrategy, got {type(strategy)}"
        )
    
    def get_weight_strategy(self) -> Optional[WeightQuantizationStrategy]:
        """Get weight quantization strategy."""
        strategy = self._strategies.get('weight')
        if strategy is None:
            return None
        if isinstance(strategy, WeightQuantizationStrategy):
            return strategy
        raise TypeError(
            f"Weight strategy must be WeightQuantizationStrategy, got {type(strategy)}"
        )

    def get_attn_q_strategy(self) -> Optional[AttnQQuantizationStrategy]:
        """Get Attention-Q quantization strategy."""
        strategy = self._strategies.get('attn_q')
        if strategy is None:
            return None
        if isinstance(strategy, AttnQQuantizationStrategy):
            return strategy
        raise TypeError(
            f"attn_q strategy must be AttnQQuantizationStrategy, got {type(strategy)}"
        )
    
    def clear(self):
        """Clear all strategies."""
        self._strategies.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Optionally clear context on exit, or keep it for reuse
        pass


# Global convenience functions
def get_quantization_context() -> QuantizationContext:
    """Get current quantization context."""
    return QuantizationContext.current()


def set_kv_cache_strategy(strategy: KVCacheQuantizationStrategy):
    """Set KV Cache quantization strategy."""
    ctx = QuantizationContext.current()
    ctx.set_strategy('kv_cache', strategy)


def get_kv_cache_strategy() -> Optional[KVCacheQuantizationStrategy]:
    """Get KV Cache quantization strategy."""
    ctx = QuantizationContext.current()
    return ctx.get_kv_cache_strategy()


def set_weight_strategy(strategy: WeightQuantizationStrategy):
    """Set weight quantization strategy."""
    ctx = QuantizationContext.current()
    ctx.set_strategy('weight', strategy)


def get_weight_strategy() -> Optional[WeightQuantizationStrategy]:
    """Get weight quantization strategy."""
    ctx = QuantizationContext.current()
    return ctx.get_weight_strategy()


def set_attn_q_strategy(strategy: AttnQQuantizationStrategy):
    """Set Attention-Q quantization strategy."""
    ctx = QuantizationContext.current()
    ctx.set_strategy('attn_q', strategy)


def get_attn_q_strategy() -> Optional[AttnQQuantizationStrategy]:
    """Get Attention-Q quantization strategy."""
    ctx = QuantizationContext.current()
    return ctx.get_attn_q_strategy()

