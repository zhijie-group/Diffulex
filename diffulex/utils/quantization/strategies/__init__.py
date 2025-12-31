"""
Quantization strategy implementations.
"""

from diffulex.utils.quantization.strategies.no_quantization import NoQuantizationStrategy
from diffulex.utils.quantization.strategies.kv_cache_bf16 import KVCacheBF16Strategy
from diffulex.utils.quantization.strategies.kv_cache_fp8_running_max import KVCacheFP8RunningMaxStrategy
from diffulex.utils.quantization.strategies.attn_q_bf16 import AttnQBF16Strategy
from diffulex.utils.quantization.strategies.attn_q_fp8_stub import AttnQFP8StubStrategy
from diffulex.utils.quantization.strategies.linear_bf16 import LinearBF16Strategy
from diffulex.utils.quantization.strategies.linear_stub import LinearStubStrategy
from diffulex.utils.quantization.strategies.linear_int8_w8a16 import LinearInt8W8A16Strategy  # noqa: F401
from diffulex.utils.quantization.strategies.linear_int4_w4a16 import LinearInt4W4A16Strategy  # noqa: F401

__all__ = [
    'NoQuantizationStrategy',
    'KVCacheBF16Strategy',
    'KVCacheFP8RunningMaxStrategy',
    'AttnQBF16Strategy',
    'AttnQFP8StubStrategy',
    'LinearBF16Strategy',
    'LinearStubStrategy',
    'LinearInt8W8A16Strategy',
    'LinearInt4W4A16Strategy',
]

