"""
Quantization strategy implementations.
"""

from diffulex.utils.quantization.strategies.no_quantization import NoQuantizationStrategy
from diffulex.utils.quantization.strategies.kv_cache_bf16 import KVCacheBF16Strategy
from diffulex.utils.quantization.strategies.kv_cache_fp8_running_max import KVCacheFP8RunningMaxStrategy
from diffulex.utils.quantization.strategies.linear_bf16 import LinearBF16Strategy
from diffulex.utils.quantization.strategies.linear_stub import LinearStubStrategy
from diffulex.utils.quantization.strategies.linear_int8_w8a16 import LinearInt8W8A16Strategy  # noqa: F401
from diffulex.utils.quantization.strategies.linear_marlin_int8_w8a16 import LinearMarlinInt8W8A16Strategy  # noqa: F401
from diffulex.utils.quantization.strategies.linear_int4_w4a16 import LinearInt4W4A16Strategy  # noqa: F401
from diffulex.utils.quantization.strategies.linear_int8_w8a8 import LinearInt8W8A8Strategy  # noqa: F401
from diffulex.utils.quantization.strategies.linear_int4_w4a8 import LinearInt4W4A8Strategy  # noqa: F401
from diffulex.utils.quantization.strategies.linear_fp8_w8a16 import LinearFP8W8A16Strategy  # noqa: F401
from diffulex.utils.quantization.strategies.linear_fp8_w8a8 import LinearFP8W8A8Strategy  # noqa: F401
from diffulex.utils.quantization.strategies.linear_gptq_w4a16 import LinearGPTQW4A16Strategy  # noqa: F401
from diffulex.utils.quantization.strategies.linear_awq_w4a16 import LinearAWQW4A16Strategy  # noqa: F401

__all__ = [
    'NoQuantizationStrategy',
    'KVCacheBF16Strategy',
    'KVCacheFP8RunningMaxStrategy',
    'LinearBF16Strategy',
    'LinearStubStrategy',
    'LinearInt8W8A16Strategy',
    'LinearMarlinInt8W8A16Strategy',
    'LinearInt4W4A16Strategy',
    'LinearInt8W8A8Strategy',
    'LinearInt4W4A8Strategy',
    'LinearFP8W8A16Strategy',
    'LinearFP8W8A8Strategy',
    'LinearGPTQW4A16Strategy',
    'LinearAWQW4A16Strategy',
]

