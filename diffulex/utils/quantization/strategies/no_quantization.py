"""
No quantization strategy (default, backward compatible).
"""

import torch
from diffulex.utils.quantization.strategy import QuantizationStrategy


class NoQuantizationStrategy(QuantizationStrategy):
    """No quantization strategy (default, backward compatible)."""
    
    @property
    def name(self) -> str:
        return "no_quantization"
    
    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        """Returns BF16 as default storage dtype."""
        return torch.bfloat16, 2
    
    def quantize(self, tensor: torch.Tensor, **kwargs) -> tuple[torch.Tensor, None]:
        """No quantization, return tensor as-is."""
        return tensor, None
    
    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: None, **kwargs) -> torch.Tensor:
        """No dequantization needed."""
        return quantized
    
    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
        """No scale needed."""
        return (0,)  # Empty shape

