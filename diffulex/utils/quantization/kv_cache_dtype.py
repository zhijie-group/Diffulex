from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch

try:
    # vLLM provides a platform-specific fp8 dtype (can be e4m3fn / e4m3fnuz, etc.)
    from vllm.platforms import current_platform  # type: ignore
except Exception:  # pragma: no cover
    current_platform = None


class KvCacheDType(IntEnum):
    BF16 = 0
    FP16 = 1
    FP32 = 2
    FP8_E4M3 = 3
    FP8_E5M2 = 4


@dataclass(frozen=True)
class KvCacheDTypeSpec:
    enum: KvCacheDType
    is_fp8: bool
    fp8_view_dtype: torch.dtype | None
    fp8_min: float | None
    fp8_max: float | None


def _normalize_kv_cache_dtype(kv_cache_dtype: str) -> str:
    s = (kv_cache_dtype or "").strip().lower()
    aliases = {
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp16": "fp16",
        "float16": "fp16",
        "fp32": "fp32",
        "float32": "fp32",
        "fp8": "fp8_e4m3",
        "fp8_e4m3": "fp8_e4m3",
        "e4m3": "fp8_e4m3",
        "fp8_e5m2": "fp8_e5m2",
        "e5m2": "fp8_e5m2",
    }
    if s not in aliases:
        raise ValueError(
            f"Unsupported kv_cache_dtype={kv_cache_dtype!r}. "
            "Supported: bf16/fp16/fp32/fp8/fp8_e4m3/fp8_e5m2"
        )
    return aliases[s]


def _get_fp8_e4m3_dtype() -> torch.dtype:
    if current_platform is None:
        if hasattr(torch, "float8_e4m3fn"):
            return torch.float8_e4m3fn  # type: ignore[attr-defined]
        raise RuntimeError("FP8 requested but vLLM current_platform is unavailable.")
    return current_platform.fp8_dtype()


def _get_fp8_e5m2_dtype() -> torch.dtype:
    if hasattr(torch, "float8_e5m2"):
        return torch.float8_e5m2  # type: ignore[attr-defined]
    if hasattr(torch, "float8_e5m2fnuz"):
        return torch.float8_e5m2fnuz  # type: ignore[attr-defined]
    raise RuntimeError(
        "FP8 E5M2 requested but this torch build does not expose float8_e5m2 dtype."
    )


def parse_kv_cache_dtype(kv_cache_dtype: str) -> KvCacheDTypeSpec:
    norm = _normalize_kv_cache_dtype(kv_cache_dtype)
    if norm == "bf16":
        return KvCacheDTypeSpec(KvCacheDType.BF16, False, None, None, None)
    if norm == "fp16":
        return KvCacheDTypeSpec(KvCacheDType.FP16, False, None, None, None)
    if norm == "fp32":
        return KvCacheDTypeSpec(KvCacheDType.FP32, False, None, None, None)

    if norm == "fp8_e4m3":
        fp8 = _get_fp8_e4m3_dtype()
        enum = KvCacheDType.FP8_E4M3
    elif norm == "fp8_e5m2":
        fp8 = _get_fp8_e5m2_dtype()
        enum = KvCacheDType.FP8_E5M2
    else:  # pragma: no cover
        raise AssertionError(norm)

    info = torch.finfo(fp8)
    return KvCacheDTypeSpec(
        enum=enum,
        is_fp8=True,
        fp8_view_dtype=fp8,
        fp8_min=float(info.min),
        fp8_max=float(info.max),
    )


def ensure_scale_tensor(
    scale: Any,
    *,
    num_kv_heads: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Returns a CUDA tensor suitable for Triton:
    - shape [num_kv_heads] (per-head)
    - dtype float32 by default
    """
    if scale is None:
        return torch.ones((num_kv_heads,), device=device, dtype=dtype)
    if isinstance(scale, (float, int)):
        return torch.full((num_kv_heads,), float(scale), device=device, dtype=dtype)
    if isinstance(scale, torch.Tensor):
        if scale.numel() == 1:
            return torch.full((num_kv_heads,), float(scale.item()), device=device, dtype=dtype)
        if scale.numel() != num_kv_heads:
            raise ValueError(
                f"scale must be scalar or shape [num_kv_heads]={num_kv_heads}, got {tuple(scale.shape)}"
            )
        return scale.to(device=device, dtype=dtype).contiguous()
    raise TypeError(f"Unsupported scale type: {type(scale)}")


def view_fp8_cache(cache: torch.Tensor, kv_cache_dtype: str) -> torch.Tensor:
    """
    FP8 KV cache uses uint8 as storage for compatibility. This returns a view tensor
    whose dtype is fp8, so Triton will see the correct element type.
    """
    spec = parse_kv_cache_dtype(kv_cache_dtype)
    if not spec.is_fp8:
        return cache
    assert spec.fp8_view_dtype is not None
    if cache.dtype == torch.uint8:
        return cache.view(spec.fp8_view_dtype)
    if cache.dtype == spec.fp8_view_dtype:
        return cache
    raise AssertionError(
        f"FP8 cache must be torch.uint8 (storage) or {spec.fp8_view_dtype}, got {cache.dtype}"
    )


