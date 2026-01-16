from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch


_EXT: Optional[object] = None
_EXT_ERR: Optional[BaseException] = None


def _build_extension() -> object:
    # Allow disabling compilation in constrained environments.
    if os.getenv("DIFFULEX_DISABLE_MARLIN", "0") == "1":
        raise RuntimeError("DIFFULEX_DISABLE_MARLIN=1 (disabled)")

    this_dir = Path(__file__).resolve().parent
    # this_dir = Diffulex/diffulex_kernel/python
    # parents[0]=Diffulex/diffulex_kernel, parents[1]=Diffulex
    repo_root = this_dir.parents[1]  # Diffulex/
    csrc_dir = repo_root / "diffulex_kernel" / "csrc" / "marlin"

    sources = [
        str(csrc_dir / "torch_bindings_marlin.cpp"),
        str(csrc_dir / "allspark_repack.cu"),
        str(csrc_dir / "allspark_qgemm_w8a16.cu"),
    ]

    # Build via torch cpp_extension
    from torch.utils.cpp_extension import load  # lazy import

    extra_cflags = ["-O3"]
    extra_cuda_cflags = ["-O3", "--use_fast_math"]
    extra_ldflags = ["-lcublas"]

    # Use a stable extension name so torch caches it in ~/.cache/torch_extensions.
    name = "diffulex_marlin_allspark_w8a16"

    return load(
        name=name,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        with_cuda=True,
        verbose=os.getenv("DIFFULEX_MARLIN_VERBOSE_BUILD", "0") == "1",
    )


def _get_ext() -> object:
    global _EXT, _EXT_ERR
    if _EXT is not None:
        return _EXT
    if _EXT_ERR is not None:
        raise _EXT_ERR
    try:
        _EXT = _build_extension()
        return _EXT
    except BaseException as e:
        _EXT_ERR = e
        raise


def is_available() -> bool:
    try:
        _ = _get_ext()
        return True
    except BaseException:
        return False


def allspark_w8a16_gemm(
    a: torch.Tensor,
    b_qweight: torch.Tensor,
    b_scales: torch.Tensor,
    b_qzeros: Optional[torch.Tensor],
    n: int,
    group_size: int,
    sm_count: int,
    sm_version: int,
    cublas_m_threshold: int,
    has_zp: bool,
    n32k16_reorder: bool,
) -> torch.Tensor:
    ext = _get_ext()
    return ext.allspark_w8a16_gemm(
        a,
        b_qweight,
        b_scales,
        b_qzeros,
        n,
        group_size,
        sm_count,
        sm_version,
        cublas_m_threshold,
        has_zp,
        n32k16_reorder,
    )


def rearrange_kn_weight_as_n32k16_order(
    b_qweight_kn: torch.Tensor,
    b_scales: torch.Tensor,
    b_zeros: Optional[torch.Tensor],
    has_zp: bool,
    b_qweight_reorder: torch.Tensor,
    b_scales_reorder: torch.Tensor,
    b_zeros_reorder: Optional[torch.Tensor],
    K: int,
    N: int,
    N_32align: int,
) -> None:
    ext = _get_ext()
    return ext.rearrange_kn_weight_as_n32k16_order(
        b_qweight_kn,
        b_scales,
        b_zeros,
        has_zp,
        b_qweight_reorder,
        b_scales_reorder,
        b_zeros_reorder,
        K,
        N,
        N_32align,
    )

