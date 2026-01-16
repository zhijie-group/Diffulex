import pytest
import torch

from types import SimpleNamespace

from diffulex.utils.quantization.factory import QuantizationStrategyFactory
from diffulex_kernel import store_kvcache_distinct_layout, load_kvcache


def _has_fp8() -> bool:
    return hasattr(torch, "float8_e4m3fn") or hasattr(torch, "float8_e4m3fnuz") or hasattr(torch, "float8_e5m2")


def _build_cu_seqlens(x: torch.Tensor) -> torch.Tensor:
    # x: [num_seqs] int32 on cuda
    return torch.tensor(
        [0] + list(torch.cumsum(x, dim=0).cpu().numpy()),
        dtype=torch.int32,
        device=x.device,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton KV-cache kernels")
@pytest.mark.skipif(not _has_fp8(), reason="This torch build does not expose FP8 dtypes")
def test_fp8_kv_cache_distinct_store_and_load():
    """
    Regression test for FP8 KV cache distinct layout:
    - store: quantize+store context into distinct cache (uint8 storage)
    - load: fused gather+dequant+scale from distinct cache into BF16 output,
            and append active KV (k_new/v_new) exactly.
    """
    torch.manual_seed(1234)
    device = torch.device("cuda")

    # Enable FP8 KV quantization strategy in the global quantization context.
    QuantizationStrategyFactory.create_from_config(SimpleNamespace(kv_cache_dtype="fp8_e4m3"))

    num_seqs = 2
    blk_sz = 64
    num_kv_heads = 4
    head_dim = 128
    x = 8
    diffusion_block_size = 32

    # ctx/new lengths (make new divisible by diffusion_block_size to match kernel loop)
    ctx_lens = torch.tensor([37, 55], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([32, 32], dtype=torch.int32, device=device)
    total_lens = ctx_lens + seq_lens

    # Build concatenated [sum(total_lens), H, D] for store reference.
    k_all = torch.randn((int(total_lens.sum().item()), num_kv_heads, head_dim), device=device, dtype=torch.bfloat16)
    v_all = torch.randn_like(k_all)

    # slot_mapping: context tokens map to their block slots; new tokens use -1 (not stored).
    slot_mapping: list[int] = []
    start = 0
    for seq_idx in range(num_seqs):
        ctx = int(ctx_lens[seq_idx].item())
        new = int(seq_lens[seq_idx].item())
        slot_mapping.extend(list(range(seq_idx * blk_sz, seq_idx * blk_sz + ctx)))
        slot_mapping.extend([-1] * new)
        start += ctx + new
    slot_mapping_ts = torch.tensor(slot_mapping, dtype=torch.int64, device=device)

    # Distinct caches (uint8 storage for FP8).
    k_cache_u8 = torch.zeros((num_seqs, num_kv_heads, head_dim // x, blk_sz, x), device=device, dtype=torch.uint8)
    v_cache_u8 = torch.zeros((num_seqs, num_kv_heads, head_dim, blk_sz), device=device, dtype=torch.uint8)

    # Scales: per-head absmax / fp8_max (same convention as strategy).
    from diffulex.utils.quantization.kv_cache_dtype import parse_kv_cache_dtype

    spec = parse_kv_cache_dtype("fp8_e4m3")
    assert spec.is_fp8 and spec.fp8_max is not None
    fp8_max = float(spec.fp8_max)
    eps = 1e-6
    k_absmax = k_all.to(torch.float32).abs().amax(dim=(0, 2))
    v_absmax = v_all.to(torch.float32).abs().amax(dim=(0, 2))
    k_scale = (k_absmax / fp8_max).clamp_min(eps).to(torch.float32)
    v_scale = (v_absmax / fp8_max).clamp_min(eps).to(torch.float32)

    # Minimal metadata required by store/load.
    block_tables = torch.arange(num_seqs, dtype=torch.int32, device=device).view(num_seqs, 1)
    md = SimpleNamespace(
        kv_cache_layout="distinct",
        need_kv_cache_store=True,
        slot_mapping=slot_mapping_ts,
        context_lens=ctx_lens,
        seq_lens_ts=seq_lens,
        block_tables=block_tables,
        cu_seqlens_q=_build_cu_seqlens(seq_lens),
        cu_seqlens_k=_build_cu_seqlens(total_lens),
        max_seqlen_q=int(seq_lens.max().item()),
        max_seqlen_k=int(total_lens.max().item()),
        seqs=[SimpleNamespace(diffusion_block_size=diffusion_block_size)],
        k_scale=k_scale,
        v_scale=v_scale,
    )

    # Store context into cache.
    store_kvcache_distinct_layout(k_all, v_all, k_cache_u8, v_cache_u8, slot_mapping_ts, md)

    # Build k_new/v_new (only active tokens, concatenated over sequences).
    k_new_list = []
    v_new_list = []
    start = 0
    for seq_idx in range(num_seqs):
        ctx = int(ctx_lens[seq_idx].item())
        new = int(seq_lens[seq_idx].item())
        k_new_list.append(k_all[start + ctx : start + ctx + new])
        v_new_list.append(v_all[start + ctx : start + ctx + new])
        start += ctx + new
    k_new = torch.cat(k_new_list, dim=0).contiguous()
    v_new = torch.cat(v_new_list, dim=0).contiguous()

    # Load (fused dequant + gather) and append new tokens.
    k_out, v_out = load_kvcache(k_cache_u8, v_cache_u8, md, k_new, v_new)

    # Split outputs per sequence to check ctx/new portions.
    out_splits_k = torch.split(k_out, total_lens.tolist(), dim=0)
    out_splits_v = torch.split(v_out, total_lens.tolist(), dim=0)
    new_splits_k = torch.split(k_new, seq_lens.tolist(), dim=0)
    new_splits_v = torch.split(v_new, seq_lens.tolist(), dim=0)

    start = 0
    for seq_idx in range(num_seqs):
        ctx = int(ctx_lens[seq_idx].item())
        new = int(seq_lens[seq_idx].item())

        k_ctx_ref = k_all[start : start + ctx].to(torch.float32)
        v_ctx_ref = v_all[start : start + ctx].to(torch.float32)
        k_ctx_got = out_splits_k[seq_idx][:ctx].to(torch.float32)
        v_ctx_got = out_splits_v[seq_idx][:ctx].to(torch.float32)

        # Quantization error tolerance (FP8).
        assert torch.allclose(k_ctx_got, k_ctx_ref, atol=2e-1, rtol=2e-1)
        assert torch.allclose(v_ctx_got, v_ctx_ref, atol=2e-1, rtol=2e-1)

        # New tokens should be appended exactly (no quantization).
        assert torch.equal(out_splits_k[seq_idx][ctx : ctx + new], new_splits_k[seq_idx])
        assert torch.equal(out_splits_v[seq_idx][ctx : ctx + new], new_splits_v[seq_idx])

        start += ctx + new

