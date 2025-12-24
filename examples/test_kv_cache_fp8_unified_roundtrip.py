import torch

from dataclasses import dataclass
from typing import List

from vllm.platforms import current_platform

from diffulex_legacy.layers.attention.ops import store_kvcache_unified_layout, load_kvcache


@dataclass
class _Seq:
    diffusion_block_size: int = 32


@dataclass
class _Ctx:
    seq_lens_ts: torch.Tensor
    context_lens: torch.Tensor
    total_lens: torch.Tensor
    block_tables: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    seq_lens: List[int] = None
    seqs: List[_Seq] = None

    def __post_init__(self):
        self.seq_lens = self.seq_lens_ts.tolist()
        # load_kvcache only reads seqs[0].diffusion_block_size
        self.seqs = [_Seq()]


def _build_cu_seqlens(x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [0] + list(torch.cumsum(x, dim=0).cpu().numpy()),
        dtype=torch.int32,
        device="cuda",
    )


if __name__ == "__main__":
    torch.random.manual_seed(114514)

    num_seqs = 4
    blk_sz = 256
    H = 4
    head_dim = 128

    # Make seq_len multiple of diffusion_block_size(32)
    seq_lens = torch.tensor([64, 32, 64, 32], dtype=torch.int32, device="cuda")
    ctx_lens = torch.tensor([119, 110, 81, 114], dtype=torch.int32, device="cuda")
    assert seq_lens.numel() == num_seqs and ctx_lens.numel() == num_seqs
    total_lens = seq_lens + ctx_lens

    # Tokens are packed per-seq: [ctx_tokens..., new_tokens...]
    kv_shape = (int(total_lens.sum().item()), H, head_dim)
    k_all = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16)
    v_all = torch.randn_like(k_all)

    # slot_mapping: map ctx tokens into block slots; new tokens -> -1 (not cached here)
    slot_mapping: list[int] = []
    start = 0
    for seq_idx in range(num_seqs):
        ctx = int(ctx_lens[seq_idx].item())
        new = int(seq_lens[seq_idx].item())
        slot_mapping.extend(list(range(seq_idx * blk_sz, seq_idx * blk_sz + ctx)))
        slot_mapping.extend([-1] * new)
        start += ctx + new
    slot_mapping_ts = torch.tensor(slot_mapping, dtype=torch.int64, device="cuda")
    assert slot_mapping_ts.numel() == kv_shape[0]

    # FP8 cache uses uint8 storage.
    kv_cache_shape = (num_seqs, blk_sz, H, head_dim)
    k_cache_u8 = torch.zeros(kv_cache_shape, device="cuda", dtype=torch.uint8)
    v_cache_u8 = torch.zeros_like(k_cache_u8)

    fp8 = current_platform.fp8_dtype()
    fp8_max = float(torch.finfo(fp8).max)
    eps = 1e-6
    k_absmax = k_all.to(torch.float32).abs().amax(dim=(0, 2))  # [H]
    v_absmax = v_all.to(torch.float32).abs().amax(dim=(0, 2))  # [H]
    k_scale = (k_absmax / fp8_max).clamp_min(eps)
    v_scale = (v_absmax / fp8_max).clamp_min(eps)

    store_kvcache_unified_layout(
        k_all,
        v_all,
        k_cache_u8,
        v_cache_u8,
        slot_mapping_ts,
        model_type="diffusion_lm",
        kv_cache_dtype="fp8_e4m3",
        k_scale=k_scale,
        v_scale=v_scale,
    )

    # Check stored ctx portion (dequantize cache and compare to original ctx tokens).
    k_cache_fp8 = k_cache_u8.view(fp8).to(torch.float32) * k_scale[None, None, :, None]
    v_cache_fp8 = v_cache_u8.view(fp8).to(torch.float32) * v_scale[None, None, :, None]
    start = 0
    for seq_idx in range(num_seqs):
        ctx = int(ctx_lens[seq_idx].item())
        new = int(seq_lens[seq_idx].item())
        k_ctx_ref = k_all[start : start + ctx].to(torch.float32)
        v_ctx_ref = v_all[start : start + ctx].to(torch.float32)
        k_ctx_got = k_cache_fp8[seq_idx, :ctx]  # [ctx, H, D]
        v_ctx_got = v_cache_fp8[seq_idx, :ctx]
        assert torch.allclose(k_ctx_got, k_ctx_ref, atol=1e-1, rtol=1e-1)
        assert torch.allclose(v_ctx_got, v_ctx_ref, atol=1e-1, rtol=1e-1)
        start += ctx + new

    # Now test load_kvcache: output = [ctx(from cache), new(from k_new)]
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

    block_tables = torch.arange(num_seqs, dtype=torch.int32, device="cuda").view(num_seqs, 1)
    cu_seqlens_q = _build_cu_seqlens(seq_lens)
    cu_seqlens_k = _build_cu_seqlens(total_lens)
    ctx = _Ctx(
        seq_lens_ts=seq_lens,
        context_lens=ctx_lens,
        total_lens=total_lens,
        block_tables=block_tables,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
    )

    k_out, v_out = load_kvcache(
        k_cache_u8,
        v_cache_u8,
        ctx,
        k_new,
        v_new,
        kv_cache_dtype="fp8_e4m3",
        k_scale=k_scale,
        v_scale=v_scale,
    )

    # Verify new part is exact and ctx part is within fp8 tolerance.
    out_splits = torch.split(k_out, total_lens.tolist(), dim=0)
    new_splits = torch.split(k_new, seq_lens.tolist(), dim=0)
    start = 0
    for seq_idx in range(num_seqs):
        ctx_len = int(ctx_lens[seq_idx].item())
        new_len = int(seq_lens[seq_idx].item())
        k_ref_ctx = k_all[start : start + ctx_len].to(k_out.dtype)
        k_got_ctx = out_splits[seq_idx][:ctx_len]
        assert torch.allclose(k_got_ctx, k_ref_ctx, atol=1e-1, rtol=1e-1)
        assert torch.equal(out_splits[seq_idx][ctx_len : ctx_len + new_len], new_splits[seq_idx])
        start += ctx_len + new_len

    print("FP8 unified KV cache store/load roundtrip: OK")


