import torch

from einops import rearrange
from vllm.platforms import current_platform

from diffulex_legacy.layers.attention.ops import store_kvcache_distinct_layout


if __name__ == "__main__":
    torch.random.manual_seed(114514)

    num_seqs = 4
    blk_sz = 256
    H = 4
    head_dim = 128
    x = 8

    seq_lens = torch.tensor([64, 32, 64, 32], dtype=torch.int32, device="cuda")
    ctx_lens = torch.tensor([119, 110, 81, 114], dtype=torch.int32, device="cuda")
    total_lens = seq_lens + ctx_lens

    kv_shape = (int(total_lens.sum().item()), H, head_dim)
    k_all = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16)
    v_all = torch.randn_like(k_all)

    slot_mapping: list[int] = []
    start = 0
    for seq_idx in range(num_seqs):
        ctx = int(ctx_lens[seq_idx].item())
        new = int(seq_lens[seq_idx].item())
        slot_mapping.extend(list(range(seq_idx * blk_sz, seq_idx * blk_sz + ctx)))
        slot_mapping.extend([-1] * new)
        start += ctx + new
    slot_mapping_ts = torch.tensor(slot_mapping, dtype=torch.int64, device="cuda")

    # Distinct cache: k [B, H, D//x, S, x], v [B, H, D, S]
    k_cache_u8 = torch.zeros((num_seqs, H, head_dim // x, blk_sz, x), device="cuda", dtype=torch.uint8)
    v_cache_u8 = torch.zeros((num_seqs, H, head_dim, blk_sz), device="cuda", dtype=torch.uint8)

    fp8 = current_platform.fp8_dtype()
    fp8_max = float(torch.finfo(fp8).max)
    eps = 1e-6
    k_absmax = k_all.to(torch.float32).abs().amax(dim=(0, 2))  # [H]
    v_absmax = v_all.to(torch.float32).abs().amax(dim=(0, 2))  # [H]
    k_scale = (k_absmax / fp8_max).clamp_min(eps)
    v_scale = (v_absmax / fp8_max).clamp_min(eps)

    store_kvcache_distinct_layout(
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

    # Dequantize and convert back to unified layout for easy checking.
    k_cache_fp8 = k_cache_u8.view(fp8).to(torch.float32)
    v_cache_fp8 = v_cache_u8.view(fp8).to(torch.float32)
    k_cache_deq = k_cache_fp8 * k_scale[None, :, None, None, None]
    v_cache_deq = v_cache_fp8 * v_scale[None, :, None, None]
    k_cache_unified = rearrange(k_cache_deq, "b h n s x -> b s h (n x)").contiguous()
    v_cache_unified = rearrange(v_cache_deq, "b h d s -> b s h d").contiguous()

    start = 0
    for seq_idx in range(num_seqs):
        ctx = int(ctx_lens[seq_idx].item())
        new = int(seq_lens[seq_idx].item())
        k_ctx_ref = k_all[start : start + ctx].to(torch.float32)
        v_ctx_ref = v_all[start : start + ctx].to(torch.float32)
        assert torch.allclose(k_cache_unified[seq_idx, :ctx], k_ctx_ref, atol=1e-1, rtol=1e-1)
        assert torch.allclose(v_cache_unified[seq_idx, :ctx], v_ctx_ref, atol=1e-1, rtol=1e-1)
        start += ctx + new

    print("FP8 distinct KV cache store roundtrip (ctx portion): OK")


