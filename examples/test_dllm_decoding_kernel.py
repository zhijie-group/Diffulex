import time
import torch

from tqdm import tqdm
from einops import rearrange
from torch.nn.functional import scaled_dot_product_attention

from diffulex.legacy.layers.attention.ops import diffusion_lm_parallel_flash_decoding, diffusion_lm_flash_decoding


if __name__ == "__main__":
    torch.random.manual_seed(114514)
    
    num = 20
    seq_lens = torch.tensor([32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32]).to(torch.int32).to("cuda")[:num]
    seq_lens = seq_lens * 2
    ctx_lens = torch.tensor([119, 110,  81, 114, 112, 119, 110,  81, 114, 112, 
                119, 110,  81, 114, 112, 120, 111,  82, 115, 113]).to(torch.int32).to("cuda")[:num]
    total_lens = seq_lens + ctx_lens
    block_tables = torch.tensor(
        [[ 0],[ 1],[ 2],[ 3],[ 4],[ 5],[ 6], 
         [ 7],[ 8],[ 9],[10],[11],[12],[13],
         [14],[15],[16],[17],[18],[19]]).to(torch.int32).to("cuda")[:num]
    
    qkv_len = sum(seq_lens)
    q_shape = (qkv_len, 28, 128)
    k_shape = (qkv_len, 4, 128)
    precision = torch.bfloat16
    q = torch.randn(q_shape).to("cuda").to(precision)
    k = torch.randn(k_shape).to("cuda").to(precision)
    v = torch.randn_like(k).to("cuda").to(precision)

    kv_cache_shape = (307, 256, 4, 128)
    x = 8
    k_cache = torch.randn(kv_cache_shape).to("cuda").to(precision)
    v_cache = torch.randn_like(k_cache).to("cuda").to(precision)
    
    kv_out_shape = (sum(total_lens), 4, 128)
    k_output = torch.empty(kv_out_shape).to("cuda").to(precision)
    v_output = torch.empty_like(k_output).to("cuda").to(precision)

    build_up_cu_seqlens = lambda x: torch.tensor([0] + list(torch.cumsum(x, dim=0).cpu().numpy())).to(torch.int32).to("cuda")
    cu_seqlens_q = build_up_cu_seqlens(seq_lens)
    cu_seqlens_k = build_up_cu_seqlens(total_lens)
    
    # Mimic the mask creation in the original code, however wrong
    mask = torch.zeros((sum(seq_lens), sum(seq_lens)), dtype=torch.bool).to("cuda")
    for idx, (q_idx, k_idx) in enumerate(zip(cu_seqlens_q[:-1], cu_seqlens_q[:-1])):
        start_idx = q_idx
        end_idx = cu_seqlens_q[idx + 1]
        cur_len = end_idx - start_idx
        for blk_idx in range(cur_len // 32):
            h_start = start_idx + blk_idx * 32
            h_end = h_start + 32
            w_start = start_idx
            w_end = h_end
            mask[h_start: h_end, w_start: w_end] = True
    
    k_trans = lambda: rearrange(k_cache, "b s h (n x) -> b h n s x", n=kv_cache_shape[-1] // x, x=x).contiguous()
    v_trans = lambda: rearrange(v_cache, "b s h d -> b h d s").contiguous()
    k_cache = k_trans()
    v_cache = v_trans()
    cu_seqlens_q = build_up_cu_seqlens(seq_lens)
    o = torch.empty_like(q).to("cuda").to(precision)

    s = time.time()
    NSTEPS = 1 
    NLayers = 1 
    T = NSTEPS * NLayers
    for _ in tqdm(range(T)):
        diffusion_lm_parallel_flash_decoding(
            q, k, v, o, str(k_cache.dtype), k_cache, v_cache,
            block_tables, cu_seqlens_q, total_lens, 
            max(total_lens), max(seq_lens),
            1.0, 1.0, diffusion_blk_sz=32, mask=mask
        )
        # diffusion_lm_flash_decoding(
        #     q, k, v, mask, k_cache, v_cache, block_tables, 
        #     cu_seqlens_q, seq_lens, total_lens, ctx_lens,
        #     diffusion_block_size=32
        # )
    
    temp_o = o[:64]
    k_cache = rearrange(k_cache, "b h n s x -> b s h (n x)").contiguous()
    v_cache = rearrange(v_cache, "b h d s -> b s h d").contiguous()
    rearrange_fn = lambda ts: rearrange(ts, 's h d -> 1 h s d').contiguous()
    k_in = rearrange_fn(torch.cat([k_cache[0][:119], k[:64]]))
    v_in = rearrange_fn(torch.cat([v_cache[0][:119], v[:64]]))
    q_in = rearrange_fn(q[:64])
    mask_in = rearrange(torch.cat([
        torch.ones((64, 119), dtype=torch.bool).to("cuda"),
        torch.tril(torch.ones((64, 64), dtype=torch.bool)).to("cuda")
    ], dim=1), "h w -> 1 1 h w").contiguous()
    ref_o = scaled_dot_product_attention(q_in, k_in, v_in, mask_in, enable_gqa=True)
    ref_o = rearrange(ref_o, '1 h s d -> s h d').contiguous()
    assert torch.allclose(temp_o, ref_o, atol=1e-4, rtol=1e-4)
    
    print(f"Time taken: {time.time() - s:.4f} seconds")
    print(f"AVG time per iteration: {(time.time() - s) / NSTEPS:.4f} seconds")