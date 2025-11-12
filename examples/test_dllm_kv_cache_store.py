import time
import torch

from einops import rearrange

from diffulex.legacy.layers.attention.attention_v4 import store_kvcache_distinct_layout, store_kvcache_unified


if __name__ == "__main__":
    torch.random.manual_seed(114514)
    
    seq_lens = torch.tensor([32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 
                             32, 32, 32, 32, 32, 32, 32, 32, 32, 32]).to(torch.int32).to("cuda")
    ctx_lens = torch.tensor([119, 110,  81, 114, 112, 119, 110,  81, 114, 112, 
                             119, 110,  81, 114, 112, 120, 111,  82, 115, 113]).to(torch.int32).to("cuda")
    total_lens = seq_lens + ctx_lens
    
    kv_shape = (sum(total_lens), 4, 128)
    k = torch.randn(kv_shape).to("cuda").to(torch.bfloat16)
    v = torch.randn_like(k).to("cuda").to(torch.bfloat16)

    kv_cache_shape = (307, 256, 4, 128)
    x = 8
    k_cache = torch.zeros(kv_cache_shape).to("cuda").to(torch.bfloat16)
    v_cache = torch.zeros_like(k_cache).to("cuda").to(torch.bfloat16)
    
    slot_mapping = []
    blk_sz = 256
    diff_blk_sz = 32
    for idx, ctx_len in enumerate(ctx_lens):
        slot_mapping.extend(list(range(idx * blk_sz, idx * blk_sz + ctx_len)))
        slot_mapping.extend([-1] * diff_blk_sz)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64).to("cuda")
    
    # unified_layout
    s = time.time()
    store_kvcache_unified(k, v, k_cache, v_cache, slot_mapping, model_type='diffusion_lm')
    print(f"Unified layout KV cache stored in {time.time() - s:.4f} seconds.")
    
    start_idx = 0
    for idx, (l, ctx) in enumerate(zip(total_lens, ctx_lens)):
        assert torch.allclose(k_cache[idx, :ctx], k[start_idx:start_idx + ctx])
        assert torch.allclose(v_cache[idx, :ctx], v[start_idx:start_idx + ctx])
        start_idx += l
        
    print("Unified layout KV cache stored successfully.")
    
    # distinct_layout
    k_cache = torch.zeros(kv_cache_shape).to("cuda").to(torch.bfloat16)
    v_cache = torch.zeros_like(k_cache).to("cuda").to(torch.bfloat16)
    k_cache = rearrange(k_cache, "b s h (n x) -> b h n s x", n=kv_cache_shape[-1] // x, x=x).contiguous()
    v_cache = rearrange(v_cache, "b s h d -> b h d s").contiguous()
    
    s = time.time()
    store_kvcache_distinct_layout(k, v, k_cache, v_cache, slot_mapping, model_type='diffusion_lm')
    print(f"Distinct layout KV cache stored in {time.time() - s:.4f} seconds.")

    k_cache = rearrange(k_cache, "b h n s x -> b s h (n x)").contiguous()
    v_cache = rearrange(v_cache, "b h d s -> b s h d").contiguous()
    
    start_idx = 0
    for idx, (l, ctx) in enumerate(zip(total_lens, ctx_lens)):
        assert torch.allclose(k_cache[idx, :ctx], k[start_idx:start_idx + ctx])
        assert torch.allclose(v_cache[idx, :ctx], v[start_idx:start_idx + ctx])
        start_idx += l
    
    print("Distinct layout KV cache stored successfully.")