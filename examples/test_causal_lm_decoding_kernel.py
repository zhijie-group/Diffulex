import torch

from diffulex.legacy.layers.attention.ops.triton_decode_attn_clm import causal_lm_decode_attention_fwd

if __name__ == "__main__":
    torch.random.manual_seed(114514)
    
    seq_lens = torch.tensor([32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32]).to(torch.int32).to("cuda")
    ctx_lens = torch.tensor([119, 110,  81, 114, 112, 119, 110,  81, 114, 112, 
                119, 110,  81, 114, 112, 120, 111,  82, 115, 113]).to(torch.int32).to("cuda")
    total_lens = seq_lens + ctx_lens
    block_tables = torch.tensor(
        [[ 0],[ 1],[ 2],[ 3],[ 4],[ 5],[ 6], 
         [ 7],[ 8],[ 9],[10],[11],[12],[13],
         [14],[15],[16],[17],[18],[19]]).to(torch.int32).to("cuda")
    q_cache_shape = (20, 28, 128)
    q = torch.randn(q_cache_shape).to("cuda")
    k = torch.randn_like(q).to("cuda")
    v = torch.randn_like(q).to("cuda")
    
    kv_cache_shape = (307, 256, 4, 128)
    k_cache = torch.randn(kv_cache_shape).to("cuda")
    v_cache = torch.randn_like(k_cache).to("cuda")

    kv_new_shape = (sum(seq_lens), 4, 128)
    k_new = torch.randn(kv_new_shape).to("cuda")
    v_new = torch.randn_like(k_new).to("cuda")
    
    kv_out_shape = (sum(total_lens), 4, 128)
    k_output = torch.empty(kv_out_shape).to("cuda")
    v_output = torch.empty_like(k_output).to("cuda")
    
    build_up_cu_seqlens = lambda x: torch.tensor([0] + list(torch.cumsum(x, dim=0).cpu().numpy())).to(torch.int32).to("cuda")
    cu_seqlens_q = build_up_cu_seqlens(seq_lens)
    cu_seqlens_k = build_up_cu_seqlens(total_lens)
    
    causal_lm_decode_attention_fwd(
        q,
        k_cache,
        v_cache,
        attn_logits=None,
        block_tables=block_tables,
        cache_seqlens=cu_seqlens_k,
        num_kv_splits=1,
        softmax_scale=None,
        page_size=256,
        logit_cap=0.0,
    )