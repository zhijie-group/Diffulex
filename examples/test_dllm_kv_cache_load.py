import torch

from typing import List
from dataclasses import dataclass

from mimic_data.mimic_slot_mapping import slot_mapping
from diffulex.legacy.layers.attention.ops import store_kvcache_unified_layout, load_kvcache, CHECK_LOADING

@dataclass
class MimicSequenceForDiffusionLM:
    diffusion_block_size: int = 32

@dataclass
class MimicContextForDiffusionLM:
    seq_lens_ts: torch.Tensor
    context_lens: torch.Tensor
    total_lens: torch.Tensor
    block_tables: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    seq_lens: List[int] = None
    seqs: List[MimicSequenceForDiffusionLM] = None
    
    def __post_init__(self):
        self.seq_lens = self.seq_lens_ts.tolist()
        self.seqs = [MimicSequenceForDiffusionLM()]
        
if __name__ == "__main__":
    torch.random.manual_seed(114514)
    
    seq_lens = torch.tensor([64, 64, 32, 64, 64, 32, 64, 32, 64, 64, 
                             32, 64, 64, 64, 64, 64, 32, 32, 64, 64]).to(torch.int32).to("cuda")
    ctx_lens = torch.tensor([119, 110,  81, 114, 112, 119, 110,  81, 114, 112, 
                             119, 110,  81, 114, 112, 120, 111,  82, 115, 113]).to(torch.int32).to("cuda")
    total_lens = seq_lens + ctx_lens
    block_tables = torch.tensor(
        [[ 0],[ 1],[ 2],[ 3],[ 4],[ 5],[ 6], 
         [ 7],[ 8],[ 9],[10],[11],[12],[13],
         [14],[15],[16],[17],[18],[19]]).to(torch.int32).to("cuda")
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64).to("cuda")
    
    kv_cache_shape = (1342, 256, 4, 128)
    k_cache = torch.randn(kv_cache_shape).to("cuda")
    v_cache = torch.randn_like(k_cache).to("cuda")

    kv_new_shape = (sum(seq_lens), 4, 128)
    k_new = torch.randn(kv_new_shape).to("cuda")
    v_new = torch.randn_like(k_new).to("cuda")
    
    build_up_cu_seqlens = lambda x: torch.tensor([0] + list(torch.cumsum(x, dim=0).cpu().numpy())).to(torch.int32).to("cuda")
    cu_seqlens_q = build_up_cu_seqlens(seq_lens)
    cu_seqlens_k = build_up_cu_seqlens(total_lens)

    context = MimicContextForDiffusionLM(
        seq_lens_ts=seq_lens,
        context_lens=ctx_lens,
        total_lens=total_lens,
        block_tables=block_tables,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k
    )
    
    store_kvcache_unified_layout(k_new, v_new, k_cache, v_cache, slot_mapping, model_type='diffusion_lm')

    k_comb, v_comb = load_kvcache(k_cache, v_cache, context, k_new, v_new)
    CHECK_LOADING(k_comb, v_comb, k_new, v_new, k_cache, v_cache, context)