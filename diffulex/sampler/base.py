import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists

from dataclasses import dataclass
from easydict import EasyDict as edict

from diffulex.engine.sequence import SequenceBase
from diffulex.logger import get_logger

logger = get_logger(__name__)


class SamplerBase(nn.Module):
    def __init__(self):
        super().__init__()
        from diffulex.attention import fetch_attn_metadata
        self.fetch_attn_metadata = fetch_attn_metadata

    def top_p_logits(self, logits, top_p):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
        mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        return logits

    def top_k_logits(self, logits, top_k):
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
        return logits

    def sample_tokens(self, logits, temperature=0.0, top_p=None, top_k=None, 
                      margin_confidence=False, neg_entropy=False):
        if temperature > 0:
            logits = logits / temperature
        if top_p is not None and top_p < 1:
            logits = self.top_p_logits(logits, top_p)
        if top_k is not None:
            logits = self.top_k_logits(logits, top_k)
        probs = torch.softmax(logits, dim=-1)

        if temperature > 0:
            try:
                x0 = dists.Categorical(probs=probs).sample()
                initial_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
            except:
                initial_confidence, x0 = probs.max(dim=-1)
        else:
            initial_confidence, x0 = probs.max(dim=-1)
        
        confidence = initial_confidence.clone()
        
        if margin_confidence:
            sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
            top1_probs = sorted_probs[:, 0] 
            top2_probs = sorted_probs[:, 1] 
            confidence = top1_probs - top2_probs 
        
        if neg_entropy:
            epsilon = 1e-10
            log_probs = torch.log(probs + epsilon)
            confidence = torch.sum(probs * log_probs, dim=-1)
        
        return confidence, x0, initial_confidence


@dataclass
class SampleOutputBase:
    true_local_ids_map: dict[str, dict[str, list[int]]]
    accepted_ids_map: dict[str, dict[str, list[int]]]
    sampled_tokens_map: dict[str, dict[str, list[int]]]
    
    def __post_init__(self):
        self.accepted_ids_map = edict(self.accepted_ids_map)
        self.sampled_tokens_map = edict(self.sampled_tokens_map)
        self.true_local_ids_map = edict(self.true_local_ids_map)
        
        
class SamplerShiftLogits(SamplerBase):
    def __init__(self):
        super().__init__()
        self.seq_last_logits_map: dict[str, torch.Tensor] = {}
        
    def _fetch_last_logits(self, logits: torch.Tensor, seq: SequenceBase) -> torch.Tensor:
        if seq.has_to_cache_block:
            last_logits = logits[seq.to_cache_last_token_id]
            self.seq_last_logits_map[seq.seq_id] = last_logits
        return self.seq_last_logits_map[seq.seq_id]
    
    def _shift_logits(self, logits, last_logit=None):
        if logits.shape[1] == 0:
            logger.warning("Logits sequence length is 0, returning empty logits")
            raise Exception("logits sequence length is 0")
            
        shifted_logits = torch.zeros_like(logits)
        shifted_logits[1:, ...] = logits[:-1, ...]
        if last_logit is not None:
            shifted_logits[0, ...] = last_logit
            return shifted_logits
        shifted_logits[0, ...] = 1.0
        return shifted_logits
    

class SamplerNoShiftLogits(SamplerBase):
    pass