import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists

from dataclasses import dataclass
from easydict import EasyDict as edict

from diffulex.config import Config
from diffulex.attention import fetch_attn_metadata


class SamplerForDiffusionLM(nn.Module):
    def __init__(self):
        super().__init__()

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
class SampleOutputForDiffusionLM:
    true_local_ids_map: dict[str, dict[str, list[int]]]
    accepted_ids_map: dict[str, list[int]]
    sampled_tokens_map: dict[str, dict[str, list[int]]]
    
    def __post_init__(self):
        self.accepted_ids_map = edict(self.accepted_ids_map)
        self.sampled_tokens_map = edict(self.sampled_tokens_map)
        self.true_local_ids_map = edict(self.true_local_ids_map)
    

class SamplerForDream(SamplerForDiffusionLM):
    def _shift_logits(self, logits, last_logit=None):
        if logits.shape[1] == 0:
            print("Warning: logits sequence length is 0, returning empty logits")
            raise Exception("logits sequence length is 0")
            
        shifted_logits = torch.zeros_like(logits)
        shifted_logits[1:, ...] = logits[:-1, ...]
        if last_logit is not None:
            shifted_logits[0, ...] = last_logit
            return shifted_logits
        shifted_logits[0, ...] = 1.0
        return shifted_logits
    
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor,
                top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
        context = fetch_attn_metadata()
        seqs = context.seqs
        split_logits = torch.split(logits, [len(seq) for seq in seqs] if context.is_prefill else context.seq_lens, dim=0)
        accepted_ids_map = {}
        sampled_tokens_map = {}
        true_local_ids_map = {}
        for temperature, seq, seq_logits in zip(temperatures, seqs, split_logits):
            true_local_ids_sub_map = {}
            accepted_ids_sub_map = {}
            sampled_tokens_sub_map = {}
            shifted_logits = self._shift_logits(seq_logits, seq.cached_or_caching_last_token_id)
            for block_id, block in enumerate(seq.diffusion_blocks):
                if not block.is_active or sum(block.local_mask_tokens) == 0:
                    continue
                
                if len(block.global_mask_token_ids) > 0:
                    mask_token_logits = shifted_logits[block.global_mask_token_ids, ...]
                    confidence, sampled_tokens, initial_confidence = self.sample_tokens(
                        mask_token_logits, 
                        temperature, 
                        top_p=top_p, 
                        top_k=top_k, 
                        neg_entropy=(neg_entropy == "neg_entropy"),
                        margin_confidence=(margin_confidence == "margin_confidence")
                    )
                    
                if block.pre_block_complete:
                    high_conf_indices = torch.where(initial_confidence > block.accept_threshold)[0]        
                    if len(high_conf_indices) == 0:
                        number_transfer_tokens = 1
                        _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                    else:
                        transfer_index = torch.tensor([], device=sampled_tokens.device, dtype=torch.long)
                    accepted_ids = torch.unique(torch.cat([transfer_index, high_conf_indices]))
                else:
                    high_conf_indices = torch.where(initial_confidence > block.accept_threshold)[0]
                    accepted_ids = high_conf_indices

                true_local_ids_sub_map[str(block_id)] = [block.local_mask_token_ids[accepted_id] for accepted_id in accepted_ids.tolist()]
                accepted_ids_sub_map[str(block_id)] = accepted_ids.tolist()
                sampled_tokens_sub_map[str(block_id)] = sampled_tokens
            
            seq_idx = str(seq.seq_id)
            true_local_ids_map[seq_idx] = true_local_ids_sub_map
            accepted_ids_map[seq_idx] = accepted_ids_sub_map
            sampled_tokens_map[seq_idx] = sampled_tokens_sub_map

        return SampleOutputForDiffusionLM(
            true_local_ids_map=true_local_ids_map,
            accepted_ids_map=accepted_ids_map,
            sampled_tokens_map=sampled_tokens_map
        )


class SamplerForLLaDA(SamplerForDiffusionLM):
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor,
                top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
        context = fetch_attn_metadata()
        seqs = context.seqs
        split_logits = torch.split(logits, [len(seq) for seq in seqs] if context.is_prefill else context.seq_lens, dim=0)
        accepted_ids_map = {}
        sampled_tokens_map = {}
        true_local_ids_map = {}
        for temperature, seq, seq_logits in zip(temperatures, seqs, split_logits):
            true_local_ids_sub_map = {}
            accepted_ids_sub_map = {}
            sampled_tokens_sub_map = {}
            for block_id, block in enumerate(seq.diffusion_blocks):
                if not block.is_active or sum(block.local_mask_tokens) == 0:
                    continue
                
                if len(block.global_mask_token_ids) > 0:
                    mask_token_logits = seq_logits[block.global_mask_token_ids, ...]
                    confidence, sampled_tokens, initial_confidence = self.sample_tokens(
                        mask_token_logits, 
                        temperature, 
                        top_p=top_p, 
                        top_k=top_k, 
                        neg_entropy=(neg_entropy == "neg_entropy"),
                        margin_confidence=(margin_confidence == "margin_confidence")
                    )
                    
                if block.pre_block_complete:
                    high_conf_indices = torch.where(initial_confidence > block.accept_threshold)[0]        
                    if len(high_conf_indices) == 0:
                        number_transfer_tokens = 1
                        _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                    else:
                        transfer_index = torch.tensor([], device=sampled_tokens.device, dtype=torch.long)
                    accepted_ids = torch.unique(torch.cat([transfer_index, high_conf_indices]))
                else:
                    high_conf_indices = torch.where(initial_confidence > block.accept_threshold)[0]
                    accepted_ids = high_conf_indices

                true_local_ids_sub_map[str(block_id)] = [block.local_mask_token_ids[accepted_id] for accepted_id in accepted_ids.tolist()]
                accepted_ids_sub_map[str(block_id)] = accepted_ids.tolist()
                sampled_tokens_sub_map[str(block_id)] = sampled_tokens
            
            seq_idx = str(seq.seq_id)
            true_local_ids_map[seq_idx] = true_local_ids_sub_map
            accepted_ids_map[seq_idx] = accepted_ids_sub_map
            sampled_tokens_map[seq_idx] = sampled_tokens_sub_map

        return SampleOutputForDiffusionLM(
            true_local_ids_map=true_local_ids_map,
            accepted_ids_map=accepted_ids_map,
            sampled_tokens_map=sampled_tokens_map
        )


class AutoSampler:
    MODEL_MAPPING = {
        "dream": SamplerForDream,
        "llada": SamplerForLLaDA
    }
    @classmethod
    def from_config(cls, config: Config):
        return cls.MODEL_MAPPING[config.model_name]()