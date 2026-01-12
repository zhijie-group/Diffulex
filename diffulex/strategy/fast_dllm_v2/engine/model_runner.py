from __future__ import annotations

import time

from multiprocessing.synchronize import Event

import torch
from tqdm import tqdm

from diffulex.config import Config
from diffulex.engine.sequence import SequenceBase
from diffulex.strategy.fast_dllm_v2.engine.sequence import FDV2Sequence
from diffulex.attention.metadata import set_fetch_fn_for_attn_metadata, set_warming_up, reset_warming_up
from diffulex.engine.model_runner import AutoModelRunner, ModelRunnerBase
from diffulex.strategy.fast_dllm_v2.attention.metadata import fetch_fdv2_attn_metadata, set_fdv2_attn_metadata, reset_fdv2_attn_metadata


@AutoModelRunner.register("fast_dllm_v2", is_default=True)
class FastDLLMV2ModelRunner(ModelRunnerBase):
    """Reference implementation of Block Diffusion decoding strategy."""
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        set_fetch_fn_for_attn_metadata(fetch_fdv2_attn_metadata)
        self.diffusion_block_size = config.diffusion_block_size
        self.mask_token_id = config.mask_token_id
        
        super().__init__(config, rank, event)

    def prepare_prefill(self, seqs: list[FDV2Sequence]):
        input_ids: list[int] = []
        positions: list[int] = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping: list[int] = []
        block_tables = None
        context_lens: list[int] = []

        for seq in seqs:
            seq.init_diffusion_blocks()

            total_seqlen = len(seq)
            input_ids.extend(seq[seq.cached_num_tokens:])
            positions.extend(range(seq.cached_num_tokens, total_seqlen))
            context_lens.append(0)

            seqlen_q = total_seqlen - seq.cached_num_tokens
            seqlen_k = total_seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:
                continue
            has_padding_mask = seq.pad_prefix_len > 0
            for i in range(0, seq.num_prefix_blocks):
                if seq.block_cache_missed[i]:
                    if has_padding_mask and i == seq.num_prefix_blocks - 1:
                        slot_mapping.extend([-1] * self.block_size)
                    else:
                        start = seq.block_table[i] * self.block_size
                        if i != seq.num_prefix_blocks - 1:
                            end = start + self.block_size
                        else:
                            end = start + seq.prefix_last_block_num_tokens
                        slot_mapping.extend(range(start, end))
                else:
                    slot_mapping.extend([-1] * self.block_size)

        block_tables = self.prepare_block_tables(seqs)
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_tensor = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        context_lens_tensor = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q_tensor = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k_tensor = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        set_fdv2_attn_metadata(
            True,
            cu_seqlens_q=cu_seqlens_q_tensor,
            cu_seqlens_k=cu_seqlens_k_tensor,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            block_tables=block_tables,
            diffusion_block_size=self.diffusion_block_size,
            kv_cache_layout=self.config.kv_cache_layout,
            attn_type="block_attention",
            decode_mode="static",
        )
        return input_ids_tensor, positions_tensor

    def prepare_decode(self, seqs: list[FDV2Sequence]):
        input_ids: list[int] = []
        positions: list[int] = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        slot_mapping: list[int] = []
        context_lens: list[int] = []
        need_kv_cache_store = False
        max_seqlen_q = 0
        max_seqlen_k = 0

        for seq in seqs:
            seq.next_diffusion_step()
            
            cur_input_ids, cur_positions, cur_context_len = seq.diffusion_decoding_inputs()

            input_ids.extend(cur_input_ids)
            positions.extend(cur_positions)
            context_lens.append(cur_context_len)

            seqlen_q = self.diffusion_block_size
            seqlen_k = self.diffusion_block_size
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            if seq.diffusion_blocks[-1].is_active:
                slot_mapping.extend([-1] * self.diffusion_block_size)
            elif seq.diffusion_blocks[-1].is_to_cache:
                need_kv_cache_store = True
                num_pages_storing = seq.num_page_blocks_in_active_diffusion_block
                total_num_pages = len(seq.block_table)
                for i in range(0, num_pages_storing):
                    start = seq.block_table[(total_num_pages - 1) - num_pages_storing + i] * self.block_size
                    end = start + self.block_size
                    slot_mapping.extend(range(start, end))
                
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_tensor = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q_tensor = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k_tensor = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens_tensor = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_fdv2_attn_metadata(
            False,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            cu_seqlens_q=cu_seqlens_q_tensor,
            cu_seqlens_k=cu_seqlens_k_tensor,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            block_tables=block_tables,
            page_block_size=self.config.kvcache_block_size,
            diffusion_block_size=self.diffusion_block_size,
            kv_cache_layout=self.config.kv_cache_layout,
            need_kv_cache_store=need_kv_cache_store,
        )
        return input_ids_tensor, positions_tensor

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512 * self.diffusion_block_size:
            return self.model.compute_logits(self.model(input_ids, positions))
        num_tokens = input_ids.size(0)
        attn_metadata = fetch_fdv2_attn_metadata()
        graph = self.graphs[next(x for x in self.graph_bs if x >= num_tokens)]
        graph_vars = self.graph_vars
        for key, value in graph_vars.items():
            if key != "outputs":
                value.zero_()
        
        num_seqs = len(attn_metadata.context_lens)
        graph_vars["input_ids"][:num_tokens] = input_ids
        graph_vars["positions"][:num_tokens] = positions
        graph_vars["slot_mapping"][:num_tokens] = attn_metadata.slot_mapping
        graph_vars["context_lens"][:num_seqs] = attn_metadata.context_lens
        graph_vars["cu_seqlens_q"][:num_seqs + 1] = attn_metadata.cu_seqlens_q
        graph_vars["cu_seqlens_k"][:num_seqs + 1] = attn_metadata.cu_seqlens_k
        graph_vars["block_tables"][:num_seqs, : attn_metadata.block_tables.size(1)] = attn_metadata.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:num_tokens])

    def run(self, seqs: list[SequenceBase], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        sample_output = self.sampler(seqs, logits, temperatures) if self.rank == 0 else None
        reset_fdv2_attn_metadata()
        return sample_output

    @torch.inference_mode()
    def capture_cudagraph(self):
        set_warming_up(True)
        config = self.config
        hf_config = config.hf_config
        max_num_seqs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        diffusion_block_size = self.diffusion_block_size
        
        max_num_tokens = max_num_seqs * diffusion_block_size
        
        input_ids = torch.zeros(max_num_tokens, dtype=torch.int64)
        positions = torch.zeros(max_num_tokens, dtype=torch.int64)
        slot_mapping = torch.zeros(max_num_tokens, dtype=torch.int32)
        context_lens = torch.zeros(max_num_seqs, dtype=torch.int32)
        block_tables = torch.zeros(max_num_seqs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_num_tokens, hf_config.hidden_size)
        
        cu_seqlens_q = torch.zeros(max_num_seqs + 1, dtype=torch.int32)
        for i in range(max_num_seqs + 1):
            cu_seqlens_q[i] = i * diffusion_block_size
        
        cu_seqlens_k = torch.zeros(max_num_seqs + 1, dtype=torch.int32)
        for i in range(max_num_seqs + 1):
            cu_seqlens_k[i] = i * config.max_model_len
        
        self.graph_bs = []
        seq_bs_list = [1, 2, 4, 8] + list(range(16, max_num_seqs + 1, 16))
        for num_seqs in seq_bs_list:
            self.graph_bs.append(num_seqs * diffusion_block_size)
        self.graphs = {}
        self.graph_pool = None
        
        for num_tokens in tqdm(reversed(self.graph_bs), desc="Capturing CUDA graphs"):
            num_seqs = num_tokens // diffusion_block_size
            graph = torch.cuda.CUDAGraph()
            
            set_fdv2_attn_metadata(
                False,
                slot_mapping=slot_mapping[:num_tokens],
                context_lens=context_lens[:num_seqs],
                cu_seqlens_q=cu_seqlens_q[:num_seqs + 1],
                cu_seqlens_k=cu_seqlens_k[:num_seqs + 1],
                max_seqlen_q=diffusion_block_size,
                max_seqlen_k=config.max_model_len,
                block_tables=block_tables[:num_seqs],
                diffusion_block_size=diffusion_block_size,
                kv_cache_layout=self.config.kv_cache_layout,
                need_kv_cache_store=True,
            )
            
            outputs[:num_tokens] = self.model(input_ids[:num_tokens], positions[:num_tokens])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:num_tokens] = self.model(input_ids[:num_tokens], positions[:num_tokens])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[num_tokens] = graph
            torch.cuda.synchronize()
            reset_fdv2_attn_metadata()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_tables=block_tables,
            outputs=outputs,
        )
        reset_warming_up()