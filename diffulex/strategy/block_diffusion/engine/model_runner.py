from __future__ import annotations

import time
from typing import list
from multiprocessing.synchronize import Event

import torch

from diffulex.config import Config
from diffulex.engine.sequence import SequenceBase
from diffulex.strategy.block_diffusion.engine.sequence import BlockDiffusionSequence
from diffulex.attention.metadata import set_fetch_fn_for_attn_metadata
from diffulex.engine.model_runner import AutoModelRunner, ModelRunnerBase
from diffulex.strategy.block_diffusion.attention.metadata import fetch_block_diffusion_attn_metadata, set_block_diffusion_attn_metadata, reset_block_diffusion_attn_metadata


@AutoModelRunner.register("block_diffusion", is_default=True)
class BlockDiffusionModelRunner(ModelRunnerBase):
    """Reference implementation of Block Diffusion decoding strategy."""

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        super().__init__(config, rank, event)
        self.diffusion_block_size = config.diffusion_block_size
        self.mask_token_id = config.mask_token_id
        self.decoding_strategy = config.decoding_strategy
        set_fetch_fn_for_attn_metadata(fetch_block_diffusion_attn_metadata)

    def warmup_model(self):
        print("Warming up model...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        test_input_ids = [0] * max_model_len
        seqs = [BlockDiffusionSequence(test_input_ids, config=self.config) for _ in range(num_seqs)]
        self.run(seqs, True)
        for seq in seqs:
            seq.post_process()
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = getattr(
            hf_config,
            "num_key_value_heads",
            getattr(hf_config, "n_kv_heads", None),
        ) // self.world_size

        if hasattr(hf_config, "head_dim"):
            head_dim = hf_config.head_dim
        elif hasattr(hf_config, "hidden_size") and hasattr(hf_config, "num_attention_heads"):
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        else:
            raise AttributeError(f"Cannot determine head_dim from config: {type(hf_config)}")

        dtype = (
            hf_config.torch_dtype
            if hasattr(hf_config, "torch_dtype") and hf_config.torch_dtype
            else torch.bfloat16
        )
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * dtype.itemsize
        )
        get_num_kvcache_blocks = (
            lambda gpu_memory_utilization: int(total * gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        try:
            num_kvcache_blocks = get_num_kvcache_blocks(config.gpu_memory_utilization)
            assert num_kvcache_blocks > 0
        except Exception:
            gpu_memory_utilization = config.gpu_memory_utilization
            while num_kvcache_blocks <= 200:
                print(
                    "Warning: GPU memory utilization "
                    f"{gpu_memory_utilization} is too low to allocate kv cache. "
                    "Automatically adding 0.05."
                )
                gpu_memory_utilization += 0.05
                num_kvcache_blocks = get_num_kvcache_blocks(gpu_memory_utilization)
            print(
                f"Set gpu_memory_utilization to {gpu_memory_utilization:.2f} "
                "to allocate kv cache."
            )
            config.gpu_memory_utilization = gpu_memory_utilization

        config.num_kvcache_blocks = num_kvcache_blocks
        print(
            "Allocated {num_blocks} blocks of size {block_size} for kv cache on rank {rank}.".format(
                num_blocks=config.num_kvcache_blocks,
                block_size=self.block_size,
                rank=self.rank,
            )
        )

        if config.kv_cache_layout == "distinct":
            x = config.k_cache_hdim_split_factor_x
            self.k_cache = torch.zeros(
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                num_kv_heads,
                head_dim // x,
                self.block_size,
                x,
            )
            self.v_cache = torch.zeros(
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                num_kv_heads,
                head_dim,
                self.block_size,
            )
            layer_id = 0
            for module in self.model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.k_cache[layer_id]
                    module.v_cache = self.v_cache[layer_id]
                    layer_id += 1
        elif config.kv_cache_layout == "unified":
            self.kv_cache = torch.zeros(
                2,
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
            )
            layer_id = 0
            for module in self.model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.kv_cache[0, layer_id]
                    module.v_cache = self.kv_cache[1, layer_id]
                    layer_id += 1
        else:
            raise ValueError(
                "Unsupported kv_cache_layout: {layout}. Supported values are 'distinct' and 'unified'.".format(
                    layout=config.kv_cache_layout
                )
            )

    def prepare_prefill(self, seqs: list[BlockDiffusionSequence]):
        input_ids: list[int] = []
        positions: list[int] = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping: list[int] = []
        block_tables = None
        context_lens: list[int] = []
        seq_lens: list[int] = []

        for seq in seqs:
            seq.next_diffusion_step(is_prefill=True)

            total_seqlen = len(seq)
            input_ids.extend(seq[seq.cached_num_tokens:])
            positions.extend(range(seq.cached_num_tokens, total_seqlen))
            seq_lens.append(total_seqlen)
            context_lens.append(0)
            assert len(input_ids) == len(positions), (
                "prepare_prefill(diffusion): len(input_ids) {len_ids} != len(positions) {len_pos}".format(
                    len_ids=len(input_ids),
                    len_pos=len(positions),
                )
            )

            seqlen_q = total_seqlen - seq.cached_num_tokens
            seqlen_k = total_seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:
                continue
            for i in range(0, seq.num_prompt_blocks):
                if seq.block_cache_missed[i]:
                    start = seq.block_table[i] * self.block_size
                    if i != seq.num_prompt_blocks - 1:
                        end = start + self.block_size
                    else:
                        end = start + seq.last_block_prompt_num_tokens
                    slot_mapping.extend(range(start, end))
                else:
                    slot_mapping.extend([-1] * self.block_size)
            slot_mapping.extend([-1] * seq.diffusion_block_size)

        block_tables = self.prepare_block_tables(seqs)

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_tensor = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        seq_lens_ts = torch.tensor(seq_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens_tensor = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q_tensor = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k_tensor = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        assert cu_seqlens_q_tensor[-1].item() == input_ids_tensor.numel(), (
            "prepare_prefill(diffusion): cu_seqlens_q[-1]={cq} != num_tokens={nt}".format(
                cq=cu_seqlens_q_tensor[-1].item(),
                nt=input_ids_tensor.numel(),
            )
        )
        assert cu_seqlens_k_tensor[-1].item() == sum(seq_lens), (
            "prepare_prefill(diffusion): cu_seqlens_k[-1]={ck} != sum(seq_lens)={sl}".format(
                ck=cu_seqlens_k_tensor[-1].item(),
                sl=sum(seq_lens),
            )
        )

        set_block_diffusion_attn_metadata(
            True,
            cu_seqlens_q=cu_seqlens_q_tensor,
            cu_seqlens_k=cu_seqlens_k_tensor,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            block_tables=block_tables,
            seqs=seqs,
            kv_cache_layout=self.config.kv_cache_layout,
            seq_lens=seq_lens,
            seq_lens_ts=seq_lens_ts,
        )
        return input_ids_tensor, positions_tensor

    def prepare_decode(self, seqs: list[BlockDiffusionSequence]):
        input_ids: list[int] = []
        positions: list[int] = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        slot_mapping: list[int] = []
        context_lens: list[int] = []
        seq_lens: list[int] = []
        seq_id_to_queue_id: dict[int, int] = {}
        need_kv_cache_store = False
        max_seqlen_q = 0
        max_seqlen_k = 0

        for seq_idx_in_queue, seq in enumerate(seqs):
            seq_id = seq.seq_id
            seq_id_to_queue_id[seq_id] = seq_idx_in_queue
            seq.next_diffusion_step()
            cur_input_ids, cur_positions, cur_context_len = seq.diffusion_decoding_inputs()

            seq_lens.append(len(cur_input_ids))
            input_ids.extend(cur_input_ids)
            positions.extend(cur_positions)
            context_lens.append(cur_context_len)

            total_seqlen = len(seq)
            seqlen_q = total_seqlen - seq.cached_num_tokens
            seqlen_k = total_seqlen
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            mem_block_to_diffusion_blocks_map = seq.mem_block_to_diffusion_blocks_map
            context_len = context_lens[seq_id_to_queue_id[seq_id]]
            for mem_block_idx in range(0, seq.num_blocks):
                start_idx = mem_block_idx * seq.block_size
                end_idx = start_idx + seq.block_size
                cur_map = mem_block_to_diffusion_blocks_map[mem_block_idx]
                is_last_block = False
                meet_active_block = False
                while start_idx < end_idx and not is_last_block and not meet_active_block:
                    local_start_idx = lambda: start_idx % seq.block_size
                    diffusion_block = seq.diffusion_blocks[cur_map[local_start_idx()]]
                    if diffusion_block.block_id == 0 and diffusion_block.cursor != start_idx:
                        diffusion_block.cursor = start_idx
                    if cur_map[local_start_idx()] == seq.num_diffusion_blocks - 1:
                        is_last_block = True

                    def get_step(diff_blk, begin_idx):
                        remaining = diff_blk.remaining_length(begin_idx)
                        if remaining + local_start_idx() <= seq.block_size:
                            return remaining
                        return seq.block_size - local_start_idx()

                    if diffusion_block.is_in_cache:
                        step = get_step(diffusion_block, start_idx)
                        diffusion_block.cursor += step
                        start_idx += step
                    elif diffusion_block.is_to_cache:
                        step = get_step(diffusion_block, start_idx)
                        diffusion_block.cursor += step
                        cur_diffusion_block_start = 0
                        cur_diffusion_block_end = step
                        start_idx += step
                        mem_block_start = (
                            seq.block_table[mem_block_idx] * self.block_size
                            + context_len % seq.block_size
                        )
                        context_len += step
                        slot_mapping.extend(
                            range(
                                mem_block_start + cur_diffusion_block_start,
                                mem_block_start + cur_diffusion_block_end,
                            )
                        )
                        need_kv_cache_store = True
                    elif diffusion_block.is_active:
                        meet_active_block = True

                if meet_active_block:
                    active = seq.active_blocks
                    first_active_idx = next((i for i, v in enumerate(active) if v), None)
                    if first_active_idx is not None:
                        num_blocks_to_pad = len(active) - first_active_idx
                        slot_mapping.extend([-1] * (num_blocks_to_pad * seq.diffusion_block_size))
                    break
            assert len(input_ids) == len(positions), (
                "Input IDs length {len_ids} does not match positions length {len_pos}".format(
                    len_ids=len(input_ids),
                    len_pos=len(positions),
                )
            )
            assert len(input_ids) == len(slot_mapping), (
                "Input IDs length {len_ids} does not match slot mapping length {len_slot}".format(
                    len_ids=len(input_ids),
                    len_slot=len(slot_mapping),
                )
            )

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_tensor = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        seq_lens_ts = torch.tensor(seq_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q_tensor = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k_tensor = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens_tensor = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_block_diffusion_attn_metadata(
            False,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            cu_seqlens_q=cu_seqlens_q_tensor,
            cu_seqlens_k=cu_seqlens_k_tensor,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            block_tables=block_tables,
            seqs=seqs,
            seq_lens=seq_lens,
            seq_lens_ts=seq_lens_ts,
            kv_cache_layout=self.config.kv_cache_layout,
            need_kv_cache_store=need_kv_cache_store,
            block_diffusion_pp=True,
        )
        return input_ids_tensor, positions_tensor

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        bs = input_ids.size(0)
        context = fetch_block_diffusion_attn_metadata()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        for key, value in graph_vars.items():
            if key != "outputs":
                value.zero_()
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run_verbose(self, seqs: list[SequenceBase], is_prefill: bool) -> list[int]:
        print("= =" * 20)
        print(f"Running {'prefill' if is_prefill else 'decode'} for {len(seqs)} sequences on rank {self.rank}")
        start = time.time()
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        print(f"Prepared input in {time.time() - start:.2f} seconds")
        start = time.time()
        logits = self.run_model(input_ids, positions, is_prefill)
        print(f"Ran model in {time.time() - start:.2f} seconds")
        start = time.time()
        sample_output = self.sampler(logits, temperatures) if self.rank == 0 else None
        print(f"Sampled tokens in {time.time() - start:.2f} seconds")
        reset_block_diffusion_attn_metadata()
        return sample_output

    def run(self, seqs: list[SequenceBase], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        sample_output = self.sampler(logits, temperatures) if self.rank == 0 else None
        reset_block_diffusion_attn_metadata()
        return sample_output

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        TODO: Varlen decoding does not support CUDA graph capture yet.
        Can be implemented, but requires drastically high overhead.
        """
        raise NotImplementedError("CUDA graph capture for DiffusionLM is not implemented yet.")
