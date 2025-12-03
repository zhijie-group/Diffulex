from __future__ import annotations

import torch

from dataclasses import dataclass
from enum import Enum, auto

from diffulex.config import Config
from diffulex.engine.sequence import AutoSequence, SequenceBase
from diffulex.sampling_params import SamplingParams


class D2FDiffusionBlockStatus(Enum):
    ACTIVE = auto()
    TO_CACHE = auto()
    IN_CACHE = auto()


@dataclass
class D2FDiffusionBlock:
    block_id: int = 0
    status: D2FDiffusionBlockStatus = D2FDiffusionBlockStatus.ACTIVE

    global_start_id: int = 0
    global_end_id: int | None = None
    cursor: int = 0

    mask_token_id: int = 151666
    size: int = 32
    is_prompt: bool = False

    accept_threshold: float = 0.95
    add_new_block_threshold: float = 0.1
    complete_threshold: float = 0.9

    seq: "D2FSequence" | None = None
    pre_block: "D2FDiffusionBlock" | None = None
    suf_block: "D2FDiffusionBlock" | None = None

    def __post_init__(self) -> None:
        self.global_end_id = self.global_start_id + self.size

    def __getitem__(self, key: int) -> int:
        return self.seq[self.global_start_id + key]  # type: ignore[index]

    def __len__(self) -> int:
        return self.size

    @property
    def current_complete_ratio(self) -> float:
        if self.size == 0:
            return 0.0
        return sum(token_id != self.mask_token_id for token_id in self.token_ids) / self.size

    @property
    def available_to_cache(self) -> bool:
        return self.current_complete_ratio == 1.0

    @property
    def is_active(self) -> bool:
        return self.status == D2FDiffusionBlockStatus.ACTIVE

    @property
    def is_in_cache(self) -> bool:
        return self.status == D2FDiffusionBlockStatus.IN_CACHE

    @property
    def is_to_cache(self) -> bool:
        return self.status == D2FDiffusionBlockStatus.TO_CACHE

    @property
    def pre_block_complete(self) -> bool:
        if self.pre_block is None:
            return True
        return self.pre_block.current_complete_ratio >= self.complete_threshold

    @property
    def add_new_block(self) -> bool:
        return self.current_complete_ratio >= self.add_new_block_threshold

    @property
    def token_ids(self) -> list[int]:
        if self.seq is None:
            raise RuntimeError("Diffusion block is not attached to a sequence.")
        return self.seq.token_ids[self.global_start_id : self.global_end_id]

    @property
    def local_mask_tokens(self) -> list[bool]:
        return [token_id == self.seq.mask_token_id for token_id in self.token_ids]  # type: ignore[arg-type]

    @property
    def local_mask_token_ids(self) -> list[int]:
        return [idx for idx, is_mask in enumerate(self.local_mask_tokens) if is_mask]

    @property
    def global_mask_token_ids(self) -> list[int]:
        if self.seq is None:
            return []
        offset = self.global_start_id
        in_cache_blocks = list(range(sum(self.seq.in_cache_blocks)))
        offset -= sum(self.seq.diffusion_blocks[block_id].size for block_id in in_cache_blocks)
        return [mask_id + offset for mask_id in self.local_mask_token_ids]

    def remaining_length(self, start_idx: int) -> int:
        return self.size - self.cursor

    def to_cache(self) -> None:
        if self.available_to_cache and not self.is_in_cache:
            self.status = D2FDiffusionBlockStatus.TO_CACHE

    def in_cache(self) -> None:
        if self.is_to_cache:
            self.status = D2FDiffusionBlockStatus.IN_CACHE

    def modify_token(self, local_token_id: int, modified_to: int) -> None:
        if self.seq is None:
            raise RuntimeError("Diffusion block is not attached to a sequence.")
        target_id = local_token_id + self.global_start_id
        assert self.seq.token_ids[target_id] == self.mask_token_id
        self.seq.token_ids[target_id] = modified_to.item()  # type: ignore[assignment]
        self.seq.new_tokens += 1


@AutoSequence.register("d2f", is_default=True)
class D2FSequence(SequenceBase):
    """Sequence implementation tailored for diffusion-based decoding."""

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams = SamplingParams(),
        config: Config | None = None,
    ):
        super().__init__(token_ids, sampling_params)
        if config is None:
            raise ValueError("SequenceForDiffusionLM requires a Config instance.")
        self.config = config
        self.decoding_strategy = config.decoding_strategy
        self.kv_cache_layout = config.kv_cache_layout
        self.eos_token_id = config.eos
        self.max_model_len = config.max_model_len
        self.mask_token_id = config.mask_token_id
        self.diffusion_block_size = config.diffusion_block_size
        self.block_mask: torch.Tensor | None = None
        self.meet_eos = False
        self.diffusion_blocks: list[D2FDiffusionBlock] = []
        self.n_steps = 0
        self.input_token_ids: list[int] = []
        self.input_num_tokens = 0
        self.input_num_prompt_tokens = 0

    def __repr__(self) -> str:
        return (
            "SequenceForDiffusionLM(seq_id={seq_id}, status={status}, num_tokens={num_tokens}, "
            "num_prompt_tokens={num_prompt_tokens}, num_cached_tokens={num_cached_tokens}, "
            "diffusion_block_size={diffusion_block_size}, mask_shape={mask_shape})"
        ).format(
            seq_id=self.seq_id,
            status=self.status.name,
            num_tokens=self.num_tokens,
            num_prompt_tokens=self.num_prompt_tokens,
            num_cached_tokens=self.num_cached_tokens,
            diffusion_block_size=self.diffusion_block_size,
            mask_shape=self.block_mask.shape if self.block_mask is not None else None,
        )

    def __getstate__(self):
        diffusion_blocks_state = []
        for block in self.diffusion_blocks:
            diffusion_blocks_state.append(
                {
                    "block_id": block.block_id,
                    "status": block.status,
                    "global_start_id": block.global_start_id,
                    "global_end_id": block.global_end_id,
                    "cursor": block.cursor,
                    "mask_token_id": block.mask_token_id,
                    "size": block.size,
                    "is_prompt": block.is_prompt,
                    "accept_threshold": block.accept_threshold,
                    "add_new_block_threshold": block.add_new_block_threshold,
                    "complete_threshold": block.complete_threshold,
                }
            )

        state = {
            "seq_id": self.seq_id,
            "status": self.status,
            "token_ids": self.token_ids,
            "last_token": self.last_token,
            "num_tokens": self.num_tokens,
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_cached_tokens": self.num_cached_tokens,
            "block_table": self.block_table,
            "block_cache_missed": self.block_cache_missed,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "ignore_eos": self.ignore_eos,
            "config": self.config,
            "decoding_strategy": self.decoding_strategy,
            "kv_cache_layout": self.kv_cache_layout,
            "eos_token_id": self.eos_token_id,
            "max_model_len": self.max_model_len,
            "mask_token_id": self.mask_token_id,
            "diffusion_block_size": self.diffusion_block_size,
            "diffusion_blocks_state": diffusion_blocks_state,
            "input_token_ids": self.input_token_ids,
            "input_num_tokens": self.input_num_tokens,
            "input_num_prompt_tokens": self.input_num_prompt_tokens,
            "new_tokens": self.new_tokens,
            "block_mask": self.block_mask,
            "meet_eos": self.meet_eos,
            "n_steps": self.n_steps,
        }
        return state

    def __setstate__(self, state):
        self.seq_id = state["seq_id"]
        self.status = state["status"]
        self.token_ids = state["token_ids"]
        self.last_token = state["last_token"]
        self.num_tokens = state["num_tokens"]
        self.num_prompt_tokens = state["num_prompt_tokens"]
        self.num_cached_tokens = state["num_cached_tokens"]
        self.block_table = state["block_table"]
        self.block_cache_missed = state["block_cache_missed"]
        self.temperature = state["temperature"]
        self.max_tokens = state["max_tokens"]
        self.ignore_eos = state["ignore_eos"]
        self.meet_eos = state["meet_eos"]

        self.config = state["config"]
        self.decoding_strategy = state.get("decoding_strategy", getattr(self.config, "decoding_strategy", None))
        self.kv_cache_layout = state.get("kv_cache_layout", getattr(self.config, "kv_cache_layout", None))
        self.eos_token_id = state["eos_token_id"]
        self.max_model_len = state["max_model_len"]
        self.mask_token_id = state["mask_token_id"]
        self.diffusion_block_size = state["diffusion_block_size"]

        self.input_token_ids = state.get("input_token_ids", [])
        self.input_num_tokens = state.get("input_num_tokens", 0)
        self.input_num_prompt_tokens = state.get("input_num_prompt_tokens", 0)
        self.new_tokens = state.get("new_tokens", 0)
        self.block_mask = state.get("block_mask")
        self.n_steps = state.get("n_steps", 0)

        if self.block_mask is not None and self.block_mask.device.index != torch.cuda.current_device():
            self.block_mask = self.block_mask.to(torch.cuda.current_device())

        self.diffusion_blocks = []
        pre_block = None
        for block_state in state["diffusion_blocks_state"]:
            block = D2FDiffusionBlock(
                block_id=block_state["block_id"],
                status=block_state["status"],
                global_start_id=block_state["global_start_id"],
                global_end_id=block_state["global_end_id"],
                cursor=block_state.get("cursor", 0),
                mask_token_id=block_state["mask_token_id"],
                size=block_state["size"],
                is_prompt=block_state["is_prompt"],
                accept_threshold=block_state.get("accept_threshold", 0.95),
                add_new_block_threshold=block_state.get("add_new_block_threshold", 0.1),
                complete_threshold=block_state.get("complete_threshold", 0.9),
                seq=self,
                pre_block=pre_block,
            )
            if pre_block is not None:
                pre_block.suf_block = block
            self.diffusion_blocks.append(block)
            pre_block = block

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.input_num_tokens

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.input_num_prompt_tokens :]

    @property
    def active_blocks(self) -> list[bool]:
        return [block.is_active for block in self.diffusion_blocks]

    @property
    def to_cache_blocks(self) -> list[bool]:
        return [block.is_to_cache for block in self.diffusion_blocks]

    @property
    def in_cache_blocks(self) -> list[bool]:
        return [block.is_in_cache for block in self.diffusion_blocks]

    @property
    def num_prompt_blocks(self) -> int:
        return (self.input_num_prompt_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_prompt_num_tokens(self) -> int:
        return self.input_num_prompt_tokens - (self.num_prompt_blocks - 1) * self.block_size

    @property
    def updated_or_updating_kv_cache_block_ids(self) -> list[int]:
        return [idx for idx, caching in enumerate(self.caching_blocks) if caching]

    @property
    def caching_blocks(self) -> list[bool]:
        return [to_cache or in_cache for to_cache, in_cache in zip(self.to_cache_blocks, self.in_cache_blocks)]

    @property
    def cached_block_ids(self) -> list[int]:
        return [idx for idx, in_cache in enumerate(self.in_cache_blocks) if in_cache]

    @property
    def mask_tokens(self) -> list[bool]:
        return [token_id == self.mask_token_id for token_id in self.token_ids]

    @property
    def caching_num_tokens(self) -> int:
        return sum(block.size for block in self.diffusion_blocks if block.is_to_cache)

    @property
    def cached_or_caching_last_token_id(self) -> int:
        cached_num_tokens = 0
        for block_id in self.updated_or_updating_kv_cache_block_ids:
            block = self.diffusion_blocks[block_id]
            cached_num_tokens += block.size
        return max(cached_num_tokens - 1, 0)

    @property
    def cached_or_caching_num_tokens(self) -> int:
        return self.cached_or_caching_last_token_id + 1

    @property
    def cached_num_tokens(self) -> int:
        return sum(block.size for block in self.diffusion_blocks if block.is_in_cache)

    @property
    def num_cached_blocks(self) -> int:
        return (self.num_cached_tokens + self.block_size - 1) // self.block_size

    @property
    def diffusion_num_tokens(self) -> int:
        return sum(self.mask_tokens)

    @property
    def mem_block_to_diffusion_blocks_map(self) -> list[list[int]]:
        mapping = []
        for block_id in range(self.num_blocks):
            window_start = block_id * self.block_size
            window_length = self.block_size if block_id < self.num_blocks - 1 else self.last_block_num_tokens
            mapping.append(
                [self.token_to_diffusion_block_id(token_id) for token_id in range(window_start, window_start + window_length)]
            )
        return mapping

    def token_to_diffusion_block_id(self, token_id: int) -> int:
        if token_id < self.input_num_tokens:
            return 0
        return (token_id - self.input_num_tokens) // self.diffusion_block_size + 1

    @property
    def num_diffusion_blocks(self) -> int:
        return len(self.diffusion_blocks)

    def diffusion_decoding_inputs(self) -> tuple[list[int], list[int], int]:
        to_cache_and_active_blocks = self.diffusion_blocks[self.cached_block_ids[-1] + 1 :]
        assert len(to_cache_and_active_blocks) == sum(self.active_blocks) + sum(self.to_cache_blocks)

        input_tokens: list[int] = []
        positions: list[int] = []
        context_len = sum(self.diffusion_blocks[block_id].size for block_id in self.cached_block_ids)
        temp_context_len = context_len
        for block in to_cache_and_active_blocks:
            input_tokens.extend(block.token_ids)
            positions.extend(range(temp_context_len, temp_context_len + block.size))
            temp_context_len += block.size

        return input_tokens, positions, context_len

    def reset_new_tokens(self) -> None:
        self.new_tokens = 0

    def post_process(self) -> None:
        for block in self.diffusion_blocks:
            block.cursor = 0
            if block.is_in_cache:
                continue
            if block.is_to_cache:
                block.in_cache()
            elif block.is_active:
                if block.available_to_cache:
                    block.to_cache()
                else:
                    break

    def set_layout(self, layout: str) -> None:
        self.kv_cache_layout = layout

    @property
    def current_block_mask(self) -> torch.Tensor:
        if self.block_mask is None:
            raise RuntimeError("Block mask not initialized.")
        if self.kv_cache_layout == "distinct":
            return self.block_mask[..., self.cached_num_tokens :, self.cached_num_tokens :]
        return self.block_mask[..., self.cached_num_tokens :, :]

    def update_block_mask(self, is_prefill: bool = False) -> None:
        if is_prefill:
            num_tokens = self.num_tokens
            mask_shape = (1, 1, num_tokens, num_tokens)
            block_mask = torch.zeros(mask_shape, dtype=torch.bool, device=torch.cuda.current_device())
            block_mask[..., : self.input_num_tokens, : self.input_num_tokens] = True
            num_diffusion_blocks = (
                self.num_tokens - self.input_num_tokens + self.diffusion_block_size - 1
            ) // self.diffusion_block_size
            for block_id in range(num_diffusion_blocks):
                start_h = self.input_num_tokens + block_id * self.diffusion_block_size
                end_h = start_h + self.diffusion_block_size
                block_mask[..., start_h:end_h, :end_h] = True
            self.block_mask = block_mask.clone()
            return

        if self.block_mask is None:
            raise RuntimeError("Prefill block mask must be created before decode updates.")
        dev = self.block_mask.device
        left_shape = (1, 1, self.num_tokens - self.diffusion_block_size, self.diffusion_block_size)
        down_shape = (1, 1, self.diffusion_block_size, self.num_tokens)
        left_cat_tensor = torch.zeros(left_shape, dtype=torch.bool, device=dev)
        down_cat_tensor = torch.ones(down_shape, dtype=torch.bool, device=dev)
        self.block_mask = torch.cat([self.block_mask, left_cat_tensor], dim=-1)
        self.block_mask = torch.cat([self.block_mask, down_cat_tensor], dim=-2)

    def next_diffusion_step(self, is_prefill: bool = False) -> None:
        self.n_steps += 1
        if is_prefill:
            self.input_token_ids = self.token_ids.copy()
            self.input_num_tokens = self.num_tokens
            self.input_num_prompt_tokens = self.num_prompt_tokens
            self.num_prompt_tokens += self.diffusion_block_size
            self.diffusion_blocks.append(
                D2FDiffusionBlock(
                    block_id=len(self.diffusion_blocks),
                    status=D2FDiffusionBlockStatus.TO_CACHE,
                    global_start_id=0,
                    mask_token_id=self.mask_token_id,
                    size=len(self.input_token_ids),
                    accept_threshold=self.config.accept_threshold,
                    add_new_block_threshold=self.config.add_new_block_threshold,
                    complete_threshold=self.config.complete_threshold,
                    is_prompt=True,
                    seq=self,
                )
            )

        if not self.diffusion_blocks:
            return

        if self.diffusion_blocks[-1].add_new_block and not self.meet_eos:
            remaining = self.max_model_len - self.num_tokens
            if remaining <= 0:
                return
            added_num_tokens = min(self.diffusion_block_size, remaining)
            diffusion_seq = [self.mask_token_id] * added_num_tokens
            current_block = D2FDiffusionBlock(
                block_id=len(self.diffusion_blocks),
                status=D2FDiffusionBlockStatus.ACTIVE,
                global_start_id=self.num_tokens,
                mask_token_id=self.mask_token_id,
                size=added_num_tokens,
                accept_threshold=self.config.accept_threshold,
                add_new_block_threshold=self.config.add_new_block_threshold,
                complete_threshold=self.config.complete_threshold,
                seq=self,
                pre_block=self.diffusion_blocks[-1],
            )
            self.diffusion_blocks[-1].suf_block = current_block
            self.token_ids += diffusion_seq
            self.num_tokens += added_num_tokens
            self.diffusion_blocks.append(current_block)
            self.update_block_mask(is_prefill=is_prefill)