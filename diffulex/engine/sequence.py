"""Sequence base class and registry."""

from __future__ import annotations

from copy import copy
from enum import Enum, auto
from itertools import count
from typing import Callable

from diffulex.config import Config
from diffulex.sampling_params import SamplingParams
from diffulex.engine.strategy_registry import DiffulexStrategyRegistry


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class SequenceBase:
    """Minimal base class that tracks prompt tokens and cache bookkeeping."""

    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams = SamplingParams()):
        self.seq_id = next(SequenceBase.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table: list[int] = []
        self.block_cache_missed: list[bool] = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.new_tokens = 0

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, key) -> int:
        return self.token_ids[key]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def prompt_token_ids(self) -> list[int]:
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, index: int) -> list[int]:
        assert 0 <= index < self.num_blocks
        return self.token_ids[index * self.block_size : (index + 1) * self.block_size]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1


SequenceFactory = Callable[[list[int], SamplingParams, Config], SequenceBase]


class AutoSequence(DiffulexStrategyRegistry):
    """Registry-driven factory for sequence implementations."""

    @classmethod
    def create(
        cls,
        config: Config,
        token_ids: list[int],
        sampling_params: SamplingParams = SamplingParams(),
    ) -> SequenceBase:
        cls._MODULE_MAPPING: dict[str, SequenceFactory]
        candidates: list[str] = []
        for attr in ("decoding_strategy",):
            value = getattr(config, attr, None)
            if isinstance(value, str) and value:
                candidates.append(value)
        candidates.append(cls._DEFAULT_KEY)

        for key in candidates:
            factory = cls._MODULE_MAPPING.get(key)
            if factory is not None:
                return factory(token_ids, sampling_params, config)

        available = ", ".join(cls.available_modules()) or "<none>"
        raise ValueError(
            "No sequence registered for decoding_strategy="
            f"'{getattr(config, 'decoding_strategy', None)}'. Available sequences: {available}."
        )