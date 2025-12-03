from typing import Callable
from collections import deque
from abc import ABC, abstractmethod

from diffulex.config import Config
from diffulex.engine.sequence import SequenceBase
from diffulex.engine.kvcache_manager import AutoKVCacheManager
from diffulex.engine.strategy_registry import DiffulexStrategyRegistry


class SchedulerBase(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = AutoKVCacheManager.from_config(config)
        self.waiting: deque[SequenceBase] = deque()
        self.running: deque[SequenceBase] = deque()

    @abstractmethod
    def is_finished(self) -> bool:
        pass

    @abstractmethod
    def add(self, seq: SequenceBase) -> None:
        pass

    @abstractmethod
    def schedule(self) -> tuple[list[SequenceBase], bool]:
        pass

    @abstractmethod
    def preempt(self, seq: SequenceBase) -> None:
        pass

    @abstractmethod
    def postprocess(self, seqs: list[SequenceBase], sampler_output):
        pass


SchedulerFactory = Callable[[Config], "SchedulerBase"]


class AutoScheduler(DiffulexStrategyRegistry):
    """Registry-driven factory for scheduler implementations."""

    @classmethod
    def from_config(cls, config: Config) -> SchedulerBase:
        cls._MODULE_MAPPING: dict[str, SchedulerFactory]
        candidates: list[str] = []
        for attr in ("decoding_strategy",):
            value = getattr(config, attr, None)
            if isinstance(value, str) and value:
                candidates.append(value)
        candidates.append(cls._DEFAULT_KEY)

        for key in candidates:
            factory = cls._MODULE_MAPPING.get(key)
            if factory is not None:
                return factory(config)

        available = ", ".join(cls.available_modules()) or "<none>"
        raise ValueError(
            "No scheduler registered for decoding_strategy="
            f"'{getattr(config, 'decoding_strategy', None)}'. Available schedulers: {available}."
        )