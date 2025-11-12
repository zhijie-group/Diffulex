from collections import deque
from abc import ABC, abstractmethod
from typing import Callable, Deque, Dict, Iterable, List, Tuple

from diffulex.config import Config
from diffulex.engine.sequence import SequenceBase
from diffulex.engine.block_manager import AutoBlockManager


class SchedulerBase(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = AutoBlockManager.from_config(config)
        self.waiting: Deque[SequenceBase] = deque()
        self.running: Deque[SequenceBase] = deque()

    @abstractmethod
    def is_finished(self) -> bool:
        pass

    @abstractmethod
    def add(self, seq: SequenceBase) -> None:
        pass

    @abstractmethod
    def schedule(self) -> Tuple[List[SequenceBase], bool]:
        pass

    @abstractmethod
    def preempt(self, seq: SequenceBase) -> None:
        pass

    @abstractmethod
    def postprocess(self, seqs: List[SequenceBase], sampler_output):
        pass


SchedulerFactory = Callable[[Config], "SchedulerBase"]
_NOT_PROVIDED = object()


class AutoScheduler:
    """Registry-driven factory for scheduler implementations."""

    _SCHEDULER_MAPPING: Dict[str, SchedulerFactory] = {}
    _DEFAULT_KEY = "__default__"

    @classmethod
    def register(
        cls,
        strategy_name: str,
        factory: SchedulerFactory | object = _NOT_PROVIDED,
        *,
        aliases: Iterable[str] = (),
        is_default: bool = False,
        exist_ok: bool = False,
    ):
        if not isinstance(strategy_name, str) or not strategy_name:
            raise ValueError("strategy_name must be a non-empty string.")
        if isinstance(aliases, str):
            raise TypeError("aliases must be an iterable of strings, not a single string.")

        def decorator(factory_fn: SchedulerFactory):
            cls._register(strategy_name, factory_fn, exist_ok=exist_ok)
            for alias in dict.fromkeys(aliases):
                if not isinstance(alias, str) or not alias:
                    raise ValueError("aliases must contain non-empty strings.")
                cls._register(alias, factory_fn, exist_ok=exist_ok)
            if is_default:
                cls._register(cls._DEFAULT_KEY, factory_fn, exist_ok=True)
            return factory_fn

        if factory is _NOT_PROVIDED:
            return decorator
        return decorator(factory)

    @classmethod
    def _register(cls, key: str, factory: SchedulerFactory, *, exist_ok: bool) -> None:
        if not exist_ok and key in cls._SCHEDULER_MAPPING and cls._SCHEDULER_MAPPING[key] is not factory:
            raise ValueError(f"Scheduler '{key}' is already registered.")
        cls._SCHEDULER_MAPPING[key] = factory

    @classmethod
    def unregister(cls, strategy_name: str) -> None:
        cls._SCHEDULER_MAPPING.pop(strategy_name, None)

    @classmethod
    def available_schedulers(cls) -> tuple[str, ...]:
        return tuple(sorted(k for k in cls._SCHEDULER_MAPPING if k != cls._DEFAULT_KEY))

    @classmethod
    def from_config(cls, config: Config) -> SchedulerBase:
        candidates: List[str] = []
        for attr in ("decoding_strategy", "model_type"):
            value = getattr(config, attr, None)
            if isinstance(value, str) and value:
                candidates.append(value)
        candidates.append(cls._DEFAULT_KEY)

        for key in candidates:
            factory = cls._SCHEDULER_MAPPING.get(key)
            if factory is not None:
                return factory(config)

        available = ", ".join(cls.available_schedulers()) or "<none>"
        raise ValueError(
            "No scheduler registered for decoding_strategy="
            f"'{getattr(config, 'decoding_strategy', None)}' or model_type="
            f"'{getattr(config, 'model_type', None)}'. Available schedulers: {available}."
        )