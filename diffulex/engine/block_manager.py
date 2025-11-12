import xxhash

import numpy as np

from collections import deque
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Deque, Iterable, List, Set

from diffulex.config import Config
from diffulex.engine.sequence import SequenceBase


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    hash: int = -1
    token_ids: List[int] = field(default_factory=list)

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


BlockManagerFactory = Callable[[Config], "BlockManagerBase"]
_NOT_PROVIDED = object()


class BlockManagerBase(ABC):
    def __init__(self, config: Config):
        num_blocks = config.num_kvcache_blocks
        block_size = config.kvcache_block_size
        assert num_blocks > 0
        self.config = config
        self.block_size = block_size
        self.blocks: List[Block] = [Block(block_id=i) for i in range(num_blocks)]
        self.hash_to_block_id: Dict[int, int] = dict()
        self.free_block_ids: Deque[int] = deque(range(num_blocks))
        self.used_block_ids: Set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: List[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _free_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: SequenceBase) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: SequenceBase):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            seq.block_cache_missed.append(cache_miss)
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def free(self, seq: SequenceBase):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._free_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    @abstractmethod
    def can_append(self, seq: SequenceBase) -> bool:
        pass

    @abstractmethod
    def may_append(self, seq: SequenceBase) -> None:
        pass


class AutoBlockManager:
    """Registry-driven factory for block manager implementations."""

    _BLOCK_MANAGER_MAPPING: Dict[str, BlockManagerFactory] = {}
    _DEFAULT_KEY = "__default__"

    @classmethod
    def register(
        cls,
        strategy_name: str,
        factory: BlockManagerFactory | object = _NOT_PROVIDED,
        *,
        aliases: Iterable[str] = (),
        is_default: bool = False,
        exist_ok: bool = False,
    ):
        if not isinstance(strategy_name, str) or not strategy_name:
            raise ValueError("strategy_name must be a non-empty string.")
        if isinstance(aliases, str):
            raise TypeError("aliases must be an iterable of strings, not a single string.")

        def decorator(factory_fn: BlockManagerFactory):
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
    def _register(cls, key: str, factory: BlockManagerFactory, *, exist_ok: bool) -> None:
        if not exist_ok and key in cls._BLOCK_MANAGER_MAPPING and cls._BLOCK_MANAGER_MAPPING[key] is not factory:
            raise ValueError(f"Block manager '{key}' is already registered.")
        cls._BLOCK_MANAGER_MAPPING[key] = factory

    @classmethod
    def unregister(cls, strategy_name: str) -> None:
        cls._BLOCK_MANAGER_MAPPING.pop(strategy_name, None)

    @classmethod
    def available_block_managers(cls) -> tuple[str, ...]:
        return tuple(sorted(k for k in cls._BLOCK_MANAGER_MAPPING if k != cls._DEFAULT_KEY))

    @classmethod
    def from_config(cls, config: Config) -> BlockManagerBase:
        candidates: List[str] = []
        for attr in ("decoding_strategy", "model_type"):
            value = getattr(config, attr, None)
            if isinstance(value, str) and value:
                candidates.append(value)
        candidates.append(cls._DEFAULT_KEY)

        for key in candidates:
            factory = cls._BLOCK_MANAGER_MAPPING.get(key)
            if factory is not None:
                return factory(config)

        available = ", ".join(cls.available_block_managers()) or "<none>"
        raise ValueError(
            "No block manager registered for decoding_strategy="
            f"'{getattr(config, 'decoding_strategy', None)}' or model_type="
            f"'{getattr(config, 'model_type', None)}'. Available block managers: {available}."
        )