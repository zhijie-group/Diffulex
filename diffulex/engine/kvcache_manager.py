import xxhash

import numpy as np

from typing import Callable
from collections import deque
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from diffulex.config import Config
from diffulex.engine.sequence import SequenceBase
from diffulex.engine.strategy_registry import DiffulexStrategyRegistry


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    hash: int = -1
    token_ids: list[int] = field(default_factory=list)

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class KVCacheManagerBase(ABC):
    def __init__(self, config: Config):
        num_blocks = config.num_kvcache_blocks
        block_size = config.kvcache_block_size
        assert num_blocks > 0
        self.config = config
        self.block_size = block_size
        self.blocks: list[Block] = [Block(block_id=i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
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


KVCacheManagerFactory = Callable[[Config], "KVCacheManagerBase"]


class AutoKVCacheManager(DiffulexStrategyRegistry):
    """Registry-driven factory for block manager implementations."""

    @classmethod
    def from_config(cls, config: Config) -> KVCacheManagerBase:
        cls._MODULE_MAPPING: dict[str, KVCacheManagerFactory]
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
            "No block manager registered for decoding_strategy="
            f"'{getattr(config, 'decoding_strategy', None)}'. Available block managers: {available}."
        )