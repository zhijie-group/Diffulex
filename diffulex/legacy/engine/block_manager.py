import xxhash

import numpy as np

from collections import deque
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Deque, Set

from diffulex.legacy.config import Config
from diffulex.legacy.engine.sequence import SequenceBase, SequenceForCausalLM, SequenceForDiffusionLM


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


class BlockManagerBase(ABC):
    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
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
    def may_append(self, seq: SequenceBase):
        pass
            

class BlockManagerForCausalLM(BlockManagerBase):
    def can_append(self, seq: SequenceForCausalLM) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
    
    def may_append(self, seq: SequenceBase):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1

class BlockManagerForDiffusionLM(BlockManagerBase):
    def can_append(self, seq: SequenceForDiffusionLM) -> bool:
        return len(self.free_block_ids) >= (seq.cached_or_caching_num_tokens % self.block_size == 1)
    
    def may_append(self, seq: SequenceForDiffusionLM):
        # Handle edge case when no tokens are cached yet
        if seq.cached_or_caching_num_tokens == 0:
            return
            
        block_table = seq.block_table
        if not block_table:
            return
            
        last_block = self.blocks[block_table[-1]]
        
        if seq.cached_or_caching_num_tokens // self.block_size == len(seq.block_table):
            if last_block.hash == -1:
                prev_block_end_token = seq.cached_or_caching_num_tokens - seq.caching_num_tokens - 1  # 256th token (0-indexed: 255)
                prev_block_idx = prev_block_end_token // self.block_size  # block containing 255th token
                
                if prev_block_idx < seq.num_blocks:
                    # This block should be full, so set its hash
                    token_ids = seq.block(prev_block_idx)
                    prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                    h = self.compute_hash(token_ids, prefix)
                    last_block.update(h, token_ids)
                    self.hash_to_block_id[h] = last_block.block_id
            
            # Now allocate a new block
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)


class AutoBlockManager(BlockManagerBase):
    BLOCK_MANAGER_MAPPING = {
        "causal_lm": BlockManagerForCausalLM,
        "diffusion_lm": BlockManagerForDiffusionLM,
    }
    @classmethod
    def from_config(cls, config: Config) -> BlockManagerBase:
        block_manager_cls = cls.BLOCK_MANAGER_MAPPING.get(config.model_type)
        if not block_manager_cls:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        return block_manager_cls(config.num_kvcache_blocks, config.kvcache_block_size)