from __future__ import annotations

from typing import TYPE_CHECKING, list

from diffulex.config import Config
from diffulex.engine.kvcache_manager import AutoKVCacheManager, KVCacheManagerBase

if TYPE_CHECKING:
    from .sequence import BlockDiffusionSequence


@AutoKVCacheManager.register("block_diffusion", is_default=True)
class BlockDiffusionKVCacheManager(KVCacheManagerBase):
    def __init__(self, config: Config):
        super().__init__(config)

    def can_append(self, seq: "BlockDiffusionSequence") -> bool:
        required = 1 if seq.cached_or_caching_num_tokens % self.block_size == 1 else 0
        return len(self.free_block_ids) >= required

    def may_append(self, seq: "BlockDiffusionSequence") -> None:
        if seq.cached_or_caching_num_tokens == 0:
            return
        block_table = seq.block_table
        if not block_table:
            return
        last_block = self.blocks[block_table[-1]]
        if seq.cached_or_caching_num_tokens // self.block_size == len(seq.block_table):
            if last_block.hash == -1:
                prev_end_token = seq.cached_or_caching_num_tokens - seq.caching_num_tokens - 1
                prev_block_idx = prev_end_token // self.block_size
                if prev_block_idx < seq.num_blocks:
                    token_ids: list[int] = seq.block(prev_block_idx)
                    prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                    h = self.compute_hash(token_ids, prefix)
                    last_block.update(h, token_ids)
                    self.hash_to_block_id[h] = last_block.block_id
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
