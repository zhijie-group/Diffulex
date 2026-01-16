from __future__ import annotations

from typing import TYPE_CHECKING

from diffulex.config import Config
from diffulex.engine.kvcache_manager import AutoKVCacheManager, KVCacheManagerBase

if TYPE_CHECKING:
    from .sequence import D2FSequence


@AutoKVCacheManager.register("d2f", is_default=True)
class D2FKVCacheManager(KVCacheManagerBase):
    def __init__(self, config: Config):
        super().__init__(config)

    def _required_kv_blocks(self, seq: "D2FSequence") -> int:
        """How many KV-cache blocks this sequence needs *now* for cached+to-cache tokens.

        NOTE: In diffusion decoding, a single decode step may move multiple tokens into
        "to_cache", which can cross multiple KV blocks. So we must ensure block_table
        is large enough for all cached_or_caching tokens, not just append one block.
        """
        n = seq.cached_or_caching_num_tokens
        if n <= 0:
            return 0
        # Need enough blocks to cover token indices [0, n-1].
        return (n + self.block_size - 1) // self.block_size

    def can_append(self, seq: "D2FSequence") -> bool:
        # We may need to allocate multiple blocks in one step (cached_or_caching can jump).
        required = self._required_kv_blocks(seq)
        missing = max(0, required - len(seq.block_table))
        return len(self.free_block_ids) >= missing

    def may_append(self, seq: "D2FSequence") -> None:
        if seq.cached_or_caching_num_tokens == 0:
            return
        block_table = seq.block_table
        if not block_table:
            # Defensive: allocate() should have populated it for prefill/prompt, but don't crash here.
            return

        required = self._required_kv_blocks(seq)
        # Allocate enough KV blocks to cover all cached_or_caching tokens.
        while len(block_table) < required:
            last_block = self.blocks[block_table[-1]]
            # Preserve the existing "finalize previous block hash" behavior before moving on.
            if last_block.hash == -1:
                prev_end_token = seq.cached_or_caching_num_tokens - seq.caching_num_tokens - 1
                prev_block_idx = prev_end_token // self.block_size
                if prev_block_idx < seq.num_blocks:
                    token_ids: list[int] = seq.block(prev_block_idx)
                    prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                    h = self.compute_hash(token_ids, prefix)
                    last_block.update(h, token_ids)
                    self.hash_to_block_id[h] = last_block.block_id

            if not self.free_block_ids:
                raise RuntimeError(
                    "D2FKVCacheManager: insufficient free KV cache blocks to append: "
                    f"required={required}, current_len={len(block_table)}, "
                    f"cached_or_caching_num_tokens={seq.cached_or_caching_num_tokens}, "
                    f"block_size={self.block_size}."
                )

            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)