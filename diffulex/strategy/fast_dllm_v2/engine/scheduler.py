from __future__ import annotations

from diffulex.config import Config
from diffulex.engine.scheduler import AutoScheduler, SchedulerBase
from diffulex.engine.sequence import SequenceStatus
from .sequence import FDV2Sequence


@AutoScheduler.register("fast_dllm_v2", is_default=True)
class FastDLLMV2Scheduler(SchedulerBase):
    def __init__(self, config: Config):
        super().__init__(config)
        self.diffusion_block_size = config.diffusion_block_size

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: FDV2Sequence) -> None:
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[FDV2Sequence], bool]:
        scheduled: list[FDV2Sequence] = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            projected = len(seq) + seq.diffusion_block_size
            if (
                num_batched_tokens + projected > self.max_num_batched_tokens
                or not self.block_manager.can_allocate(seq)
            ):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += projected - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled.append(seq)
        if scheduled:
            return scheduled, True

        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled.append(seq)
        if not scheduled:
            diag = {
                "phase": "decode",
                "waiting": len(self.waiting),
                "running": len(self.running),
                "max_num_seqs": self.max_num_seqs,
                "max_num_batched_tokens": self.max_num_batched_tokens,
                "diffusion_block_size": self.diffusion_block_size,
            }
            candidates = list(self.running)[:3] + list(self.waiting)[:2]
            details = []
            for idx, candidate in enumerate(candidates):
                try:
                    can_append = self.block_manager.can_append(candidate)
                except Exception:
                    can_append = "error"
                details.append(
                    f"[{idx}] status={candidate.status.name}, len={len(candidate)}, "
                    f"diff_block={getattr(candidate, 'diffusion_block_size', '?')}, "
                    f"new_tokens={getattr(candidate, 'new_tokens', '?')}, "
                    f"cached={getattr(candidate, 'num_cached_tokens', '?')}, "
                    f"can_append={can_append}"
                )
            raise RuntimeError(
                "BDScheduler: unable to schedule any sequence in decode; "
                f"state={diag}; details={' | '.join(details)}"
            )
        self.running.extendleft(reversed(scheduled))
        return scheduled, False

    def preempt(self, seq: FDV2Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.block_manager.free(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[FDV2Sequence],
        sample_output,
    ) -> dict[int, int]:
        n_diff_steps: dict[int, int] = {}
        for seq in seqs:
            seq.reset_new_tokens()
            seq_id = str(seq.seq_id)
            true_ids_map = sample_output.true_local_ids_map.get(seq_id, {})
            accepted_ids_map = sample_output.accepted_ids_map.get(seq_id, {})
            sampled_tokens_map = sample_output.sampled_tokens_map.get(seq_id, {})
            for block_id, accepted_ids in accepted_ids_map.items():
                if not accepted_ids:
                    continue
                diffusion_block = seq.diffusion_blocks[int(block_id)]
                sampled_tokens = sampled_tokens_map.get(block_id, [])
                true_local_ids = true_ids_map.get(block_id, [])
                for true_local_id, accepted_id in zip(true_local_ids, accepted_ids):
                    token = sampled_tokens[accepted_id]
                    diffusion_block.modify_token(true_local_id, token)
                    if (
                        (not seq.ignore_eos and token.item() == self.eos)
                        or seq.num_completion_tokens >= seq.max_tokens
                    ):
                        seq.meet_eos = True
            if seq.meet_eos and seq.diffusion_blocks[-1].available_to_cache:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.free(seq)
                if seq in self.running:
                    self.running.remove(seq)
                n_diff_steps[seq.seq_id] = seq.n_steps
            seq.post_process()
        return n_diff_steps
