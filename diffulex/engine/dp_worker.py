import os
import sys
import torch
import atexit
import traceback
import faulthandler

import multiprocessing as mp

from typing import Any
from multiprocessing.connection import wait as mp_wait

from diffulex.config import Config
from diffulex.engine.tp_worker import DiffulexTPWorker
from diffulex.sampling_params import SamplingParams
from diffulex.logger import get_logger

logger = get_logger(__name__)


def _dp_child_entry(config: Config, dp_idx: int, local_devices: list[int], conn):
    """Child process entry point: create an LLMEngine for this DP rank and serve RPC via Pipe."""
    try:
        # Enable Python-level crash diagnostics for hard crashes (segfault, OOM kill signals, etc.).
        try:
            faulthandler.enable(all_threads=True)
        except Exception:
            pass
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in local_devices)
        cfg = Config(
            model=config.model,
            lora_path=config.lora_path,
            model_name=config.model_name,
            decoding_strategy=config.decoding_strategy,
            mask_token_id=config.mask_token_id,
            diffusion_block_size=config.diffusion_block_size,
            accept_threshold=config.accept_threshold,
            complete_threshold=config.complete_threshold,
            add_new_block_threshold=config.add_new_block_threshold,
            use_lora=config.use_lora,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_num_seqs=config.max_num_seqs,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            data_parallel_size=1,
            tensor_parallel_size=config.tensor_parallel_size,
            master_addr=config.master_addr,
            master_port=int(config.master_port) + dp_idx,
            shm_name=f"{config.shm_name}_{dp_idx}",
            enforce_eager=config.enforce_eager,
            kvcache_block_size=config.kvcache_block_size,
            num_kvcache_blocks=config.num_kvcache_blocks,
            k_cache_hdim_split_factor_x=config.k_cache_hdim_split_factor_x,
            kv_cache_layout=config.kv_cache_layout,
        )
        setattr(cfg, "device_start", 0)
        setattr(cfg, "device_ids", local_devices)

        engine = DiffulexTPWorker(cfg.model, **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__.keys() if k != "model"})

        while True:
            msg = conn.recv()
            if not msg:
                continue
            cmd, *args = msg
            if cmd == "generate":
                plist, sp_arg, use_tqdm = args
                out = engine.generate(plist, sp_arg, use_tqdm)
                conn.send(("ok", out))
                continue
            if cmd == "exit":
                engine.exit()
                conn.send(("ok", None))
                break
            elif cmd == "add_request":
                prompt, sp = args
                sid = engine.add_request(prompt, sp)
                conn.send(("ok", sid))
            elif cmd == "step":
                out = engine.step()
                conn.send(("ok", out))
            elif cmd == "is_finished":
                conn.send(("ok", engine.is_finished()))
            else:
                conn.send(("err", f"unknown_cmd:{cmd}"))
    except Exception as e:
        # Include full traceback for easier debugging and also log as a fallback.
        tb = traceback.format_exc()
        msg = f"{type(e).__name__}: {e}\n{tb}"
        try:
            conn.send(("err", msg))
        except Exception:
            pass
        try:
            # Use logger for error reporting
            child_logger = get_logger(f"diffulex.engine.dp_worker.child_{dp_idx}")
            child_logger.error(f"[DP Child {dp_idx}] Unhandled exception:\n{msg}")
        except Exception:
            # Final fallback to stderr
            try:
                print(f"[DP Child {dp_idx}] Unhandled exception:\n{msg}", file=sys.stderr, flush=True)
            except Exception:
                pass


class DiffulexDPWorker:
    """Data-parallel wrapper that runs N independent TP groups as child processes and aggregates results."""
    def __init__(self, model, **kwargs):
        config_fields = {f for f in Config.__dataclass_fields__.keys()}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = cfg = Config(model, **config_kwargs)
        self.dp_size = cfg.data_parallel_size
        assert self.dp_size > 1, "Use LLMEngine directly when data_parallel_size == 1"

        ctx = mp.get_context("spawn")
        self.conns: list[Any] = []
        self.ps: list[mp.Process] = []
        # Topology check and mapping
        base_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if base_visible:
            vis = [int(x) for x in base_visible.split(',') if x.strip() != '']
        else:
            vis = list(range(torch.cuda.device_count()))

        need_gpus = self.dp_size * cfg.tensor_parallel_size
        assert len(vis) >= need_gpus, f"Require {need_gpus} GPUs (dp={self.dp_size}, tp={cfg.tensor_parallel_size}), visible {len(vis)}"

        # Optional overrides: kwargs['device_ids'] or env D2F_DEVICE_MAP
        override = None
        if 'device_ids' in kwargs and kwargs['device_ids']:
            override = list(kwargs['device_ids'])
        elif os.environ.get('D2F_DEVICE_MAP'):
            override = [int(x) for x in os.environ['D2F_DEVICE_MAP'].split(',') if x.strip() != '']
        if override is not None:
            assert len(override) >= need_gpus, f"device_ids length {len(override)} < required {need_gpus}"
            # All override devices must be in visible list
            assert all(d in vis for d in override[:need_gpus]), "device_ids must be subset of CUDA_VISIBLE_DEVICES"
            plan = override[:need_gpus]
        else:
            plan = vis[:need_gpus]

        tp = cfg.tensor_parallel_size
        for dp_idx in range(self.dp_size):
            local_devices = plan[dp_idx*tp:(dp_idx+1)*tp]
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(target=_dp_child_entry, args=(cfg, dp_idx, local_devices, child_conn))
            p.start()
            self.ps.append(p)
            self.conns.append(parent_conn)
        self._rr = 0  # round-robin pointer
        self._gid_counter = 0
        self._gid_map = {}  # (replica, local_id) -> global_id
        self._rev_gid_map = {}  # global_id -> (replica, local_id)
        atexit.register(self.exit)

    def _ask(self, replica: int, cmd: str, *args):
        conn = self.conns[replica]
        conn.send((cmd, *args))
        try:
            tag, payload = conn.recv()
        except EOFError:
            p = self.ps[replica]
            exitcode = p.exitcode
            raise RuntimeError(
                f"DP child #{replica} terminated unexpectedly (exitcode={exitcode}). "
                f"This may indicate OOM or a native crash. Try setting env: "
                f"PYTHONFAULTHANDLER=1 CUDA_LAUNCH_BLOCKING=1 TORCH_SHOW_CPP_STACKTRACES=1 to get more diagnostics."
            )
        if tag == "ok":
            return payload
        raise RuntimeError(f"DP child #{replica} error: {payload}")

    def exit(self):
        for i, p in enumerate(self.ps):
            if p.is_alive():
                try:
                    self._ask(i, "exit")
                except Exception:
                    pass
                p.join(timeout=5)

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        target = self._rr
        self._rr = (self._rr + 1) % self.dp_size
        local_id = self._ask(target, "add_request", prompt, sampling_params)
        gid = self._gid_counter
        self._gid_counter += 1
        self._gid_map[(target, local_id)] = gid
        self._rev_gid_map[gid] = (target, local_id)
        return gid

    def step(self):
        all_outputs = []
        total_tokens = 0
        any_prefill = False
        merged_diff_steps = {}
        merged_deltas = []
        for i in range(self.dp_size):
            done = self._ask(i, "is_finished")
            if done:
                continue
            outputs, num_tokens, is_prefill, n_diff_steps, deltas = self._ask(i, "step")
            if outputs:
                # remap local seq_ids to global ids
                for sid, toks in outputs:
                    gid = self._gid_map.get((i, sid), None)
                    if gid is not None:
                        all_outputs.append((gid, toks))
            total_tokens += num_tokens
            any_prefill = any_prefill or is_prefill
            if n_diff_steps:
                merged_diff_steps.update(n_diff_steps)
            if deltas:
                for sid, toks, fin in deltas:
                    gid = self._gid_map.get((i, sid), None)
                    if gid is not None:
                        merged_deltas.append((gid, toks, fin))
        return all_outputs, total_tokens, any_prefill, merged_diff_steps, merged_deltas

    def is_finished(self):
        return all(self._ask(i, "is_finished") for i in range(self.dp_size))

    def generate(self, prompts: list[str] | list[list[int]], sampling_params: SamplingParams | list[SamplingParams], use_tqdm: bool = True):
        """Load-balanced generate with random shuffling and stable order restoration.
        - Randomly shuffle inputs to balance load across DP replicas.
        - Partition shuffled list evenly among replicas.
        - Send to children, collect outputs, then unshuffle to original order.
        """
        import random
        n = len(prompts)
        idxs = list(range(n))
        random.shuffle(idxs)
        shuffled_prompts = [prompts[i] for i in idxs]
        # Align sampling params with shuffled prompts
        if isinstance(sampling_params, list):
            if len(sampling_params) == n:
                shuffled_sps = [sampling_params[i] for i in idxs]
            elif len(sampling_params) == self.dp_size:
                # per-shard SP; keep as-is and broadcast per-shard below
                shuffled_sps = sampling_params
            else:
                shuffled_sps = [sampling_params[0]] * n
        else:
            shuffled_sps = sampling_params

        # Even partition of shuffled inputs
        base = n // self.dp_size
        rem = n % self.dp_size
        slices = {}
        start = 0
        for i in range(self.dp_size):
            add = base + (1 if i < rem else 0)
            end = start + add
            if start < end:
                slices[i] = (start, end)
            start = end

        pending = {}
        conn_to_idx = {}
        collected = {}
        for i, (s, e) in slices.items():
            if isinstance(shuffled_sps, list):
                if len(shuffled_sps) == n:
                    sp_arg = shuffled_sps[s:e]
                elif len(shuffled_sps) == self.dp_size:
                    sp_arg = shuffled_sps[i]
                else:
                    sp_arg = shuffled_sps[0]
            else:
                sp_arg = shuffled_sps
            conn = self.conns[i]
            conn.send(("generate", shuffled_prompts[s:e], sp_arg, use_tqdm))
            pending[i] = True
            conn_to_idx[conn] = i
        # Collect
        while pending:
            ready = mp_wait([self.conns[i] for i in pending.keys()])
            for conn in ready:
                try:
                    tag, payload = conn.recv()
                except EOFError:
                    idx = conn_to_idx[conn]
                    p = self.ps[idx]
                    exitcode = p.exitcode
                    raise RuntimeError(
                        f"DP child #{idx} terminated unexpectedly during generate (exitcode={exitcode}). "
                        f"Enable envs: PYTHONFAULTHANDLER=1 CUDA_LAUNCH_BLOCKING=1 TORCH_SHOW_CPP_STACKTRACES=1 for more info."
                    )
                idx = conn_to_idx[conn]
                if tag == "ok":
                    collected[idx] = payload
                else:
                    raise RuntimeError(f"DP child #{idx} error: {payload}")
                del pending[idx]
        # Restore to original order
        restored = [None] * n
        for i, (s, e) in slices.items():
            outs = collected.get(i, [])
            # outs are aligned with shuffled order s:e
            for local_k, out in enumerate(outs):
                global_pos = s + local_k
                orig_idx = idxs[global_pos]
                restored[orig_idx] = out
        assert all(x is not None for x in restored), "Mismatch in outputs after DP collection"
        return restored
