#!/usr/bin/env python3
"""
用 torch.profiler 跑 Diffulex(D2F/Dream) 的性能剖析，并导出 flamegraph 所需 stacks。

设计目标：
- 直接复用 Diffulex 的配置入口（kv_cache_dtype / linear_*_dtype / decode_mode 等）
- 默认强制 TP=1/DP=1，避免 tp_worker 的 spawn 子进程导致 profiler 采不到 CUDA kernel
- 两阶段：先编译/初始化 warmup（不计入 profile），再进入 torch.profiler 采集窗口

输出：
- Chrome trace:  *.json  （可用 chrome://tracing 或 Perfetto 打开）
- Stacks:        *.stacks （用于生成火焰图，格式兼容 Brendan Gregg flamegraph 工具链）

示例：
  # BF16 基线
  python profile/torch_d2f_profiler.py --tag bf16 --kv-cache-dtype bf16

  # FP8 KV + W8A16（对比量化为何更慢）
  python profile/torch_d2f_profiler.py --tag w8a16_fp8kv --kv-cache-dtype fp8_e4m3 \
    --linear-attn-weight-dtype int8 --linear-mlp-weight-dtype int8

  # 指定 decode_mode（auto/varlen/static）
  python profile/torch_d2f_profiler.py --tag fp8kv_static --kv-cache-dtype fp8_e4m3 --decode-mode static
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

# Make stdout/stderr line-buffered so progress logs are visible even when redirected/captured.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# Ensure import from current repo.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from diffulex import Diffulex, SamplingParams
from diffulex_profiler import DiffulexProfiler, ProfilerConfig


def _default_prompts() -> List[str]:
    return [
        "What is 2+2?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate factorial.",
    ]


def _load_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompts_file:
        p = Path(args.prompts_file)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise ValueError("--prompts-file 必须是 JSON list[str]")
        return data
    if args.prompt:
        return args.prompt
    return _default_prompts()


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> None:
    parser = argparse.ArgumentParser("Diffulex torch.profiler flamegraph (D2F/Dream)")

    parser.add_argument("--model-path", type=str, required=True, help="模型路径（必填）")
    parser.add_argument("--lora-path", type=str, default="", help="LoRA 路径（可选）")
    parser.add_argument("--use-lora", action="store_true", help="启用 LoRA（需同时提供 --lora-path）")
    parser.add_argument("--cuda-home", type=str, default="", help="（可选）设置 CUDA_HOME/CUDA_PATH 并更新 PATH/LD_LIBRARY_PATH")

    parser.add_argument("--tag", type=str, default="torch_profile", help="输出文件名前缀")
    parser.add_argument("--out-dir", type=str, default="log/torch_profiles", help="输出目录（相对仓库根）")

    # Quantization / KV settings
    parser.add_argument("--kv-cache-dtype", type=str, default="bf16", help="bf16/fp8_e4m3/fp8_e5m2 (也支持别名 fp8/e4m3/e5m2)")
    parser.add_argument("--kv-cache-layout", type=str, default="unified", choices=["unified", "distinct"])
    parser.add_argument("--decode-mode", type=str, default="auto", choices=["auto", "varlen", "static"])

    parser.add_argument("--linear-attn-weight-dtype", type=str, default="bf16")
    parser.add_argument("--linear-mlp-weight-dtype", type=str, default="bf16")
    parser.add_argument("--linear-attn-act-dtype", type=str, default="bf16")
    parser.add_argument("--linear-mlp-act-dtype", type=str, default="bf16")

    # CUDA Graph
    parser.add_argument(
        "--use-cudagraph",
        action="store_true",
        help="启用 CUDA Graph（仅 decode_mode=static 且 shape 稳定时有意义）；默认关闭以避免 capture 成本影响分析。",
    )

    # Engine settings (force single-process profiling by default)
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="建议保持 1，否则会 spawn 子进程导致采集不到 CUDA")
    parser.add_argument("--data-parallel-size", type=int, default=1)
    # Distributed comm (avoid port conflicts with other local runs)
    parser.add_argument("--master-addr", type=str, default="localhost")
    parser.add_argument("--master-port", type=int, default=2333)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--max-model-len", type=int, default=1024)

    # Prompts / decode
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt", type=str, action="append", help="可多次传入，作为 prompts 列表；不传则用内置默认 prompts")
    parser.add_argument("--prompts-file", type=str, default="", help="JSON list[str] 文件路径")

    # Warmup + profiler schedule
    parser.add_argument("--compile-warmup-iters", type=int, default=1, help="用于 kernel 编译/缓存的 warmup 次数（不进入 profiler）")
    parser.add_argument("--profile-wait", type=int, default=0)
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-active", type=int, default=1)
    parser.add_argument("--profile-repeat", type=int, default=1)
    parser.add_argument(
        "--use-diffulex-profiler",
        action="store_true",
        help="改用 diffulex_profiler 的 PyTorchProfilerBackend（会导出 trace/stacks/top，并额外导出 summary/json）",
    )
    parser.add_argument(
        "--no-torch-profiler",
        action="store_true",
        help="仅运行一次稳态 generate（包含 compile warmup），不启用 torch.profiler。用于配合 ncu 等外部 profiler，避免 CUPTI 冲突。",
    )
    parser.add_argument(
        "--nvtx-range",
        type=str,
        default="",
        help="（可选）用 NVTX 把 profiled generate 包起来，便于 ncu 用 --nvtx-include 精准过滤。示例：--nvtx-range d2f_generate",
    )

    args = parser.parse_args()

    if args.cuda_home:
        cuda_home = Path(args.cuda_home)
        if not cuda_home.exists():
            raise FileNotFoundError(f"--cuda-home 不存在: {cuda_home}")
        os.environ["CUDA_HOME"] = str(cuda_home)
        os.environ["CUDA_PATH"] = str(cuda_home)
        os.environ["PATH"] = f"{cuda_home}/bin:{os.environ.get('PATH', '')}"
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_home}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
        os.environ["LIBRARY_PATH"] = f"{cuda_home}/lib64:{os.environ.get('LIBRARY_PATH', '')}"
        os.environ["CPATH"] = f"{cuda_home}/include:{os.environ.get('CPATH', '')}"
        os.environ["CUDACXX"] = str(cuda_home / "bin" / "nvcc")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    if args.tensor_parallel_size != 1 or args.data_parallel_size != 1:
        print(
            "[WARN] 你设置了 TP/DP != 1。Diffulex 会 spawn 子进程运行模型，"
            "torch.profiler 在父进程里通常采不到子进程里的 CUDA kernel。"
            "建议用 TP=1/DP=1 跑 profile。"
        )

    prompts = _load_prompts(args)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    out_root = _mkdir(_REPO_ROOT / args.out_dir)
    run_dir = _mkdir(out_root / time.strftime("%Y%m%d_%H%M%S"))
    print(f"[INFO] 输出目录: {run_dir}")

    # Build Diffulex
    use_lora = args.use_lora or bool(args.lora_path)
    llm = Diffulex(
        str(model_path),
        lora_path=args.lora_path,
        use_lora=use_lora,
        model_name="dream",
        decoding_strategy="d2f",
        enforce_eager=not args.use_cudagraph,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        master_addr=args.master_addr,
        master_port=args.master_port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max(1024, args.max_model_len),
        max_num_seqs=min(4, len(prompts)),
        kv_cache_dtype=args.kv_cache_dtype,
        kv_cache_layout=args.kv_cache_layout,
        decode_mode=None if args.decode_mode == "auto" else args.decode_mode,
        linear_attn_weight_dtype=args.linear_attn_weight_dtype,
        linear_mlp_weight_dtype=args.linear_mlp_weight_dtype,
        linear_attn_act_dtype=args.linear_attn_act_dtype,
        linear_mlp_act_dtype=args.linear_mlp_act_dtype,
    )

    try:
        # Compile / cache warmup (exclude from profile)
        for i in range(max(0, args.compile_warmup_iters)):
            print(f"[INFO] compile warmup {i+1}/{args.compile_warmup_iters} ...")
            with torch.profiler.record_function("diffulex.generate(warmup)"):
                _ = llm.generate(prompts, sampling_params, use_tqdm=False)
            torch.cuda.synchronize()

        # For external profilers (e.g., ncu). Avoid enabling torch.profiler (CUPTI) here.
        if args.no_torch_profiler:
            print("[INFO] --no-torch-profiler: 运行一次稳态 generate（不启用 torch.profiler）...")
            nvtx_handle = None
            nvtx_pushed = False
            if args.nvtx_range and torch.cuda.is_available():
                # Nsight Compute CLI --nvtx-include matches start/end ranges (not push/pop ranges).
                # Prefer range_start/range_end if available; fallback to push/pop for other tools.
                try:
                    nvtx_handle = torch.cuda.nvtx.range_start(args.nvtx_range)
                except Exception:
                    try:
                        torch.cuda.nvtx.range_push(args.nvtx_range)
                        nvtx_pushed = True
                    except Exception:
                        pass
            try:
                with torch.profiler.record_function("diffulex.generate(profiled)"):
                    _ = llm.generate(prompts, sampling_params, use_tqdm=False)
                torch.cuda.synchronize()
            finally:
                if args.nvtx_range and torch.cuda.is_available():
                    if nvtx_handle is not None:
                        try:
                            torch.cuda.nvtx.range_end(nvtx_handle)
                        except Exception:
                            pass
                    elif nvtx_pushed:
                        try:
                            torch.cuda.nvtx.range_pop()
                        except Exception:
                            pass
            print(f"[INFO] 完成（无 torch.profiler 输出）。输出目录: {run_dir}")
            return

        # Option A: use Diffulex built-in profiler framework.
        if args.use_diffulex_profiler:
            profiler = DiffulexProfiler(
                config=ProfilerConfig(
                    enabled=True,
                    backend="pytorch",
                    output_dir=str(run_dir),
                    export_formats=["json", "summary"],
                    pytorch_profiler_config={
                        # Ensure artifacts are written into the same run_dir.
                        "output_dir": str(run_dir),
                        "record_shapes": True,
                        "profile_memory": True,
                        "with_stack": True,
                        # Also export stacks/top table for flamegraph + quick inspection.
                        "export_stacks": True,
                        "stacks_metric": "self_cuda_time_total",
                        "export_table": True,
                        "table_row_limit": 80,
                    },
                )
            )

            # In this mode, we don't use torch.profiler schedule; we just profile the steady-state generate.
            print("[INFO] 使用 diffulex_profiler(pytorch backend) 采集一次稳态 generate ...")
            with profiler.profile(
                "diffulex.generate(profiled)",
                metadata={
                    "tag": args.tag,
                    "decode_mode": args.decode_mode,
                    "kv_cache_dtype": args.kv_cache_dtype,
                    "linear_attn_weight_dtype": args.linear_attn_weight_dtype,
                    "linear_mlp_weight_dtype": args.linear_mlp_weight_dtype,
                    "linear_attn_act_dtype": args.linear_attn_act_dtype,
                    "linear_mlp_act_dtype": args.linear_mlp_act_dtype,
                },
            ):
                _ = llm.generate(prompts, sampling_params, use_tqdm=False)
                torch.cuda.synchronize()
            print("[INFO] diffulex_profiler 采集完成（trace/stacks/top 已导出到输出目录）。")
            profiler.export(str(run_dir / f"{args.tag}"))
            print(f"[INFO] 输出目录: {run_dir}")
            return

        # Option B: raw torch.profiler with schedule (more controllable / multi-step).
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        def _trace_handler(prof: torch.profiler.profile) -> None:
            # One trace per active window.
            step = getattr(prof, "step_num", None)
            suffix = f"_step{step}" if step is not None else ""
            trace_path = run_dir / f"{args.tag}{suffix}.trace.json"
            stacks_path = run_dir / f"{args.tag}{suffix}.stacks"
            summary_path = run_dir / f"{args.tag}{suffix}.top.txt"
            prof.export_chrome_trace(str(trace_path))
            # 用 self_cuda_time_total 更聚焦 kernel 开销；若只关心 CPU 改成 self_cpu_time_total
            try:
                prof.export_stacks(str(stacks_path), "self_cuda_time_total")
            except Exception:
                # CUDA 不可用/未编译 kineto 时可能失败，仍保留 trace
                pass
            try:
                top = prof.key_averages().table(
                    sort_by="self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total",
                    row_limit=50,
                )
                summary_path.write_text(top, encoding="utf-8")
            except Exception:
                pass

        schedule = torch.profiler.schedule(
            wait=max(0, args.profile_wait),
            warmup=max(0, args.profile_warmup),
            active=max(1, args.profile_active),
            repeat=max(1, args.profile_repeat),
        )
        total_steps = args.profile_wait + args.profile_warmup + args.profile_active * args.profile_repeat
        print(
            f"[INFO] profiler schedule: wait={args.profile_wait}, warmup={args.profile_warmup}, "
            f"active={args.profile_active}, repeat={args.profile_repeat} -> total_steps={total_steps}"
        )

        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=_trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for step in range(total_steps):
                print(f"[INFO] profiled generate step {step+1}/{total_steps} ...")
                with torch.profiler.record_function("diffulex.generate(profiled)"):
                    _ = llm.generate(prompts, sampling_params, use_tqdm=False)
                torch.cuda.synchronize()
                prof.step()

        print("[INFO] 采集完成。你可以用 trace.json 打开时间线，用 .stacks 生成火焰图。")
        print(f"[INFO] 输出目录: {run_dir}")
    finally:
        try:
            llm.exit()
        except Exception:
            pass


if __name__ == "__main__":
    main()

