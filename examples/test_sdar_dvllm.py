import argparse
import os
import shutil
from pathlib import Path
import sys


# Ensure we import Diffulex from THIS repo (fast_dllm_v2 workspace), not an installed copy.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _convert_safetensors_keys(src_safetensors: Path, dst_safetensors: Path) -> None:
    """Convert HF-style SDAR weight keys to Diffulex-native names.

    HF checkpoint keys are like:
      - model.embed_tokens.weight
      - model.layers.0.self_attn.q_proj.weight
      - lm_head.weight

    Diffulex native SDAR (this repo) expects:
      - model.embed_tokens.weight
      - model.layers.0.self_attn.q_proj.weight
      - lm_head.weight

    So conversion is only needed when the source checkpoint is missing the leading
    "model." prefix (some export pipelines do that).
    """
    from safetensors.torch import safe_open, save_file

    tensors = {}
    with safe_open(str(src_safetensors), framework="pt", device="cpu") as f:
        for k in f.keys():
            new_k = k
            # Add missing prefix for backbone weights.
            if not new_k.startswith("model.") and new_k != "lm_head.weight":
                new_k = "model." + new_k
            tensors[new_k] = f.get_tensor(k)

    dst_safetensors.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dst_safetensors))


def ensure_converted_model_dir(src_model_dir: Path, out_dir: Path) -> Path:
    """Create a converted model dir that Diffulex-native SDAR can load, if needed."""
    marker = out_dir / ".diffulex_sdar_converted"
    dst_safetensors = out_dir / "model.safetensors"
    src_safetensors = src_model_dir / "model.safetensors"

    if not src_safetensors.exists():
        raise FileNotFoundError(f"Missing {src_safetensors}")

    # If the checkpoint already matches Diffulex module names, use it directly.
    from safetensors.torch import safe_open

    with safe_open(str(src_safetensors), framework="pt", device="cpu") as f:
        keys = set(f.keys())
    if "model.embed_tokens.weight" in keys:
        return src_model_dir

    if marker.exists() and dst_safetensors.exists():
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy non-weight artifacts required by AutoConfig/AutoTokenizer.
    for name in [
        "config.json",
        "configuration_sdar.py",
        "modeling_sdar.py",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "generation_config.json",
        "chat_template.jinja",
        "README.md",
        "tokenization_qwen2.py",
        "tokenization_qwen2_fast.py",
    ]:
        src = src_model_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    # Convert weights.
    _convert_safetensors_keys(src_safetensors, dst_safetensors)

    marker.write_text(f"converted_from={src_model_dir}\n", encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="/home/lzx/SDAR/training/model/SDAR-1.7B-Chat",
        help="SDAR HF model directory (contains config.json + model.safetensors).",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--converted-dir",
        type=str,
        default="/home/lzx/tmp/diffulex_sdar_converted",
        help="Output directory for converted checkpoint keys (Diffulex-native).",
    )
    parser.add_argument("--prompt", type=str, default="你好，请用一句话介绍 SDAR。")
    parser.add_argument("--max-len", type=int, default=128)
    args = parser.parse_args()

    src_model_dir = Path(args.model)
    converted_dir = Path(args.converted_dir)
    model_dir = ensure_converted_model_dir(src_model_dir, converted_dir)

    # IMPORTANT: do not import diffulex before conversion; it may eagerly load config.
    import socket
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer

    # Minimal single-process distributed init (required by Diffulex TP layers).
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))

    torch.cuda.set_device(args.device)
    dist.init_process_group("nccl", rank=0, world_size=1)

    # Build Config + load model weights using Diffulex loader.
    from diffulex.config import Config
    from diffulex.model.auto_model import AutoModelForDiffusionLM

    cfg = Config(
        model=str(model_dir),
        model_name="sdar",
        tensor_parallel_size=1,
        data_parallel_size=1,
        enforce_eager=True,
    )

    dtype = getattr(cfg.hf_config, "torch_dtype", None) or torch.bfloat16
    torch.set_default_dtype(dtype)
    torch.set_default_device(f"cuda:{args.device}")

    model = AutoModelForDiffusionLM.from_config(cfg).eval()

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, use_fast=True)
    ids = tokenizer.encode(args.prompt, add_special_tokens=True)[: args.max_len]
    input_ids = torch.tensor(ids, dtype=torch.int64, device=f"cuda:{args.device}")
    positions = torch.arange(input_ids.numel(), dtype=torch.int64, device=f"cuda:{args.device}")

    # Provide minimal attention metadata so Diffulex Attention can run in "prefill" mode.
    # In the real engine this is provided by strategy-specific runners.
    from diffulex.attention.metadata import set_fetch_fn_for_attn_metadata
    from types import SimpleNamespace

    n = int(input_ids.numel())
    cu = torch.tensor([0, n], dtype=torch.int32, device=f"cuda:{args.device}")

    def _fetch_attn_metadata():
        return SimpleNamespace(
            # Core fields used by attn_impl.Attention
            is_prefill=True,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=n,
            max_seqlen_k=n,
            block_tables=None,
            slot_mapping=None,
            # KV cache controls
            kv_cache_layout="unified",
            need_kv_cache_store=False,
            # Fields referenced in decode path (kept for completeness)
            seqs=[],
            total_lens=[],
            seq_lens=[],
            seq_lens_ts=None,
            block_mask=None,
        )

    set_fetch_fn_for_attn_metadata(_fetch_attn_metadata)

    with torch.inference_mode():
        hs = model(input_ids, positions)
        logits = model.compute_logits(hs)
        next_id = int(logits[-1].argmax().item())

    print("=" * 80)
    print(f"[model_dir] {model_dir}")
    print(f"[prompt] {args.prompt}")
    print(f"[input_len] {len(ids)}")
    print(f"[next_token_id] {next_id}")
    print(f"[next_token] {tokenizer.decode([next_id])!r}")

    dist.destroy_process_group()


if __name__ == "__main__":
    # Avoid tokenizer parallel warnings in multi-proc.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()


