"""
Argument Parser - Command line argument parsing for benchmark
"""

import argparse
from pathlib import Path


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser for benchmark
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Diffulex Benchmark using lm-evaluation-harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using configuration file (recommended)
  python -m diffulex_bench.main --config diffulex_bench/configs/example.yml

  # Using command line arguments
  python -m diffulex_bench.main \\
    --model-path /path/to/model \\
    --dataset gsm8k \\
    --dataset-limit 100 \\
    --output-dir ./results

  # With custom model settings
  python -m diffulex_bench.main \\
    --model-path /path/to/model \\
    --model-name dream \\
    --decoding-strategy d2f \\
    --dataset gsm8k \\
    --temperature 0.0 \\
    --max-tokens 256
        """
    )
    
    # Logging arguments
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (optional)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path (YAML or JSON). Default: configs/example.yml",
    )
    
    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        help="Model path",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Tokenizer path (defaults to model-path)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dream",
        choices=["dream", "sdar", "fast_dllm_v2"],
        help="Model name",
    )
    parser.add_argument(
        "--decoding-strategy",
        type=str,
        default="d2f",
        choices=["d2f", "block_diffusion", "fast_dllm"],
        help="Decoding strategy",
    )
    parser.add_argument(
        "--mask-token-id",
        type=int,
        default=151666,
        help="Mask token ID",
    )
    
    # Inference arguments
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="Data parallel size",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum model length",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=4096,
        help="Maximum number of batched tokens",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=128,
        help="Maximum number of sequences",
    )
    
    # Sampling arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS token",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        help="Dataset/task name (e.g., gsm8k, humaneval)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Limit number of samples",
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        default=True,
        help="Save results to file",
    )
    parser.add_argument(
        "--no-save-results",
        dest="save_results",
        action="store_false",
        help="Do not save results to file",
    )
    
    # LoRA arguments
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="",
        help="LoRA path",
    )
    
    # Engine arguments
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Enforce eager mode (disable CUDA graphs)",
    )
    parser.add_argument(
        "--kv-cache-layout",
        type=str,
        default="unified",
        choices=["unified", "distinct"],
        help="KV cache layout",
    )
    
    # D2F-specific arguments
    parser.add_argument(
        "--accept-threshold",
        type=float,
        default=0.9,
        help="Accept threshold for D2F",
    )
    parser.add_argument(
        "--complete-threshold",
        type=float,
        default=0.95,
        help="Complete threshold for D2F",
    )
    parser.add_argument(
        "--add-new-block-threshold",
        type=float,
        default=0.1,
        help="Add new block threshold for D2F",
    )
    parser.add_argument(
        "--diffusion-block-size",
        type=int,
        default=32,
        help="Diffusion block size",
    )
    
    return parser


def get_default_config_path() -> Path:
    """
    Get default configuration file path
    
    Returns:
        Path to default config file
    """
    config_dir = Path(__file__).parent / "configs"
    default_config = config_dir / "example.yml"
    return default_config

