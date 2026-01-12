"""
Diffulex Benchmark - Benchmark framework for evaluating Diffulex inference engine performance
"""

from diffulex_bench.runner import BenchmarkRunner
from diffulex_bench.datasets import load_benchmark_dataset
from diffulex_bench.metrics import compute_metrics
from diffulex.logger import setup_logger, get_logger
from diffulex_bench.config import BenchmarkConfig, EngineConfig, EvalConfig

# Import lm_eval model to register it
try:
    from diffulex_bench.lm_eval_model import DiffulexLM
    __all__ = [
        "BenchmarkRunner",
        "load_benchmark_dataset",
        "compute_metrics",
        "setup_logger",
        "get_logger",
        "BenchmarkConfig",
        "EngineConfig",
        "EvalConfig",
        "DiffulexLM",
    ]
except ImportError:
    __all__ = [
        "BenchmarkRunner",
        "load_benchmark_dataset",
        "compute_metrics",
        "setup_logger",
        "get_logger",
        "BenchmarkConfig",
        "EngineConfig",
        "EvalConfig",
    ]

