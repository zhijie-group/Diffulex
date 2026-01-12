"""
Benchmark Main Entry - Main entry point for benchmark using lm-evaluation-harness
"""

import sys
import logging
from pathlib import Path

from diffulex_bench.config import BenchmarkConfig, EngineConfig, EvalConfig
from diffulex.logger import setup_logger, get_logger
from diffulex_bench.arg_parser import create_argument_parser, get_default_config_path

try:
    from lm_eval.__main__ import cli_evaluate
except ImportError:
    cli_evaluate = None


def config_to_model_args(config: BenchmarkConfig) -> str:
    """
    Convert BenchmarkConfig to lm_eval model_args string format
    
    Args:
        config: Benchmark configuration
        
    Returns:
        Model arguments string in key=value format
    """
    engine = config.engine
    eval_config = config.eval
    
    args_dict = {
        'pretrained': engine.model_path,
        'model_name': engine.model_name,
        'decoding_strategy': engine.decoding_strategy,
        'mask_token_id': engine.mask_token_id,
        'tensor_parallel_size': engine.tensor_parallel_size,
        'data_parallel_size': engine.data_parallel_size,
        'gpu_memory_utilization': engine.gpu_memory_utilization,
        'max_model_len': engine.max_model_len,
        'max_num_batched_tokens': engine.max_num_batched_tokens,
        'max_num_seqs': engine.max_num_seqs,
        'temperature': eval_config.temperature,
        'max_new_tokens': eval_config.max_tokens,
        'use_lora': engine.use_lora,
        'enforce_eager': engine.enforce_eager,
        'kv_cache_layout': engine.kv_cache_layout,
        'accept_threshold': engine.accept_threshold,
        'complete_threshold': engine.complete_threshold,
        'add_new_block_threshold': engine.add_new_block_threshold,
        'diffusion_block_size': engine.diffusion_block_size,
        'wait_ready': True,
    }
    
    if engine.tokenizer_path:
        args_dict['tokenizer_path'] = engine.tokenizer_path
    
    if engine.use_lora and engine.lora_path:
        args_dict['lora_path'] = engine.lora_path
    
    # Convert to string format: key1=value1,key2=value2
    args_list = [f"{k}={v}" for k, v in args_dict.items()]
    return ','.join(args_list)


def dataset_name_to_tasks(dataset_name: str) -> str:
    """
    Convert dataset name to lm_eval task name
    
    Args:
        dataset_name: Dataset name (e.g., "gsm8k", "humaneval")
        
    Returns:
        lm_eval task name
    """
    mapping = {
        'gsm8k': 'gsm8k',
        'humaneval': 'humaneval',
    }
    return mapping.get(dataset_name, dataset_name)


def run_benchmark(config: BenchmarkConfig) -> None:
    """
    Run benchmark using lm-evaluation-harness
    
    Args:
        config: Benchmark configuration
    """
    logger = get_logger(__name__)
    
    if cli_evaluate is None:
        logger.error(
            "lm-evaluation-harness is not installed. "
            "Please install it with: pip install lm-eval"
        )
        sys.exit(1)
    
    benchmark_info = [
        '=' * 80, 
        'Diffulex Benchmark (using lm-evaluation-harness)',
        '=' * 80,
        f'Model: {config.engine.model_path}',
        f'Model Name: {config.engine.model_name}',
        f'Decoding Strategy: {config.engine.decoding_strategy}',
        f'Tasks: {config.eval.dataset_name}',
        f'Output Directory: {config.eval.output_dir}',
        '=' * 80,
    ]
    logger.info('\n'.join(benchmark_info))
    
    # Convert config to lm_eval arguments
    model_args = config_to_model_args(config)
    tasks = dataset_name_to_tasks(config.eval.dataset_name)
    
    # Prepare sys.argv for lm_eval
    original_argv = sys.argv.copy()
    
    try:
        sys.argv = [
            "lm_eval",
            "--model", "diffulex",
            "--model_args", model_args,
            "--tasks", tasks,
            "--batch_size", "1",
            "--output_path", config.eval.output_dir,
        ]
        
        if config.eval.dataset_limit:
            sys.argv.extend(["--limit", str(config.eval.dataset_limit)])
        
        # Add any additional lm_eval arguments from config if needed
        # For now, we use default batch_size=1
        
        lm_eval_info = [
            '=' * 80,
            'Starting lm-evaluation-harness evaluation...',
            '=' * 80,
            f'Model args: {model_args}',
            f'Tasks: {tasks}',
            '=' * 80,
        ]
        logger.info('\n'.join(lm_eval_info))
        
        # Run lm_eval
        cli_evaluate()
        
        logger.success("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Restore original argv
        sys.argv = original_argv


def load_config_from_args(args) -> BenchmarkConfig:
    """
    Load configuration from command line arguments
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        BenchmarkConfig instance
    """
    logger = get_logger(__name__)
    
    # Try to load from config file
    if args.config:
        config_path = Path(args.config)
    else:
        # Try default config path
        default_config = get_default_config_path()
        if default_config.exists():
            config_path = default_config
            logger.info(f"Using default config: {config_path}")
        else:
            config_path = None
    
    if config_path and config_path.exists():
        if config_path.suffix in ['.yaml', '.yml']:
            config = BenchmarkConfig.from_yaml(str(config_path))
        elif config_path.suffix == '.json':
            config = BenchmarkConfig.from_json(str(config_path))
        else:
            logger.error(f"Unsupported config file format: {config_path.suffix}")
            sys.exit(1)
        logger.info(f"Loaded configuration from: {config_path}")
        
        # Override with command line arguments if provided
        if args.model_path:
            config.engine.model_path = args.model_path
        if args.dataset:
            config.eval.dataset_name = args.dataset
        if args.dataset_limit is not None:
            config.eval.dataset_limit = args.dataset_limit
        if args.output_dir:
            config.eval.output_dir = args.output_dir
    else:
        if not args.model_path:
            logger.error("Either --config or --model-path must be provided")
            sys.exit(1)
        
        # Create config from command line arguments
        engine = EngineConfig(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            model_name=args.model_name,
            decoding_strategy=args.decoding_strategy,
            mask_token_id=args.mask_token_id,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=args.data_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=getattr(args, 'max_num_batched_tokens', 4096),
            max_num_seqs=getattr(args, 'max_num_seqs', 128),
            use_lora=args.use_lora,
            lora_path=args.lora_path,
            enforce_eager=getattr(args, 'enforce_eager', False),
            kv_cache_layout=getattr(args, 'kv_cache_layout', 'unified'),
            accept_threshold=args.accept_threshold,
            complete_threshold=args.complete_threshold,
            add_new_block_threshold=args.add_new_block_threshold,
            diffusion_block_size=args.diffusion_block_size,
        )
        
        eval_config = EvalConfig(
            dataset_name=args.dataset,
            dataset_split=getattr(args, 'dataset_split', 'test'),
            dataset_limit=args.dataset_limit,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            ignore_eos=getattr(args, 'ignore_eos', False),
            output_dir=args.output_dir,
            save_results=args.save_results,
        )
        
        config = BenchmarkConfig(engine=engine, eval=eval_config)
    
    return config


def main():
    """Main function"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logger
    log_level = getattr(logging, args.log_level.upper())
    setup_logger("diffulex_bench", level=log_level, log_file=args.log_file)
    
    # Load configuration
    config = load_config_from_args(args)
    
    # Run benchmark using lm_eval
    run_benchmark(config)


if __name__ == "__main__":
    main()
