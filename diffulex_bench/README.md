# Diffulex Benchmark

Benchmark framework for evaluating Diffulex inference engine using lm-evaluation-harness.

## Features

- ✅ **lm-evaluation-harness Integration**: Full support for 50+ evaluation tasks
- ✅ **YAML Configuration**: Clean and readable configuration files
- ✅ **Professional Logging**: Colored output with rich formatting
- ✅ **Flexible Configuration**: Support both config files and command-line arguments
- ✅ **Multiple Models**: Support for Dream, SDAR, Fast-dLLM-v2 models
- ✅ **Multiple Strategies**: D2F, Block Diffusion, Fast-dLLM decoding strategies

## Quick Start

### Installation

```bash
# Install dependencies
pip install lm-eval rich colorama

# Install diffulex (if not already installed)
pip install -e .
```

### Using Configuration File (Recommended)

1. **Create or use existing config file**:

```bash
# Copy example config
cp diffulex_bench/configs/example.yml my_config.yml

# Edit the config file
vim my_config.yml
```

2. **Run benchmark**:

```bash
python -m diffulex_bench.main --config my_config.yml
```

### Using Command Line Arguments

```bash
python -m diffulex_bench.main \
    --model-path /path/to/model \
    --model-name dream \
    --decoding-strategy d2f \
    --dataset gsm8k \
    --dataset-limit 100 \
    --temperature 0.0 \
    --max-tokens 256 \
    --output-dir ./results
```

## Configuration Files

Configuration files are located in `diffulex_bench/configs/` directory. We use YAML format for better readability.

### Configuration Structure

Configurations are organized into two sections:

1. **`engine`**: Engine configuration (model weights, LoRA, model name, strategy, inference parameters)
2. **`eval`**: Evaluation configuration (dataset, tasks, sampling parameters, output settings)

### Example Configuration

See `diffulex_bench/configs/example.yml` for a complete example:

```yaml
# Engine configuration - Parameters for Diffulex engine
engine:
  # Model and weights
  model_path: "/path/to/your/model"
  model_name: "dream"
  decoding_strategy: "d2f"
  mask_token_id: 151666
  
  # LoRA configuration
  use_lora: false
  lora_path: ""
  
  # Parallelism and memory
  tensor_parallel_size: 1
  data_parallel_size: 1
  gpu_memory_utilization: 0.9
  max_model_len: 2048
  
  # D2F-specific parameters
  accept_threshold: 0.9
  complete_threshold: 0.95
  add_new_block_threshold: 0.1

# Evaluation configuration - Parameters for benchmark
eval:
  # Task/Dataset
  dataset_name: "gsm8k"
  dataset_limit: 100
  
  # Sampling
  temperature: 0.0
  max_tokens: 256
  
  # Output
  output_dir: "benchmark_results"
```

### Pre-configured Examples

- `configs/example.yml`: Complete example with all options
- `configs/dream_d2f_gsm8k.yml`: Dream model with D2F strategy on GSM8K

## Supported Tasks

The framework supports all tasks available in lm-evaluation-harness, including:

- **GSM8K**: Math word problems
- **HumanEval**: Code generation
- **HellaSwag**: Commonsense reasoning
- **MMLU**: Massive multitask language understanding
- And 50+ more tasks...

See [lm-evaluation-harness tasks](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_table.md) for the complete list.

## Model Configuration

### Model Types

- `dream`: Dream model
- `sdar`: SDAR model
- `fast_dllm_v2`: Fast-dLLM-v2 model

### Decoding Strategies

- `d2f`: Discrete Diffusion Forcing
- `block_diffusion`: Block Diffusion
- `fast_dllm`: Fast-dLLM

### Example: Dream with D2F

```yaml
engine:
  model_path: "/path/to/dream/model"
  model_name: "dream"
  decoding_strategy: "d2f"
  mask_token_id: 151666
  accept_threshold: 0.9
  complete_threshold: 0.95
  add_new_block_threshold: 0.1

eval:
  dataset_name: "gsm8k"
  temperature: 0.0
  max_tokens: 256
```

## Command Line Arguments

### Basic Arguments

```bash
--config PATH              # Configuration file path (YAML or JSON)
--model-path PATH          # Model path (required if no config)
--dataset TASK             # Task name (e.g., gsm8k, humaneval)
--output-dir PATH          # Output directory
```

### Model Arguments

```bash
--model-name NAME          # Model name: dream, sdar, fast_dllm_v2
--decoding-strategy STR    # Strategy: d2f, block_diffusion, fast_dllm
--mask-token-id ID         # Mask token ID
```

### Inference Arguments

```bash
--tensor-parallel-size N   # Tensor parallel size
--data-parallel-size N     # Data parallel size
--gpu-memory-utilization F # GPU memory utilization (0.0-1.0)
--max-model-len N          # Maximum model length
```

### Sampling Arguments

```bash
--temperature F            # Sampling temperature
--max-tokens N             # Maximum tokens to generate
```

### Logging Arguments

```bash
--log-file PATH            # Log file path (optional)
--log-level LEVEL          # Log level: DEBUG, INFO, WARNING, ERROR
```

## Output

Results are saved to the output directory (default: `benchmark_results/`) with:

- Evaluation results in JSON format
- Detailed metrics and statistics
- Configuration used for the run
- Timestamp information

## Examples

### Example 1: GSM8K Evaluation

```bash
python -m diffulex_bench.main \
    --config diffulex_bench/configs/dream_d2f_gsm8k.yml \
    --dataset-limit 100
```

### Example 2: Custom Configuration

```bash
python -m diffulex_bench.main \
    --model-path /path/to/model \
    --model-name dream \
    --decoding-strategy d2f \
    --dataset gsm8k \
    --temperature 0.0 \
    --max-tokens 512 \
    --output-dir ./my_results \
    --log-file ./benchmark.log
```

### Example 3: Using Default Config

```bash
# If configs/example.yml exists, it will be used automatically
python -m diffulex_bench.main \
    --model-path /path/to/model \
    --dataset gsm8k
```

## Architecture

```
main.py (Entry Point)
    ↓
arg_parser.py (Argument Parsing)
    ↓
config.py (Configuration Management)
    ↓
run_benchmark() (Benchmark Execution)
    ↓
lm_eval.cli_evaluate() (Evaluation Framework)
    ↓
DiffulexLM (Model Interface)
    ↓
BenchmarkRunner (Engine Wrapper)
    ↓
Diffulex (Inference Engine)
```

## Advanced Usage

### Custom Model Integration

The framework uses `DiffulexLM` class which wraps `BenchmarkRunner`. You can extend it for custom models:

```python
from diffulex_bench.lm_eval_model import DiffulexLM

# DiffulexLM automatically registers with lm_eval
# Use it in lm_eval commands
```

### Programmatic Usage

```python
from diffulex_bench.config import BenchmarkConfig, EngineConfig, EvalConfig
from diffulex_bench.main import run_benchmark

# Load from YAML file
config = BenchmarkConfig.from_yaml("diffulex_bench/configs/example.yml")
run_benchmark(config)

# Or create programmatically
engine = EngineConfig(
    model_path="/path/to/model",
    model_name="dream",
    decoding_strategy="d2f",
)
eval_config = EvalConfig(
    dataset_name="gsm8k",
    temperature=0.0,
    max_tokens=256,
)
config = BenchmarkConfig(engine=engine, eval=eval_config)
run_benchmark(config)
```

## Troubleshooting

### Common Issues

1. **lm-eval not found**: Install with `pip install lm-eval`
2. **Config file not found**: Check path or use absolute path
3. **Model loading fails**: Verify model path and model_name match
4. **Out of memory**: Reduce `gpu_memory_utilization` or `max_model_len`

### Getting Help

- Check logs with `--log-level DEBUG`
- Save logs to file with `--log-file benchmark.log`
- Verify configuration with `--config` option

## Notes

1. The framework uses **lm-evaluation-harness** for all evaluation logic
2. Configuration files use **YAML** format (JSON also supported)
3. All evaluation metrics are computed by lm-eval
4. Results follow lm-eval output format
5. GPU environment is recommended for best performance
