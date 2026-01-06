# Diffulex Profiler

A modular profiling framework for performance analysis of the Diffulex inference engine. This module provides comprehensive performance metrics collection, multiple profiling backends, and flexible result export capabilities.

## Features

- **Multiple Profiling Backends**: Support for simple timing, VizTracer, and PyTorch Profiler
- **Comprehensive Metrics**: Collect timing, throughput, GPU utilization, memory usage, and custom metrics
- **Flexible Export**: Export results in JSON, CSV, or human-readable summary formats
- **Easy Integration**: Simple context manager API for seamless integration with existing code
- **Modular Design**: Extensible architecture for adding custom backends and exporters

## Installation

The profiler is included as part of the Diffulex package. No additional installation is required beyond the standard Diffulex dependencies.

Optional dependencies for advanced features:
- `viztracer`: For detailed function call tracing (already in dependencies)
- `pynvml`: For detailed GPU utilization metrics (optional)

## Quick Start

### Basic Usage

```python
from diffulex_profiler import DiffulexProfiler, ProfilerConfig
from diffulex import Diffulex, SamplingParams

# Initialize profiler
profiler = DiffulexProfiler(
    config=ProfilerConfig(
        enabled=True,
        backend="simple",
        output_dir="log/profiles"
    )
)

# Initialize Diffulex engine
llm = Diffulex(model_path, model_name="dream", ...)

# Profile inference
with profiler.profile("inference", metadata={"batch_size": 10}):
    outputs = llm.generate(prompts, sampling_params)
    total_tokens = sum(len(o['token_ids']) for o in outputs)
    profiler.record_throughput(total_tokens)

# Export results
profiler.export("log/profiles/inference_profile.json")
```

### Advanced Usage with Multiple Sections

```python
profiler = DiffulexProfiler(
    config=ProfilerConfig(
        enabled=True,
        backend="simple",
        collect_gpu_metrics=True,
        collect_memory_metrics=True,
        export_formats=["json", "csv", "summary"]
    )
)

# Profile different sections
with profiler.profile("model_loading"):
    llm = Diffulex(model_path, ...)

with profiler.profile("prefill", metadata={"num_prompts": len(prompts)}):
    # Prefill phase
    pass

with profiler.profile("decode"):
    outputs = llm.generate(prompts, sampling_params)
    profiler.record_throughput(sum(len(o['token_ids']) for o in outputs))

# Get summary
summary = profiler.get_summary()
print(f"Total duration: {summary['total_duration_sec']:.2f}s")
print(f"Average throughput: {summary['avg_throughput_tokens_per_sec']:.2f} tok/s")

# Export all results
profiler.export()
```

## Configuration

### ProfilerConfig

The `ProfilerConfig` class provides comprehensive configuration options:

```python
@dataclass
class ProfilerConfig:
    enabled: bool = True                    # Enable/disable profiling
    backend: str = "simple"                 # Backend: "simple", "viztracer", "pytorch"
    output_dir: str = "log/profiles"        # Output directory for results
    output_file: Optional[str] = None       # Optional custom output filename
    collect_gpu_metrics: bool = True        # Collect GPU metrics
    collect_memory_metrics: bool = True     # Collect memory metrics
    collect_timing: bool = True             # Collect timing information
    export_formats: List[str] = ["json", "summary"]  # Export formats
    viztracer_config: Optional[Dict] = None # VizTracer-specific config
    pytorch_profiler_config: Optional[Dict] = None  # PyTorch Profiler config
```

## Profiling Backends

### Simple Timer Backend (Default)

The simplest backend that only tracks execution time. No additional dependencies required.

```python
profiler = DiffulexProfiler(
    config=ProfilerConfig(backend="simple")
)
```

### VizTracer Backend

For detailed function call tracing and visualization. Requires `viztracer` package.

```python
profiler = DiffulexProfiler(
    config=ProfilerConfig(
        backend="viztracer",
        viztracer_config={
            "output_file": "trace.json",
            "file_info": True,
        }
    )
)
```

### PyTorch Profiler Backend

For GPU/CPU operation-level profiling. Built into PyTorch.

```python
profiler = DiffulexProfiler(
    config=ProfilerConfig(
        backend="pytorch",
        pytorch_profiler_config={
            "activities": [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            "record_shapes": True,
            "profile_memory": True,
        }
    )
)
```

## Metrics Collection

The profiler automatically collects:

- **Timing**: Start time, end time, duration
- **Throughput**: Tokens per second (when recorded via `record_throughput()`)
- **GPU Metrics**: Utilization, memory usage, device information
- **Memory Metrics**: System memory usage and deltas
- **Custom Metrics**: User-defined metrics via `record_metric()`

### Recording Custom Metrics

```python
with profiler.profile("custom_section"):
    # Your code here
    profiler.record_metric("num_sequences", 10)
    profiler.record_metric("avg_length", 128.5)
    profiler.record_throughput(total_tokens=1000)
```

## Export Formats

### JSON Export

Structured JSON format suitable for programmatic analysis:

```python
profiler = DiffulexProfiler(
    config=ProfilerConfig(export_formats=["json"])
)
profiler.export("results.json")
```

### CSV Export

Tabular format for spreadsheet analysis:

```python
profiler = DiffulexProfiler(
    config=ProfilerConfig(export_formats=["csv"])
)
profiler.export("results.csv")
```

### Summary Export

Human-readable text summary:

```python
profiler = DiffulexProfiler(
    config=ProfilerConfig(export_formats=["summary"])
)
profiler.export("results.txt")
```

## Integration Examples

### Integration with Diffulex Engine

```python
from diffulex_profiler import DiffulexProfiler, ProfilerConfig
from diffulex import Diffulex, SamplingParams

# Setup
profiler = DiffulexProfiler(ProfilerConfig(enabled=True))
llm = Diffulex(model_path, model_name="dream", ...)
sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

# Profile generation
prompts = ["What is 2+2?", "Explain quantum computing"]
with profiler.profile("generate", metadata={"num_prompts": len(prompts)}):
    outputs = llm.generate(prompts, sampling_params)
    total_tokens = sum(len(o['token_ids']) for o in outputs)
    profiler.record_throughput(total_tokens)
    profiler.record_metric("num_outputs", len(outputs))
    profiler.record_metric("avg_diff_steps", 
                          sum(o['n_diff_steps'] for o in outputs) / len(outputs))

# Export
profiler.export("generation_profile.json")
summary = profiler.get_summary()
print(f"Throughput: {summary['avg_throughput_tokens_per_sec']:.2f} tok/s")
```

### Batch Profiling

```python
profiler = DiffulexProfiler(ProfilerConfig(enabled=True))

for batch_idx, batch in enumerate(batches):
    with profiler.profile(f"batch_{batch_idx}", metadata={"batch_size": len(batch)}):
        outputs = llm.generate(batch, sampling_params)
        profiler.record_throughput(sum(len(o['token_ids']) for o in outputs))

profiler.export("batch_profiles.json")
```

## API Reference

### DiffulexProfiler

Main profiler class.

#### Methods

- `profile(name: str, metadata: Optional[Dict] = None)`: Context manager for profiling
- `start(name: str, metadata: Optional[Dict] = None)`: Start profiling a section
- `stop()`: Stop profiling current section
- `record_metric(name: str, value: Any)`: Record a custom metric
- `record_throughput(tokens: int, duration: Optional[float] = None)`: Record throughput
- `export(output_path: Optional[str] = None)`: Export results
- `get_summary() -> Dict[str, Any]`: Get summary statistics
- `clear()`: Clear all collected metrics

### PerformanceMetrics

Container for performance metrics.

#### Attributes

- `name`: Section name
- `duration`: Duration in seconds
- `total_tokens`: Total tokens processed
- `throughput_tokens_per_sec`: Throughput in tokens/second
- `gpu_utilization`: GPU utilization percentage
- `memory_delta_mb`: Memory usage delta in MB
- `custom_metrics`: Dictionary of custom metrics
- `metadata`: User-provided metadata

## Best Practices

1. **Use Context Managers**: Always use the `profile()` context manager for automatic cleanup
2. **Record Throughput**: Call `record_throughput()` after inference to get accurate throughput metrics
3. **Add Metadata**: Include relevant metadata (batch size, model config, etc.) for better analysis
4. **Choose Appropriate Backend**: Use "simple" for basic timing, "viztracer" for detailed tracing, "pytorch" for GPU profiling
5. **Export Regularly**: Export results periodically for long-running experiments
6. **Clear When Needed**: Use `clear()` to reset metrics between different profiling sessions

## Troubleshooting

### Profiler Not Collecting Metrics

- Ensure `enabled=True` in `ProfilerConfig`
- Check that you're using the context manager correctly
- Verify that `start()` and `stop()` are called in pairs

### GPU Metrics Not Available

- Ensure CUDA is available: `torch.cuda.is_available()`
- Install `pynvml` for detailed GPU utilization: `pip install pynvml`

### Backend Import Errors

- Simple backend is always available
- VizTracer backend requires: `pip install viztracer`
- PyTorch Profiler is built into PyTorch

## Contributing

To add a new profiling backend:

1. Create a new class inheriting from `ProfilerBackend`
2. Implement `start()` and `stop()` methods
3. Add it to `backends/__init__.py`
4. Update `DiffulexProfiler._init_backend()` to support it

To add a new exporter:

1. Create a new class inheriting from `ProfilerExporter`
2. Implement `export()` method
3. Add it to `exporters/__init__.py`
4. Update `DiffulexProfiler._init_exporters()` to support it

## License

Same as the main Diffulex project.

