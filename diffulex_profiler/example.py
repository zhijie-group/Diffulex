"""
Example usage of Diffulex Profiler.

This example demonstrates how to use the profiler to collect performance metrics
during Diffulex inference.
"""
from diffulex_profiler import DiffulexProfiler, ProfilerConfig
from diffulex import Diffulex, SamplingParams


def example_basic_usage():
    """Basic profiling example."""
    # Initialize profiler
    profiler = DiffulexProfiler(
        config=ProfilerConfig(
            enabled=True,
            backend="simple",
            output_dir="log/profiles",
            collect_gpu_metrics=True,
            collect_memory_metrics=True,
        )
    )
    
    # Initialize Diffulex engine
    model_path = "/path/to/your/model"
    llm = Diffulex(
        model_path,
        model_name="dream",
        tensor_parallel_size=1,
        data_parallel_size=1,
        gpu_memory_utilization=0.25,
        max_model_len=2048,
        decoding_strategy="d2f",
    )
    
    # Prepare prompts
    prompts = ["What is 2+2?", "Explain quantum computing"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    
    # Profile inference
    with profiler.profile("inference", metadata={"num_prompts": len(prompts)}):
        outputs = llm.generate(prompts, sampling_params)
        total_tokens = sum(len(o['token_ids']) for o in outputs)
        profiler.record_throughput(total_tokens)
        profiler.record_metric("num_outputs", len(outputs))
        profiler.record_metric("avg_diff_steps", 
                              sum(o['n_diff_steps'] for o in outputs) / len(outputs))
    
    # Export results
    profiler.export("inference_profile.json")
    
    # Get summary
    summary = profiler.get_summary()
    print(f"Total duration: {summary['total_duration_sec']:.2f}s")
    print(f"Average throughput: {summary['avg_throughput_tokens_per_sec']:.2f} tok/s")


def example_multiple_sections():
    """Example with multiple profiling sections."""
    profiler = DiffulexProfiler(
        config=ProfilerConfig(
            enabled=True,
            backend="simple",
            export_formats=["json", "csv", "summary"]
        )
    )
    
    # Profile model loading
    with profiler.profile("model_loading"):
        llm = Diffulex(model_path, model_name="dream", ...)
    
    # Profile prefill
    prompts = ["Prompt 1", "Prompt 2"]
    with profiler.profile("prefill", metadata={"num_prompts": len(prompts)}):
        # Prefill operations
        pass
    
    # Profile decode
    with profiler.profile("decode"):
        outputs = llm.generate(prompts, SamplingParams())
        profiler.record_throughput(sum(len(o['token_ids']) for o in outputs))
    
    # Export all results
    profiler.export("multi_section_profile.json")


def example_viztracer_backend():
    """Example using VizTracer backend for detailed tracing."""
    profiler = DiffulexProfiler(
        config=ProfilerConfig(
            enabled=True,
            backend="viztracer",
            viztracer_config={
                "output_file": "trace.json",
                "file_info": True,
            }
        )
    )
    
    with profiler.profile("detailed_trace"):
        # Your code here
        pass
    
    profiler.export()


def example_pytorch_profiler():
    """Example using PyTorch Profiler for GPU/CPU profiling."""
    from torch.profiler import ProfilerActivity
    
    profiler = DiffulexProfiler(
        config=ProfilerConfig(
            enabled=True,
            backend="pytorch",
            pytorch_profiler_config={
                "activities": [ProfilerActivity.CPU, ProfilerActivity.CUDA],
                "record_shapes": True,
                "profile_memory": True,
            }
        )
    )
    
    with profiler.profile("gpu_profiling"):
        # Your code here
        pass
    
    profiler.export()


if __name__ == "__main__":
    example_basic_usage()

