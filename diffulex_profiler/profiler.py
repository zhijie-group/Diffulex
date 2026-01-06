"""
Core profiler implementation for Diffulex.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List
from contextlib import contextmanager
from pathlib import Path

import torch

from diffulex_profiler.metrics import PerformanceMetrics, collect_gpu_metrics, collect_memory_metrics
from diffulex_profiler.backends import ProfilerBackend, SimpleTimerBackend
from diffulex_profiler.exporters import ProfilerExporter, JSONExporter, SummaryExporter
from diffulex.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProfilerConfig:
    """Configuration for the profiler."""
    enabled: bool = True
    backend: str = "simple"  # "simple", "viztracer", "pytorch"
    output_dir: str = "log/profiles"
    output_file: Optional[str] = None
    collect_gpu_metrics: bool = True
    collect_memory_metrics: bool = True
    collect_timing: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "summary"])
    viztracer_config: Optional[Dict[str, Any]] = None
    pytorch_profiler_config: Optional[Dict[str, Any]] = None


class DiffulexProfiler:
    """
    Main profiler class for collecting performance metrics during Diffulex inference.
    
    Example:
        >>> profiler = DiffulexProfiler(config=ProfilerConfig(enabled=True))
        >>> with profiler.profile("inference"):
        ...     outputs = llm.generate(prompts, sampling_params)
        >>> profiler.export("log/profiles/result.json")
    """
    
    def __init__(self, config: Optional[ProfilerConfig] = None):
        self.config = config or ProfilerConfig()
        self.metrics: List[PerformanceMetrics] = []
        self.current_metrics: Optional[PerformanceMetrics] = None
        self.backend: Optional[ProfilerBackend] = None
        self.exporters: List[ProfilerExporter] = []
        
        if not self.config.enabled:
            return
            
        self._init_backend()
        self._init_exporters()
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _init_backend(self):
        """Initialize the profiling backend."""
        if self.config.backend == "simple":
            self.backend = SimpleTimerBackend()
        elif self.config.backend == "viztracer":
            try:
                from diffulex_profiler.backends import VizTracerBackend
                viztracer_config = self.config.viztracer_config or {}
                # Pass output_dir to VizTracerBackend so it can save files in the correct location
                if "output_dir" not in viztracer_config:
                    viztracer_config["output_dir"] = self.config.output_dir
                self.backend = VizTracerBackend(**viztracer_config)
            except ImportError:
                logger.warning("VizTracer not available, falling back to simple timer")
                self.backend = SimpleTimerBackend()
        elif self.config.backend == "pytorch":
            try:
                from diffulex_profiler.backends import PyTorchProfilerBackend
                pytorch_config = self.config.pytorch_profiler_config or {}
                self.backend = PyTorchProfilerBackend(**pytorch_config)
            except ImportError:
                logger.warning("PyTorch Profiler not available, falling back to simple timer")
                self.backend = SimpleTimerBackend()
        else:
            logger.warning(f"Unknown backend '{self.config.backend}', using simple timer")
            self.backend = SimpleTimerBackend()
    
    def _init_exporters(self):
        """Initialize exporters based on config."""
        for fmt in self.config.export_formats:
            if fmt == "json":
                self.exporters.append(JSONExporter())
            elif fmt == "csv":
                from diffulex_profiler.exporters import CSVExporter
                self.exporters.append(CSVExporter())
            elif fmt == "summary":
                self.exporters.append(SummaryExporter())
            else:
                logger.warning(f"Unknown export format '{fmt}', skipping")
    
    @contextmanager
    def profile(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for profiling a code block.
        
        Args:
            name: Name of the profiling section
            metadata: Optional metadata to attach to the metrics
            
        Example:
            >>> with profiler.profile("model_forward", {"batch_size": 32}):
            ...     output = model(input_ids)
        """
        if not self.config.enabled:
            yield
            return
        
        self.start(name, metadata)
        try:
            yield
        finally:
            self.stop()
    
    def start(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Start profiling a section."""
        if not self.config.enabled:
            return
        
        self.current_metrics = PerformanceMetrics(
            name=name,
            metadata=metadata or {},
        )
        
        if self.config.collect_timing:
            self.current_metrics.start_time = time.perf_counter()
        
        if self.backend:
            self.backend.start(name)
        
        if self.config.collect_gpu_metrics and torch.cuda.is_available():
            self.current_metrics.gpu_metrics_start = collect_gpu_metrics()
        
        if self.config.collect_memory_metrics:
            self.current_metrics.memory_metrics_start = collect_memory_metrics()
    
    def stop(self):
        """Stop profiling the current section."""
        if not self.config.enabled or self.current_metrics is None:
            return
        
        if self.config.collect_timing:
            self.current_metrics.end_time = time.perf_counter()
            self.current_metrics.duration = (
                self.current_metrics.end_time - self.current_metrics.start_time
            )
        
        if self.backend:
            backend_data = self.backend.stop()
            if backend_data:
                self.current_metrics.backend_data = backend_data
        
        if self.config.collect_gpu_metrics and torch.cuda.is_available():
            self.current_metrics.gpu_metrics_end = collect_gpu_metrics()
            if self.current_metrics.gpu_metrics_start and self.current_metrics.gpu_metrics_end:
                self.current_metrics.gpu_utilization = (
                    self.current_metrics.gpu_metrics_end.get("utilization", 0) -
                    self.current_metrics.gpu_metrics_start.get("utilization", 0)
                )
        
        if self.config.collect_memory_metrics:
            self.current_metrics.memory_metrics_end = collect_memory_metrics()
            if (self.current_metrics.memory_metrics_start and 
                self.current_metrics.memory_metrics_end):
                start_mem = self.current_metrics.memory_metrics_start.get("used_mb", 0)
                end_mem = self.current_metrics.memory_metrics_end.get("used_mb", 0)
                self.current_metrics.memory_delta_mb = end_mem - start_mem
        
        self.metrics.append(self.current_metrics)
        self.current_metrics = None
    
    def record_metric(self, name: str, value: Any):
        """Record a custom metric."""
        if not self.config.enabled or self.current_metrics is None:
            return
        self.current_metrics.custom_metrics[name] = value
    
    def record_throughput(self, tokens: int, duration: Optional[float] = None):
        """Record throughput in tokens per second."""
        if not self.config.enabled or self.current_metrics is None:
            return
        if duration is None:
            duration = self.current_metrics.duration
        if duration and duration > 0:
            self.current_metrics.throughput_tokens_per_sec = tokens / duration
            self.current_metrics.total_tokens = tokens
    
    def export(self, output_path: Optional[str] = None):
        """
        Export profiling results using configured exporters.
        
        Args:
            output_path: Optional custom output path. If not provided, uses config output_file
                         or generates one based on timestamp.
        """
        if not self.config.enabled or not self.metrics:
            logger.warning("No metrics to export")
            return
        
        if output_path is None:
            if self.config.output_file:
                output_path = str(Path(self.config.output_dir) / self.config.output_file)
            else:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = str(Path(self.config.output_dir) / f"profile_{timestamp}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        for exporter in self.exporters:
            try:
                exporter.export(self.metrics, output_path)
            except Exception as e:
                logger.error(f"Failed to export using {exporter.__class__.__name__}: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected metrics."""
        if not self.metrics:
            return {}
        
        total_duration = sum(m.duration for m in self.metrics if m.duration)
        total_tokens = sum(m.total_tokens for m in self.metrics if m.total_tokens)
        avg_throughput = (
            total_tokens / total_duration 
            if total_duration > 0 and total_tokens > 0 
            else 0
        )
        
        return {
            "total_sections": len(self.metrics),
            "total_duration_sec": total_duration,
            "total_tokens": total_tokens,
            "avg_throughput_tokens_per_sec": avg_throughput,
            "sections": [
                {
                    "name": m.name,
                    "duration_sec": m.duration,
                    "throughput_tokens_per_sec": m.throughput_tokens_per_sec,
                    "total_tokens": m.total_tokens,
                }
                for m in self.metrics
            ],
        }
    
    def clear(self):
        """Clear all collected metrics."""
        self.metrics.clear()
        self.current_metrics = None