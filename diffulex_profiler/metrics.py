"""
Performance metrics collection and data structures.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import torch

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Container for performance metrics collected during profiling."""
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    total_tokens: int = 0
    throughput_tokens_per_sec: float = 0.0
    gpu_metrics_start: Optional[Dict[str, Any]] = None
    gpu_metrics_end: Optional[Dict[str, Any]] = None
    gpu_utilization: float = 0.0
    memory_metrics_start: Optional[Dict[str, Any]] = None
    memory_metrics_end: Optional[Dict[str, Any]] = None
    memory_delta_mb: float = 0.0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    backend_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "name": self.name,
            "metadata": self.metadata,
            "duration_sec": self.duration,
            "total_tokens": self.total_tokens,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "gpu_utilization": self.gpu_utilization,
            "memory_delta_mb": self.memory_delta_mb,
            "custom_metrics": self.custom_metrics,
            "backend_data": self.backend_data,
        }


def collect_gpu_metrics() -> Dict[str, Any]:
    """Collect current GPU metrics."""
    if not torch.cuda.is_available():
        return {}
    
    metrics = {}
    try:
        device = torch.cuda.current_device()
        metrics["device"] = device
        metrics["device_name"] = torch.cuda.get_device_name(device)
        
        memory_stats = torch.cuda.memory_stats(device)
        metrics["allocated_mb"] = memory_stats.get("allocated_bytes.all.current", 0) / (1024 ** 2)
        metrics["reserved_mb"] = memory_stats.get("reserved_bytes.all.current", 0) / (1024 ** 2)
        metrics["peak_allocated_mb"] = memory_stats.get("allocated_bytes.all.peak", 0) / (1024 ** 2)
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics["utilization"] = util.gpu
            metrics["memory_utilization"] = util.memory
        except (ImportError, Exception):
            pass
        
    except Exception:
        pass
    
    return metrics


def collect_cpu_metrics() -> Dict[str, Any]:
    """Collect current CPU metrics."""
    if not PSUTIL_AVAILABLE:
        return {}
    try:
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_count": psutil.cpu_count(),
            "load_avg": psutil.getloadavg() if hasattr(psutil, "getloadavg") else None,
        }
    except Exception:
        return {}


def collect_memory_metrics() -> Dict[str, Any]:
    """Collect current memory metrics."""
    if not PSUTIL_AVAILABLE:
        return {}
    try:
        mem = psutil.virtual_memory()
        return {
            "total_mb": mem.total / (1024 ** 2),
            "available_mb": mem.available / (1024 ** 2),
            "used_mb": mem.used / (1024 ** 2),
            "percent": mem.percent,
        }
    except Exception:
        return {}

