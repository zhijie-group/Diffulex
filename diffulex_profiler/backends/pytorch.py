"""
PyTorch Profiler backend.
"""
from typing import Optional, Dict, Any
from pathlib import Path

try:
    import torch
    from torch.profiler import profile, record_function, ProfilerActivity
    PYTORCH_PROFILER_AVAILABLE = True
except ImportError:
    PYTORCH_PROFILER_AVAILABLE = False
    profile = None
    record_function = None
    ProfilerActivity = None

from diffulex_profiler.backends.base import ProfilerBackend
from diffulex.logger import get_logger

logger = get_logger(__name__)


class PyTorchProfilerBackend(ProfilerBackend):
    """PyTorch Profiler-based backend for GPU/CPU operation profiling."""
    
    def __init__(self, output_dir: Optional[str] = None, activities: Optional[list] = None, **kwargs):
        if not PYTORCH_PROFILER_AVAILABLE:
            raise ImportError("PyTorch Profiler is not available")
        
        self.output_dir = Path(output_dir) if output_dir else Path("log/profiles")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if activities is None:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
        
        self.activities = activities
        self.config = kwargs
        self.profiler: Optional[profile] = None
        self.current_name: Optional[str] = None
    
    def start(self, name: str) -> None:
        """Start PyTorch Profiler."""
        if self.profiler is not None:
            logger.warning("PyTorch Profiler already started, stopping previous instance")
            self.stop()
        
        self.current_name = name
        self.profiler = profile(
            activities=self.activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            **self.config
        )
        self.profiler.__enter__()
    
    def stop(self) -> Optional[Dict[str, Any]]:
        """Stop PyTorch Profiler and export trace."""
        if self.profiler is None:
            return None
        
        self.profiler.__exit__(None, None, None)
        
        trace_file = self.output_dir / f"pytorch_trace_{self.current_name}.json"
        try:
            self.profiler.export_chrome_trace(str(trace_file))
        except Exception as e:
            logger.warning(f"Failed to export PyTorch trace: {e}")
            trace_file = None
        
        result = {
            "backend": "pytorch",
            "trace_file": str(trace_file) if trace_file else None,
            "name": self.current_name,
        }
        
        try:
            events = self.profiler.key_averages()
            result["summary"] = {
                "total_events": len(events),
                "cpu_time_total_ms": sum(e.cpu_time_total_us for e in events) / 1000,
                "cuda_time_total_ms": sum(e.cuda_time_total_us for e in events) / 1000 if torch.cuda.is_available() else 0,
            }
        except Exception as e:
            logger.warning(f"Failed to get profiler summary: {e}")
        
        self.profiler = None
        self.current_name = None
        
        return result
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler is not None:
            self.stop()

