"""
Profiling backends for different profiling tools.
"""
from diffulex_profiler.backends.base import ProfilerBackend
from diffulex_profiler.backends.simple import SimpleTimerBackend

__all__ = [
    "ProfilerBackend",
    "SimpleTimerBackend",
]

# Optional backends
try:
    from diffulex_profiler.backends.viztracer import VizTracerBackend
    __all__.append("VizTracerBackend")
except ImportError:
    pass

try:
    from diffulex_profiler.backends.pytorch import PyTorchProfilerBackend
    __all__.append("PyTorchProfilerBackend")
except ImportError:
    pass

