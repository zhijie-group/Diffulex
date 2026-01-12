"""
VizTracer profiling backend.
"""
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from viztracer import VizTracer
    VIZTRACER_AVAILABLE = True
except ImportError:
    VIZTRACER_AVAILABLE = False
    VizTracer = None

from diffulex_profiler.backends.base import ProfilerBackend
from diffulex.logger import get_logger

logger = get_logger(__name__)


class VizTracerBackend(ProfilerBackend):
    """VizTracer-based profiling backend for detailed function call tracing."""
    
    def __init__(self, output_file: Optional[str] = None, output_dir: Optional[str] = None, **kwargs):
        if not VIZTRACER_AVAILABLE:
            raise ImportError("VizTracer is not installed. Install it with: pip install viztracer")
        
        self.output_file = output_file
        self.output_dir = output_dir
        self.tracer: Optional[VizTracer] = None
        self.config = kwargs
    
    def start(self, name: str) -> None:
        """Start VizTracer."""
        if self.tracer is not None:
            logger.warning("VizTracer already started, stopping previous instance")
            self.stop()
        
        if self.output_file:
            output_file = self.output_file
        else:
            output_file = f"viztracer_{name}.json"
        
        # If output_dir is specified, prepend it to the output_file path
        if self.output_dir:
            output_file = str(Path(self.output_dir) / Path(output_file).name)
            # Ensure output directory exists
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VizTracer output file: {output_file}")
        self.tracer = VizTracer(output_file=output_file, **self.config)
        self.tracer.start()
    
    def stop(self) -> Optional[Dict[str, Any]]:
        """Stop VizTracer and return trace file path."""
        if self.tracer is None:
            return None
        
        self.tracer.stop()
        output_file = self.tracer.output_file
        
        result = {
            "backend": "viztracer",
            "output_file": str(output_file),
        }
        
        self.tracer = None
        return result
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tracer is not None:
            self.stop()

