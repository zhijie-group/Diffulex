"""
Simple timer-based profiling backend.
"""
import time
from typing import Optional, Dict, Any

from diffulex_profiler.backends.base import ProfilerBackend


class SimpleTimerBackend(ProfilerBackend):
    """Simple timer-based profiling backend that only tracks time."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.current_name: Optional[str] = None
    
    def start(self, name: str) -> None:
        """Start timing."""
        self.current_name = name
        self.start_time = time.perf_counter()
    
    def stop(self) -> Optional[Dict[str, Any]]:
        """Stop timing and return duration."""
        if self.start_time is None:
            return None
        
        duration = time.perf_counter() - self.start_time
        result = {
            "duration_sec": duration,
            "name": self.current_name,
        }
        
        self.start_time = None
        self.current_name = None
        
        return result
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            self.stop()

