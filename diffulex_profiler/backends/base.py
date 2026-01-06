"""
Base class for profiling backends.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class ProfilerBackend(ABC):
    """Abstract base class for profiling backends."""
    
    @abstractmethod
    def start(self, name: str) -> None:
        """Start profiling a section."""
        pass
    
    @abstractmethod
    def stop(self) -> Optional[Dict[str, Any]]:
        """Stop profiling and return collected data."""
        pass
    
    @abstractmethod
    def __enter__(self):
        """Context manager entry."""
        pass
    
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass

