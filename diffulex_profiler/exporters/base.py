"""
Base class for profiler exporters.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from diffulex_profiler.metrics import PerformanceMetrics


class ProfilerExporter(ABC):
    """Abstract base class for exporting profiling results."""
    
    @abstractmethod
    def export(self, metrics: List[PerformanceMetrics], output_path: Path) -> None:
        """
        Export metrics to a file.
        
        Args:
            metrics: List of performance metrics to export
            output_path: Base path for output (exporter may add extension)
        """
        pass

