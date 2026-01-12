"""
JSON exporter for profiling results.
"""
import json
from pathlib import Path
from typing import List

from diffulex_profiler.exporters.base import ProfilerExporter
from diffulex_profiler.metrics import PerformanceMetrics


class JSONExporter(ProfilerExporter):
    """Export profiling results to JSON format."""
    
    def export(self, metrics: List[PerformanceMetrics], output_path: Path) -> None:
        """Export metrics to JSON file."""
        output_file = output_path.with_suffix(".json")
        
        data = {
            "metrics": [m.to_dict() for m in metrics],
            "summary": self._compute_summary(metrics),
        }
        
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def _compute_summary(self, metrics: List[PerformanceMetrics]) -> dict:
        """Compute summary statistics."""
        if not metrics:
            return {}
        
        total_duration = sum(m.duration for m in metrics if m.duration)
        total_tokens = sum(m.total_tokens for m in metrics if m.total_tokens)
        
        return {
            "total_sections": len(metrics),
            "total_duration_sec": total_duration,
            "total_tokens": total_tokens,
            "avg_throughput_tokens_per_sec": (
                total_tokens / total_duration if total_duration > 0 else 0
            ),
        }

