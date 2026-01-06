"""
CSV exporter for profiling results.
"""
import csv
from pathlib import Path
from typing import List

from diffulex_profiler.exporters.base import ProfilerExporter
from diffulex_profiler.metrics import PerformanceMetrics


class CSVExporter(ProfilerExporter):
    """Export profiling results to CSV format."""
    
    def export(self, metrics: List[PerformanceMetrics], output_path: Path) -> None:
        """Export metrics to CSV file."""
        output_file = output_path.with_suffix(".csv")
        
        if not metrics:
            return
        
        fieldnames = set(["name", "duration_sec", "total_tokens", "throughput_tokens_per_sec"])
        
        for m in metrics:
            fieldnames.update(m.custom_metrics.keys())
            if m.metadata:
                fieldnames.update(f"metadata_{k}" for k in m.metadata.keys())
        
        fieldnames = sorted(list(fieldnames))
        
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for m in metrics:
                row = {
                    "name": m.name,
                    "duration_sec": m.duration,
                    "total_tokens": m.total_tokens,
                    "throughput_tokens_per_sec": m.throughput_tokens_per_sec,
                }
                row.update(m.custom_metrics)
                for k, v in m.metadata.items():
                    row[f"metadata_{k}"] = v
                writer.writerow(row)

