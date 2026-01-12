"""
Summary exporter for profiling results (human-readable text output).
"""
from pathlib import Path
from typing import List

from diffulex_profiler.exporters.base import ProfilerExporter
from diffulex_profiler.metrics import PerformanceMetrics
from diffulex.logger import get_logger

logger = get_logger(__name__)


class SummaryExporter(ProfilerExporter):
    """Export profiling results as a human-readable summary."""
    
    def export(self, metrics: List[PerformanceMetrics], output_path: Path) -> None:
        """Export metrics as a text summary."""
        output_file = output_path.with_suffix(".txt")
        
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("Diffulex Profiling Summary")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        total_duration = sum(m.duration for m in metrics if m.duration)
        total_tokens = sum(m.total_tokens for m in metrics if m.total_tokens)
        avg_throughput = (
            total_tokens / total_duration if total_duration > 0 and total_tokens > 0 else 0
        )
        
        summary_lines.append(f"Total Sections: {len(metrics)}")
        summary_lines.append(f"Total Duration: {total_duration:.2f} seconds")
        summary_lines.append(f"Total Tokens: {total_tokens}")
        summary_lines.append(f"Average Throughput: {avg_throughput:.2f} tokens/sec")
        summary_lines.append("")
        
        summary_lines.append("-" * 80)
        summary_lines.append("Section Details:")
        summary_lines.append("-" * 80)
        
        for m in metrics:
            summary_lines.append(f"\nSection: {m.name}")
            summary_lines.append(f"  Duration: {m.duration:.4f} seconds")
            if m.total_tokens > 0:
                summary_lines.append(f"  Tokens: {m.total_tokens}")
                summary_lines.append(f"  Throughput: {m.throughput_tokens_per_sec:.2f} tokens/sec")
            if m.gpu_utilization != 0:
                summary_lines.append(f"  GPU Utilization: {m.gpu_utilization:.2f}%")
            if m.memory_delta_mb != 0:
                summary_lines.append(f"  Memory Delta: {m.memory_delta_mb:.2f} MB")
            if m.custom_metrics:
                summary_lines.append(f"  Custom Metrics: {m.custom_metrics}")
            if m.metadata:
                summary_lines.append(f"  Metadata: {m.metadata}")
            if m.backend_data and m.backend_data.get("backend") == "viztracer":
                output_file = m.backend_data.get("output_file", "N/A")
                summary_lines.append(f"  VizTracer Output: {output_file}")
        
        summary_lines.append("")
        summary_lines.append("=" * 80)
        
        with open(output_file, "w") as f:
            f.write("\n".join(summary_lines))
        
        logger.info("\n".join(summary_lines))

