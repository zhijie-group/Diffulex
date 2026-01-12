"""
Exporters for profiling results.
"""
from diffulex_profiler.exporters.base import ProfilerExporter
from diffulex_profiler.exporters.json import JSONExporter
from diffulex_profiler.exporters.summary import SummaryExporter

__all__ = [
    "ProfilerExporter",
    "JSONExporter",
    "SummaryExporter",
]

try:
    from diffulex_profiler.exporters.csv import CSVExporter
    __all__.append("CSVExporter")
except ImportError:
    pass

