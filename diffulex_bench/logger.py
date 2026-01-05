"""
Logger module for diffulex_bench - Re-exports from diffulex.logger
"""

# Re-export logger functionality from diffulex core package
from diffulex.logger import (
    setup_logger,
    get_logger,
    LoggerMixin,
)

__all__ = [
    "setup_logger",
    "get_logger",
    "LoggerMixin",
]
