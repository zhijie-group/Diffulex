"""
Professional logging setup with colored output for Diffulex
"""

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.traceback import install as install_rich_traceback
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import colorama
    from colorama import Fore, Style, init as init_colorama
    COLORAMA_AVAILABLE = True
    init_colorama(autoreset=True)
except ImportError:
    COLORAMA_AVAILABLE = False


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support"""
    
    if COLORAMA_AVAILABLE:
        COLORS = {
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Style.BRIGHT,
        }
    else:
        COLORS = {}
    
    RESET = Style.RESET_ALL if COLORAMA_AVAILABLE else ''
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "diffulex",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    use_rich: bool = True,
) -> logging.Logger:
    """
    Setup a professional logger with colored output
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        use_rich: Whether to use rich library for better formatting
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False  # Prevent propagation to root logger to avoid duplicate output
    
    # Use Rich if available and requested
    if use_rich and RICH_AVAILABLE:
        console = Console(stderr=True)
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
        handler.setFormatter(logging.Formatter(
            "%(message)s",
            datefmt="[%X]"
        ))
        logger.addHandler(handler)
        
        # Install rich traceback
        install_rich_traceback(show_locals=True)
    else:
        # Fallback to colored console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if COLORAMA_AVAILABLE:
            formatter = ColoredFormatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "diffulex") -> logging.Logger:
    """
    Get or create a logger
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Setup default logger if not already configured
        setup_logger(name)
    # Ensure propagate is False to avoid duplicate output
    logger.propagate = False
    return logger


class LoggerMixin:
    """Mixin class to add logger property to classes"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return get_logger(self.__class__.__module__)


# Add success method to logger
def _add_success_method():
    """Add success method to logging.Logger class"""
    if RICH_AVAILABLE:
        def success(self, message: str, *args, **kwargs):
            """Log success message with rich formatting"""
            self.info(f"[green]✓[/green] {message}", *args, **kwargs)
    else:
        def success(self, message: str, *args, **kwargs):
            """Log success message"""
            if COLORAMA_AVAILABLE:
                self.info(f"{Fore.GREEN}✓{Style.RESET_ALL} {message}", *args, **kwargs)
            else:
                self.info(f"✓ {message}", *args, **kwargs)
    
    if not hasattr(logging.Logger, 'success'):
        logging.Logger.success = success


# Initialize success method
_add_success_method()

