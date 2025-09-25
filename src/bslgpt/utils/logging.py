"""Logging configuration for BSL GPT application."""

import logging
import sys
from typing import Any, Dict

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    if STRUCTLOG_AVAILABLE:
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )
    
    # Get the root logger and set level
    logging.getLogger().setLevel(numeric_level)


def get_logger(name: str):
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger
    """
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)