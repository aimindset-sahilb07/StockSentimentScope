"""
Centralized logging configuration for StockSentimentScope.
"""
import logging
from typing import Optional

# Default logging format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logging(level: int = logging.INFO, format_str: Optional[str] = None):
    """
    Setup basic logging configuration for the application.
    
    Args:
        level: Logging level (default: INFO)
        format_str: Custom format string (default: None, uses DEFAULT_FORMAT)
    """
    logging.basicConfig(
        level=level,
        format=format_str or DEFAULT_FORMAT
    )
    
def get_logger(name: str):
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Name for the logger, typically __name__
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
