"""
Caching utilities for StockSentimentScope.
Provides a consistent interface for caching across the application.
"""
import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

import streamlit as st

from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

def cached_data(
    ttl_seconds: Optional[int] = 3600,
    max_entries: Optional[int] = None,
    show_spinner: bool = False,
):
    """
    Decorator for caching function results using Streamlit's cache_data.
    
    Args:
        ttl_seconds: Time to live in seconds (None for indefinite)
        max_entries: Maximum number of entries to keep in cache
        show_spinner: Whether to show a spinner when computing
    
    Returns:
        Decorated function with caching behavior
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        @st.cache_data(ttl=ttl_seconds, max_entries=max_entries, show_spinner=show_spinner)
        def wrapper(*args, **kwargs) -> T:
            logger.debug(f"Cache call: {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator
