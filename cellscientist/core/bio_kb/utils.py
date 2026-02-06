# -*- coding: utf-8 -*-
"""BioKB Utility Functions.

This module provides:
- Timeout decorators using ThreadPoolExecutor
- Graceful fallback decorators
- Common helper functions
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Optional, TypeVar


T = TypeVar('T')


def with_timeout(seconds: int, fallback: Optional[Any] = None) -> Callable:
    """Decorator to add timeout protection to a function using ThreadPoolExecutor.
    
    Args:
        seconds: Timeout in seconds
        fallback: Value to return on timeout (default: None)
        
    Returns:
        Decorated function that times out after specified seconds
        
    Example:
        @with_timeout(30, fallback=[])
        def query_api():
            return requests.get("https://api.example.com").json()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except FuturesTimeoutError:
                    return fallback
                except Exception:
                    return fallback
        return wrapper
    return decorator


def graceful_fallback(fallback_value: Any) -> Callable:
    """Decorator to return fallback value on any exception.
    
    Args:
        fallback_value: Value to return on exception
        
    Returns:
        Decorated function that returns fallback on error
        
    Example:
        @graceful_fallback([])
        def query_database():
            return db.query("SELECT * FROM table")
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception:
                return fallback_value
        return wrapper
    return decorator


def now_iso() -> str:
    """Get current UTC timestamp in ISO format.
    
    Returns:
        ISO 8601 timestamp string (e.g., "2026-02-06T15:30:00Z")
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha1_hash(s: str) -> str:
    """Generate SHA1 hash of a string.
    
    Args:
        s: Input string
        
    Returns:
        Hexadecimal SHA1 hash
    """
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def ensure_dir(path: str) -> None:
    """Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
    """
    if path:
        os.makedirs(path, exist_ok=True)
