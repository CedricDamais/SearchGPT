"""Caching utilities for SearchGPT."""

from functools import wraps
from typing import Any, Callable, Optional
import hashlib
import json


class SimpleCache:
    """Simple in-memory cache implementation."""
    
    def __init__(self, ttl: int = 3600):
        """
        Initialize cache.
        
        Args:
            ttl: Time to live in seconds
        """
        self._cache: dict[str, Any] = {}
        self.ttl = ttl
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self._cache[key] = value
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()


cache = SimpleCache()


def cached(ttl: Optional[int] = None):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds (uses global default if None)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = cache._generate_key(*args, **kwargs)
            result = cache.get(key)
            
            if result is None:
                result = func(*args, **kwargs)
                cache.set(key, result)
            
            return result
        return wrapper
    return decorator
