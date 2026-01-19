"""
ShadowBridge Utility Functions
------------------------------
Common utilities including retry logic with exponential backoff.
"""

import functools
import logging
import random
import time
from typing import Callable, Type, Tuple, Optional, Any

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator for exponential backoff retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts (0 = no retries)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        exponential_base: Base for exponential growth (default 2.0)
        jitter: Add random jitter to prevent thundering herd
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback(exception, attempt_number) called before each retry
    
    Example:
        @retry_with_backoff(max_retries=3, exceptions=(ConnectionError, TimeoutError))
        def fetch_data():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt >= max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # Add jitter (±25%)
                    if jitter:
                        delay = delay * (0.75 + random.random() * 0.5)
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    if on_retry:
                        try:
                            on_retry(e, attempt + 1)
                        except Exception as callback_error:
                            logger.debug(f"on_retry callback error: {callback_error}")
                    
                    time.sleep(delay)
            
            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


def retry_with_backoff_async(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Async version of retry_with_backoff decorator.
    
    Same parameters as retry_with_backoff but for async functions.
    """
    import asyncio
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt >= max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
                        raise
                    
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    if jitter:
                        delay = delay * (0.75 + random.random() * 0.5)
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    if on_retry:
                        try:
                            on_retry(e, attempt + 1)
                        except Exception as callback_error:
                            logger.debug(f"on_retry callback error: {callback_error}")
                    
                    await asyncio.sleep(delay)
            
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


class RetryContext:
    """
    Context manager for retry logic when you need more control.
    
    Example:
        with RetryContext(max_retries=3) as retry:
            while retry.should_continue():
                try:
                    result = do_something()
                    break
                except ConnectionError as e:
                    retry.record_failure(e)
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.attempt = 0
        self.last_exception = None
        self._should_continue = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False  # Don't suppress exceptions
    
    def should_continue(self) -> bool:
        """Returns True if more attempts should be made."""
        return self._should_continue and self.attempt <= self.max_retries
    
    def record_failure(self, exception: Exception):
        """Record a failure and sleep before next attempt."""
        self.last_exception = exception
        self.attempt += 1
        
        if self.attempt > self.max_retries:
            self._should_continue = False
            raise exception
        
        delay = min(
            self.base_delay * (self.exponential_base ** (self.attempt - 1)),
            self.max_delay
        )
        if self.jitter:
            delay = delay * (0.75 + random.random() * 0.5)
        
        logger.warning(
            f"Attempt {self.attempt} failed: {exception}. "
            f"Retrying in {delay:.2f}s..."
        )
        time.sleep(delay)
    
    def success(self):
        """Mark operation as successful, stop retrying."""
        self._should_continue = False
