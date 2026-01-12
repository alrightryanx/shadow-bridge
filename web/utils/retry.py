"""
Retry utilities with exponential backoff for resilient network operations.
"""

import time
from typing import Callable, Optional, Any
import logging

log = logging.getLogger(__name__)


class ExponentialBackoff:
    """Exponential backoff calculator for retry logic.

    Example:
        backoff = ExponentialBackoff(base=1.0, multiplier=1.5, max_delay=30.0)
        for attempt in range(max_retries):
            try:
                # Attempt operation
                break
            except Exception:
                if attempt < max_retries - 1:
                    delay = backoff.next()
                    time.sleep(delay)
    """

    def __init__(self, base: float = 1.0, multiplier: float = 1.5, max_delay: float = 30.0):
        """Initialize exponential backoff calculator.

        Args:
            base: Initial delay in seconds (default: 1.0)
            multiplier: Multiplier for each retry (default: 1.5)
            max_delay: Maximum delay in seconds (default: 30.0)
        """
        self.base = base
        self.multiplier = multiplier
        self.max_delay = max_delay
        self.attempts = 0

    def next(self) -> float:
        """Get next delay and increment attempt counter.

        Returns:
            Delay in seconds for next retry
        """
        delay = min(self.base * (self.multiplier ** self.attempts), self.max_delay)
        self.attempts += 1
        return delay

    def reset(self):
        """Reset attempt counter."""
        self.attempts = 0


def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
) -> Any:
    """Execute function with exponential backoff retry logic.

    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 30.0)
        exceptions: Tuple of exceptions to catch and retry (default: Exception)
        on_retry: Optional callback called on each retry: (attempt, exception, delay)

    Returns:
        Result of func() if successful

    Raises:
        Last exception if all retries exhausted

    Example:
        def risky_operation():
            # ... may fail ...
            return result

        result = retry_with_backoff(
            risky_operation,
            max_retries=5,
            base_delay=2.0,
            exceptions=(ConnectionError, TimeoutError)
        )
    """
    backoff = ExponentialBackoff(base=base_delay, multiplier=1.5, max_delay=max_delay)
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = backoff.next()
                if on_retry:
                    on_retry(attempt + 1, e, delay)
                else:
                    log.debug(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}")
                time.sleep(delay)
            else:
                # Last attempt failed - raise
                raise

    # Should never reach here but satisfy type checker
    if last_exception:
        raise last_exception
