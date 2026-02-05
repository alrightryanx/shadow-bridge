"""
Tests for retry decorator and context manager.
"""

import pytest
import time
from unittest.mock import MagicMock, call

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shadow_bridge.utils.retry import retry_with_backoff, RetryContext


class TestRetryWithBackoff:
    """Tests for the retry_with_backoff decorator."""
    
    def test_success_on_first_try(self):
        """Function succeeds on first attempt, no retries needed."""
        mock_func = MagicMock(return_value="success")
        
        @retry_with_backoff(max_retries=3)
        def test_func():
            return mock_func()
        
        result = test_func()
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_success_after_failures(self):
        """Function fails twice then succeeds."""
        call_count = [0]
        
        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = flaky_func()
        
        assert result == "success"
        assert call_count[0] == 3
    
    def test_max_retries_exceeded(self):
        """Function always fails, raises after max retries."""
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            always_fails()
    
    def test_specific_exceptions_only(self):
        """Only specified exceptions trigger retry."""
        call_count = [0]
        
        @retry_with_backoff(
            max_retries=3, 
            base_delay=0.01,
            exceptions=(ConnectionError,)
        )
        def wrong_exception():
            call_count[0] += 1
            raise ValueError("Wrong type")
        
        with pytest.raises(ValueError):
            wrong_exception()
        
        # Should not retry for ValueError
        assert call_count[0] == 1
    
    def test_on_retry_callback(self):
        """on_retry callback is called before each retry."""
        callback = MagicMock()
        call_count = [0]
        
        @retry_with_backoff(
            max_retries=2,
            base_delay=0.01,
            on_retry=callback
        )
        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Fail")
            return "success"
        
        result = flaky_func()
        
        assert result == "success"
        assert callback.call_count == 2  # Called before 2nd and 3rd attempts
    
    def test_zero_retries(self):
        """max_retries=0 means no retries, just one attempt."""
        call_count = [0]
        
        @retry_with_backoff(max_retries=0)
        def no_retries():
            call_count[0] += 1
            raise ConnectionError("Fail")
        
        with pytest.raises(ConnectionError):
            no_retries()
        
        assert call_count[0] == 1
    
    def test_backoff_timing(self):
        """Verify exponential backoff timing is approximately correct."""
        start_time = time.time()
        call_count = [0]
        
        @retry_with_backoff(
            max_retries=2,
            base_delay=0.1,
            jitter=False  # Disable jitter for predictable timing
        )
        def timed_func():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ConnectionError("Fail")
            return "success"
        
        result = timed_func()
        elapsed = time.time() - start_time
        
        assert result == "success"
        # Should have waited ~0.1s + ~0.2s = ~0.3s total
        assert 0.2 < elapsed < 0.5


class TestRetryContext:
    """Tests for the RetryContext context manager."""
    
    def test_success_on_first_try(self):
        """Operation succeeds immediately."""
        attempts = 0
        
        with RetryContext(max_retries=3) as retry:
            while retry.should_continue():
                attempts += 1
                retry.success()
        
        assert attempts == 1
    
    def test_success_after_failures(self):
        """Operation fails then succeeds."""
        attempts = 0
        
        with RetryContext(max_retries=3, base_delay=0.01) as retry:
            while retry.should_continue():
                attempts += 1
                if attempts < 3:
                    retry.record_failure(ConnectionError("Fail"))
                else:
                    retry.success()
        
        assert attempts == 3
    
    def test_max_retries_exceeded(self):
        """Operation always fails, raises after max retries."""
        attempts = 0
        
        with pytest.raises(ConnectionError):
            with RetryContext(max_retries=2, base_delay=0.01) as retry:
                while retry.should_continue():
                    attempts += 1
                    retry.record_failure(ConnectionError("Always fails"))
        
        assert attempts == 3  # Initial + 2 retries
