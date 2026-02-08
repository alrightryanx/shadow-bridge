"""In-process IP-based rate limiter for the API.

Default: 120 requests/minute per IP. Localhost is exempt.
"""

import logging
import threading
import time
from flask import request, jsonify, current_app

log = logging.getLogger(__name__)

DEFAULT_RATE_LIMIT = 120  # requests per minute


class RateLimiter:
    """Thread-safe sliding-window rate limiter keyed by IP address."""

    def __init__(self, requests_per_minute: int = DEFAULT_RATE_LIMIT):
        self.rpm = requests_per_minute
        self._buckets: dict = {}  # ip -> list of timestamps
        self._lock = threading.Lock()

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        cutoff = now - 60

        with self._lock:
            timestamps = self._buckets.get(ip, [])
            # Prune old entries
            timestamps = [t for t in timestamps if t > cutoff]

            if len(timestamps) >= self.rpm:
                self._buckets[ip] = timestamps
                return False

            timestamps.append(now)
            self._buckets[ip] = timestamps
            return True

    def get_remaining(self, ip: str) -> int:
        now = time.time()
        cutoff = now - 60
        with self._lock:
            timestamps = self._buckets.get(ip, [])
            recent = [t for t in timestamps if t > cutoff]
            return max(0, self.rpm - len(recent))


# Module-level singleton
_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """Return the singleton rate limiter."""
    return _limiter


def configure_rate_limit(rpm: int):
    """Reconfigure the rate limit (called from config loading)."""
    global _limiter
    _limiter = RateLimiter(rpm)


def _is_localhost(remote_addr: str) -> bool:
    return remote_addr in ("127.0.0.1", "::1", "localhost")


def check_rate_limit():
    """Flask before_request hook that enforces rate limits on /api/* routes."""
    if current_app.config.get("TESTING"):
        return None

    if not request.path.startswith("/api/"):
        return None

    if _is_localhost(request.remote_addr):
        return None

    if not _limiter.is_allowed(request.remote_addr):
        log.warning(f"Rate limit exceeded for {request.remote_addr}")
        return jsonify({
            "error": "Rate limit exceeded",
            "retry_after_seconds": 60,
        }), 429

    return None
