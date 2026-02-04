"""
Rate Limiter Service - AGI-Readiness Trust & Safety

Provides rate limiting for AI agent operations:
1. Per-agent rate limits
2. Per-action rate limits
3. Global system limits
4. Burst protection
5. Quota management

This prevents runaway AI operations and ensures fair resource usage.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from collections import defaultdict
import json
from pathlib import Path

from .agent_protocol import AgentId

logger = logging.getLogger(__name__)

# Persistence
RATE_CONFIG_FILE = Path.home() / ".shadowai" / "rate_limits.json"


class LimitScope(Enum):
    """Scope of rate limit."""
    GLOBAL = "global"           # Applies to all agents
    PER_AGENT = "per_agent"     # Per individual agent
    PER_ACTION = "per_action"   # Per action type
    PER_RESOURCE = "per_resource"  # Per resource being accessed


class LimitResult(Enum):
    """Result of rate limit check."""
    ALLOWED = "allowed"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    BURST_LIMITED = "burst_limited"


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit."""
    id: str
    name: str
    description: str
    scope: LimitScope

    # Limits
    requests_per_minute: Optional[int] = None
    requests_per_hour: Optional[int] = None
    requests_per_day: Optional[int] = None

    # Burst protection (max requests in sliding window)
    burst_limit: Optional[int] = None
    burst_window_seconds: int = 10

    # Quota (daily/monthly limits)
    daily_quota: Optional[int] = None
    monthly_quota: Optional[int] = None

    # Matching criteria
    agent_types: Optional[List[str]] = None  # None = all agents
    action_patterns: Optional[List[str]] = None  # Glob patterns for actions

    # State
    enabled: bool = True

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'scope': self.scope.value,
            'requests_per_minute': self.requests_per_minute,
            'requests_per_hour': self.requests_per_hour,
            'requests_per_day': self.requests_per_day,
            'burst_limit': self.burst_limit,
            'burst_window_seconds': self.burst_window_seconds,
            'daily_quota': self.daily_quota,
            'monthly_quota': self.monthly_quota,
            'agent_types': self.agent_types,
            'action_patterns': self.action_patterns,
            'enabled': self.enabled
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'RateLimitConfig':
        return cls(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            scope=LimitScope(data['scope']),
            requests_per_minute=data.get('requests_per_minute'),
            requests_per_hour=data.get('requests_per_hour'),
            requests_per_day=data.get('requests_per_day'),
            burst_limit=data.get('burst_limit'),
            burst_window_seconds=data.get('burst_window_seconds', 10),
            daily_quota=data.get('daily_quota'),
            monthly_quota=data.get('monthly_quota'),
            agent_types=data.get('agent_types'),
            action_patterns=data.get('action_patterns'),
            enabled=data.get('enabled', True)
        )


@dataclass
class RateLimitStatus:
    """Current status of rate limiting for an entity."""
    config_id: str
    result: LimitResult
    requests_remaining_minute: Optional[int] = None
    requests_remaining_hour: Optional[int] = None
    requests_remaining_day: Optional[int] = None
    quota_remaining_day: Optional[int] = None
    quota_remaining_month: Optional[int] = None
    reset_time: Optional[datetime] = None
    retry_after_seconds: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            'config_id': self.config_id,
            'result': self.result.value,
            'requests_remaining_minute': self.requests_remaining_minute,
            'requests_remaining_hour': self.requests_remaining_hour,
            'requests_remaining_day': self.requests_remaining_day,
            'quota_remaining_day': self.quota_remaining_day,
            'quota_remaining_month': self.quota_remaining_month,
            'reset_time': self.reset_time.isoformat() if self.reset_time else None,
            'retry_after_seconds': self.retry_after_seconds
        }


class RateLimiter:
    """
    Central rate limiting service for AI agents.

    Provides:
    - Sliding window rate limiting
    - Burst protection
    - Quota management
    - Per-agent and global limits
    - Usage tracking and analytics
    """

    def __init__(self):
        self._lock = Lock()

        # Rate limit configurations
        self._configs: Dict[str, RateLimitConfig] = {}

        # Request timestamps: key -> list of timestamps
        self._request_times: Dict[str, List[datetime]] = defaultdict(list)

        # Daily/monthly quotas: key -> (date/month, count)
        self._daily_usage: Dict[str, Tuple[str, int]] = {}
        self._monthly_usage: Dict[str, Tuple[str, int]] = {}

        # Metrics
        self._metrics = {
            'checks_performed': 0,
            'requests_allowed': 0,
            'requests_rate_limited': 0,
            'requests_quota_exceeded': 0,
            'requests_burst_limited': 0
        }

        self._load_config()
        self._create_default_limits()

    def _load_config(self) -> None:
        """Load rate limit configs from disk."""
        try:
            if RATE_CONFIG_FILE.exists():
                with open(RATE_CONFIG_FILE, 'r', encoding="utf-8") as f:
                    data = json.load(f)

                for config_data in data.get('configs', []):
                    config = RateLimitConfig.from_dict(config_data)
                    self._configs[config.id] = config

                logger.info(f"Loaded {len(self._configs)} rate limit configs")
        except Exception as e:
            logger.warning(f"Failed to load rate limits: {e}")

    def _save_config(self) -> None:
        """Save rate limit configs to disk."""
        try:
            RATE_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(RATE_CONFIG_FILE, 'w', encoding="utf-8") as f:
                json.dump({
                    'configs': [c.to_dict() for c in self._configs.values()],
                    'saved_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save rate limits: {e}")

    def _create_default_limits(self) -> None:
        """Create default rate limits if none exist."""
        if self._configs:
            return

        # Global system limits
        self.add_config(RateLimitConfig(
            id="global-minute",
            name="Global Per-Minute Limit",
            description="Limit all agents to 100 requests/minute combined",
            scope=LimitScope.GLOBAL,
            requests_per_minute=100,
            burst_limit=20,
            burst_window_seconds=5
        ))

        # Per-agent limits
        self.add_config(RateLimitConfig(
            id="agent-minute",
            name="Per-Agent Per-Minute Limit",
            description="Each agent limited to 30 requests/minute",
            scope=LimitScope.PER_AGENT,
            requests_per_minute=30,
            requests_per_hour=500
        ))

        # Daily quota per agent
        self.add_config(RateLimitConfig(
            id="agent-daily-quota",
            name="Agent Daily Quota",
            description="Each agent limited to 5000 operations/day",
            scope=LimitScope.PER_AGENT,
            daily_quota=5000
        ))

        # API call limits
        self.add_config(RateLimitConfig(
            id="external-api",
            name="External API Calls",
            description="Limit external API calls to 20/minute",
            scope=LimitScope.PER_ACTION,
            requests_per_minute=20,
            requests_per_hour=200,
            action_patterns=["external_api", "network_*"]
        ))

        # Code execution limits
        self.add_config(RateLimitConfig(
            id="code-execution",
            name="Code Execution",
            description="Limit code executions to 10/minute",
            scope=LimitScope.PER_ACTION,
            requests_per_minute=10,
            burst_limit=5,
            burst_window_seconds=10,
            action_patterns=["execute_code", "run_*"]
        ))

        self._save_config()
        logger.info("Created default rate limits")

    def add_config(self, config: RateLimitConfig) -> None:
        """Add or update a rate limit config."""
        with self._lock:
            self._configs[config.id] = config
            self._save_config()

    def remove_config(self, config_id: str) -> bool:
        """Remove a rate limit config."""
        with self._lock:
            if config_id in self._configs:
                del self._configs[config_id]
                self._save_config()
                return True
            return False

    def get_config(self, config_id: str) -> Optional[RateLimitConfig]:
        """Get a config by ID."""
        return self._configs.get(config_id)

    def list_configs(self) -> List[RateLimitConfig]:
        """List all rate limit configs."""
        return list(self._configs.values())

    def check_rate_limit(
        self,
        agent_id: AgentId,
        action: str,
        record: bool = True
    ) -> RateLimitStatus:
        """
        Check if a request is within rate limits.

        Args:
            agent_id: The requesting agent
            action: The action being performed
            record: Whether to record this request (set False for dry-run)

        Returns:
            RateLimitStatus with result and remaining quotas
        """
        with self._lock:
            self._metrics['checks_performed'] += 1
            now = datetime.now()
            agent_key = str(agent_id)

            # Check each applicable config
            for config in self._configs.values():
                if not config.enabled:
                    continue

                # Determine key based on scope
                if config.scope == LimitScope.GLOBAL:
                    key = f"global:{config.id}"
                elif config.scope == LimitScope.PER_AGENT:
                    key = f"agent:{agent_key}:{config.id}"
                elif config.scope == LimitScope.PER_ACTION:
                    if not self._matches_action(config, action):
                        continue
                    key = f"action:{action}:{config.id}"
                else:
                    continue

                # Check rate limits
                status = self._check_limits(config, key, now, record)

                if status.result != LimitResult.ALLOWED:
                    if status.result == LimitResult.RATE_LIMITED:
                        self._metrics['requests_rate_limited'] += 1
                    elif status.result == LimitResult.QUOTA_EXCEEDED:
                        self._metrics['requests_quota_exceeded'] += 1
                    elif status.result == LimitResult.BURST_LIMITED:
                        self._metrics['requests_burst_limited'] += 1

                    return status

            # All checks passed
            self._metrics['requests_allowed'] += 1
            return RateLimitStatus(
                config_id="none",
                result=LimitResult.ALLOWED
            )

    def _matches_action(self, config: RateLimitConfig, action: str) -> bool:
        """Check if action matches config patterns."""
        if not config.action_patterns:
            return True

        import fnmatch
        for pattern in config.action_patterns:
            if fnmatch.fnmatch(action.lower(), pattern.lower()):
                return True
        return False

    def _check_limits(
        self,
        config: RateLimitConfig,
        key: str,
        now: datetime,
        record: bool
    ) -> RateLimitStatus:
        """Check rate limits for a specific key."""
        # Clean old timestamps
        self._cleanup_timestamps(key, now)

        times = self._request_times.get(key, [])

        # Check burst limit
        if config.burst_limit:
            burst_cutoff = now - timedelta(seconds=config.burst_window_seconds)
            burst_count = len([t for t in times if t > burst_cutoff])

            if burst_count >= config.burst_limit:
                return RateLimitStatus(
                    config_id=config.id,
                    result=LimitResult.BURST_LIMITED,
                    retry_after_seconds=config.burst_window_seconds
                )

        # Check per-minute limit
        if config.requests_per_minute:
            minute_ago = now - timedelta(minutes=1)
            minute_count = len([t for t in times if t > minute_ago])

            if minute_count >= config.requests_per_minute:
                return RateLimitStatus(
                    config_id=config.id,
                    result=LimitResult.RATE_LIMITED,
                    requests_remaining_minute=0,
                    retry_after_seconds=60
                )

        # Check per-hour limit
        if config.requests_per_hour:
            hour_ago = now - timedelta(hours=1)
            hour_count = len([t for t in times if t > hour_ago])

            if hour_count >= config.requests_per_hour:
                return RateLimitStatus(
                    config_id=config.id,
                    result=LimitResult.RATE_LIMITED,
                    requests_remaining_hour=0,
                    retry_after_seconds=3600
                )

        # Check daily quota
        if config.daily_quota:
            today = now.strftime("%Y-%m-%d")
            quota_key = f"{key}:daily"
            date, count = self._daily_usage.get(quota_key, (today, 0))

            if date != today:
                count = 0

            if count >= config.daily_quota:
                return RateLimitStatus(
                    config_id=config.id,
                    result=LimitResult.QUOTA_EXCEEDED,
                    quota_remaining_day=0,
                    retry_after_seconds=self._seconds_until_midnight()
                )

            if record:
                self._daily_usage[quota_key] = (today, count + 1)

        # Check monthly quota
        if config.monthly_quota:
            month = now.strftime("%Y-%m")
            quota_key = f"{key}:monthly"
            m, count = self._monthly_usage.get(quota_key, (month, 0))

            if m != month:
                count = 0

            if count >= config.monthly_quota:
                return RateLimitStatus(
                    config_id=config.id,
                    result=LimitResult.QUOTA_EXCEEDED,
                    quota_remaining_month=0
                )

            if record:
                self._monthly_usage[quota_key] = (month, count + 1)

        # Record timestamp if allowed
        if record:
            self._request_times[key].append(now)

        # Calculate remaining
        minute_remaining = None
        hour_remaining = None

        if config.requests_per_minute:
            minute_ago = now - timedelta(minutes=1)
            minute_count = len([t for t in self._request_times[key] if t > minute_ago])
            minute_remaining = config.requests_per_minute - minute_count

        if config.requests_per_hour:
            hour_ago = now - timedelta(hours=1)
            hour_count = len([t for t in self._request_times[key] if t > hour_ago])
            hour_remaining = config.requests_per_hour - hour_count

        return RateLimitStatus(
            config_id=config.id,
            result=LimitResult.ALLOWED,
            requests_remaining_minute=minute_remaining,
            requests_remaining_hour=hour_remaining
        )

    def _cleanup_timestamps(self, key: str, now: datetime) -> None:
        """Remove timestamps older than 24 hours."""
        if key not in self._request_times:
            return

        cutoff = now - timedelta(hours=24)
        self._request_times[key] = [
            t for t in self._request_times[key] if t > cutoff
        ]

    def _seconds_until_midnight(self) -> int:
        """Calculate seconds until midnight."""
        now = datetime.now()
        midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return int((midnight - now).total_seconds())

    def get_usage(self, agent_id: AgentId) -> Dict[str, Any]:
        """Get usage statistics for an agent."""
        agent_key = str(agent_id)
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        month = now.strftime("%Y-%m")

        usage = {
            'agent_id': agent_key,
            'limits': {}
        }

        for config in self._configs.values():
            if config.scope != LimitScope.PER_AGENT:
                continue

            key = f"agent:{agent_key}:{config.id}"
            times = self._request_times.get(key, [])

            minute_ago = now - timedelta(minutes=1)
            hour_ago = now - timedelta(hours=1)

            limit_usage = {
                'config_name': config.name,
                'requests_last_minute': len([t for t in times if t > minute_ago]),
                'requests_last_hour': len([t for t in times if t > hour_ago])
            }

            if config.requests_per_minute:
                limit_usage['limit_per_minute'] = config.requests_per_minute

            if config.requests_per_hour:
                limit_usage['limit_per_hour'] = config.requests_per_hour

            if config.daily_quota:
                quota_key = f"{key}:daily"
                date, count = self._daily_usage.get(quota_key, (today, 0))
                if date == today:
                    limit_usage['daily_usage'] = count
                    limit_usage['daily_quota'] = config.daily_quota

            usage['limits'][config.id] = limit_usage

        return usage

    def reset_agent_limits(self, agent_id: AgentId) -> None:
        """Reset rate limits for an agent."""
        agent_key = str(agent_id)
        keys_to_remove = [k for k in self._request_times.keys() if agent_key in k]

        for key in keys_to_remove:
            del self._request_times[key]

        # Also reset quotas
        keys_to_remove = [k for k in self._daily_usage.keys() if agent_key in k]
        for key in keys_to_remove:
            del self._daily_usage[key]

        keys_to_remove = [k for k in self._monthly_usage.keys() if agent_key in k]
        for key in keys_to_remove:
            del self._monthly_usage[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            'total_configs': len(self._configs),
            'enabled_configs': len([c for c in self._configs.values() if c.enabled]),
            'tracked_keys': len(self._request_times),
            **self._metrics
        }


# Global instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter

    if _rate_limiter is None:
        _rate_limiter = RateLimiter()

    return _rate_limiter
