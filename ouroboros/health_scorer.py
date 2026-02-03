"""
Ouroboros V2 - Health Scorer

Computes a 0-100 health score across categories:
- api_reliability: Success rate of AI API calls
- response_performance: Average response time vs baseline
- cache_efficiency: Cache hit rate
- agent_stability: Agent uptime and task completion
- resource_usage: Token/cost efficiency

Pushes notifications when score drops below threshold.
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class HealthScore:
    """Composite health score with category breakdown."""
    overall: float
    api_reliability: float
    response_performance: float
    cache_efficiency: float
    agent_stability: float
    resource_usage: float
    timestamp: str
    status: str  # healthy, degraded, critical

    def to_dict(self) -> Dict:
        return asdict(self)


class HealthScorer:
    """
    Computes system health score from analytics data.

    Score ranges:
    - 80-100: Healthy (green)
    - 50-79: Degraded (yellow)
    - 0-49: Critical (red)
    """

    HEALTHY_THRESHOLD = 80
    DEGRADED_THRESHOLD = 50

    # Baseline targets
    TARGET_SUCCESS_RATE = 0.95        # 95% API success rate
    TARGET_RESPONSE_TIME_MS = 5000    # 5 second response time
    TARGET_CACHE_HIT_RATE = 0.30      # 30% cache hit rate
    TARGET_AGENT_UPTIME = 0.90        # 90% agent uptime

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def compute_score(self) -> HealthScore:
        """Compute the overall health score."""
        now = datetime.now()
        since_24h = (now - timedelta(hours=24)).isoformat()

        try:
            conn = self._get_connection()

            api_reliability = self._score_api_reliability(conn, since_24h)
            response_performance = self._score_response_performance(conn, since_24h)
            cache_efficiency = self._score_cache_efficiency(conn, since_24h)
            agent_stability = self._score_agent_stability(conn)
            resource_usage = self._score_resource_usage(conn, since_24h)

            conn.close()

            # Weighted average
            overall = (
                api_reliability * 0.30 +
                response_performance * 0.25 +
                cache_efficiency * 0.15 +
                agent_stability * 0.15 +
                resource_usage * 0.15
            )

            # Determine status
            if overall >= self.HEALTHY_THRESHOLD:
                status = 'healthy'
            elif overall >= self.DEGRADED_THRESHOLD:
                status = 'degraded'
            else:
                status = 'critical'

            return HealthScore(
                overall=round(overall, 1),
                api_reliability=round(api_reliability, 1),
                response_performance=round(response_performance, 1),
                cache_efficiency=round(cache_efficiency, 1),
                agent_stability=round(agent_stability, 1),
                resource_usage=round(resource_usage, 1),
                timestamp=now.isoformat(),
                status=status
            )

        except Exception as e:
            logger.error(f"[HealthScorer] Failed to compute score: {e}")
            return HealthScore(
                overall=0, api_reliability=0, response_performance=0,
                cache_efficiency=0, agent_stability=0, resource_usage=0,
                timestamp=now.isoformat(), status='critical'
            )

    def _score_api_reliability(self, conn: sqlite3.Connection, since: str) -> float:
        """Score based on API success rate (0-100)."""
        try:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
                FROM usage_analytics
                WHERE request_timestamp > ?
            """, (since,))
            row = cursor.fetchone()
            total = row[0] or 0
            successes = row[1] or 0

            if total == 0:
                return 100.0  # No data = assume healthy

            success_rate = successes / total
            # Scale: 95%+ = 100, 80% = 50, below 60% = 0
            if success_rate >= self.TARGET_SUCCESS_RATE:
                return 100.0
            elif success_rate >= 0.80:
                return 50.0 + (success_rate - 0.80) / (self.TARGET_SUCCESS_RATE - 0.80) * 50.0
            elif success_rate >= 0.60:
                return (success_rate - 0.60) / 0.20 * 50.0
            else:
                return max(0.0, success_rate * 50.0)

        except Exception as e:
            logger.error(f"API reliability scoring failed: {e}")
            return 50.0

    def _score_response_performance(self, conn: sqlite3.Connection, since: str) -> float:
        """Score based on average response time (0-100)."""
        try:
            cursor = conn.execute("""
                SELECT AVG(response_time_ms)
                FROM usage_analytics
                WHERE response_time_ms IS NOT NULL AND request_timestamp > ?
            """, (since,))
            avg_ms = cursor.fetchone()[0]

            if avg_ms is None:
                return 100.0

            # Scale: under 5s = 100, 10s = 50, over 30s = 0
            if avg_ms <= self.TARGET_RESPONSE_TIME_MS:
                return 100.0
            elif avg_ms <= 10000:
                return 50.0 + (10000 - avg_ms) / 5000 * 50.0
            elif avg_ms <= 30000:
                return (30000 - avg_ms) / 20000 * 50.0
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Response performance scoring failed: {e}")
            return 50.0

    def _score_cache_efficiency(self, conn: sqlite3.Connection, since: str) -> float:
        """
        Score based on cache efficiency.
        Since we don't have direct cache hit data in the DB,
        we approximate by looking at duplicate queries.
        """
        try:
            # Count total requests and unique queries in the window
            cursor = conn.execute("""
                SELECT COUNT(*) as total,
                       COUNT(DISTINCT model || ':' || COALESCE(working_directory, ''))
                       as unique_contexts
                FROM usage_analytics
                WHERE request_timestamp > ?
            """, (since,))
            row = cursor.fetchone()
            total = row[0] or 0
            unique = row[1] or 0

            if total == 0:
                return 100.0

            # If many requests go to varied contexts, cache won't help much
            # Higher ratio of total/unique means more cache potential
            reuse_ratio = (total - unique) / max(total, 1)

            # Score: 30%+ reuse = 100, 10% = 50, 0% = 30 (minimum since no-cache isn't a failure)
            if reuse_ratio >= self.TARGET_CACHE_HIT_RATE:
                return 100.0
            else:
                return 30.0 + (reuse_ratio / self.TARGET_CACHE_HIT_RATE) * 70.0

        except Exception as e:
            logger.error(f"Cache efficiency scoring failed: {e}")
            return 50.0

    def _score_agent_stability(self, conn: sqlite3.Connection) -> float:
        """Score based on agent health and uptime."""
        try:
            # Check agents table for status
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN status = 'killed' OR status = 'error' THEN 1 ELSE 0 END) as failed
                FROM agents
            """)
            row = cursor.fetchone()
            total = row[0] or 0
            active = row[1] or 0
            failed = row[2] or 0

            if total == 0:
                return 100.0

            # Score based on active ratio
            active_ratio = active / total
            if active_ratio >= self.TARGET_AGENT_UPTIME:
                return 100.0
            elif active_ratio >= 0.50:
                return 50.0 + (active_ratio - 0.50) / (self.TARGET_AGENT_UPTIME - 0.50) * 50.0
            else:
                return active_ratio * 100.0

        except Exception as e:
            logger.error(f"Agent stability scoring failed: {e}")
            return 50.0

    def _score_resource_usage(self, conn: sqlite3.Connection, since: str) -> float:
        """Score based on cost efficiency."""
        try:
            cursor = conn.execute("""
                SELECT AVG(total_cost), AVG(tokens_used)
                FROM usage_analytics
                WHERE request_timestamp > ?
            """, (since,))
            row = cursor.fetchone()
            avg_cost = row[0] or 0
            avg_tokens = row[1] or 0

            if avg_tokens == 0:
                return 100.0

            # Cost per 1k tokens
            cost_per_1k = (avg_cost / avg_tokens) * 1000 if avg_tokens > 0 else 0

            # Score: under $0.01/1k = 100, $0.05/1k = 50, over $0.10/1k = 0
            if cost_per_1k <= 0.01:
                return 100.0
            elif cost_per_1k <= 0.05:
                return 50.0 + (0.05 - cost_per_1k) / 0.04 * 50.0
            elif cost_per_1k <= 0.10:
                return (0.10 - cost_per_1k) / 0.05 * 50.0
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Resource usage scoring failed: {e}")
            return 50.0

    def push_notification_if_degraded(self, score: HealthScore, user_id: str = 'system') -> bool:
        """Push a notification if health score drops below threshold."""
        if score.overall >= self.HEALTHY_THRESHOLD:
            return False

        try:
            conn = self._get_connection()

            import uuid
            notification_id = str(uuid.uuid4())
            severity = 'warning' if score.status == 'degraded' else 'error'

            # Build message
            categories = []
            if score.api_reliability < 80:
                categories.append(f"API Reliability: {score.api_reliability}")
            if score.response_performance < 80:
                categories.append(f"Response Time: {score.response_performance}")
            if score.agent_stability < 80:
                categories.append(f"Agent Stability: {score.agent_stability}")

            message = (
                f"System health score dropped to {score.overall}/100 ({score.status}). "
                f"Degraded areas: {', '.join(categories) if categories else 'multiple metrics'}"
            )

            conn.execute("""
                INSERT INTO notifications (id, user_id, title, message, type, is_read, created_at)
                VALUES (?, ?, ?, ?, ?, 0, ?)
            """, (
                notification_id, user_id,
                f"Health Alert: {score.status.upper()}",
                message, severity, datetime.now().isoformat()
            ))
            conn.commit()
            conn.close()

            logger.warning(f"[HealthScorer] Notification pushed: {message}")
            return True

        except Exception as e:
            logger.error(f"[HealthScorer] Failed to push notification: {e}")
            return False
