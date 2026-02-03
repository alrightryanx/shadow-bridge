"""
Ouroboros V2 - Telemetry Trend Analyzer

Queries usage_analytics table for 24h/7d/30d windows.
Computes rolling averages: response_time_ms, error_rate, token usage.
Triggers WARNING (50% degradation vs baseline) or CRITICAL (100% degradation).
"""

import sqlite3
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TrendAlert:
    """Represents a trend-based alert."""
    id: str
    metric_name: str
    severity: str  # WARNING or CRITICAL
    current_value: float
    baseline_value: float
    deviation_pct: float
    message: str
    recommendation: str
    created_at: str


class TrendAnalyzer:
    """
    Analyzes usage analytics data for performance degradation trends.

    Monitors:
    - Response time (avg response_time_ms)
    - Error rate (% of failed requests)
    - Token usage (avg tokens per request)
    - Cost efficiency (cost per token)
    """

    # Thresholds
    WARNING_DEVIATION_PCT = 50.0   # 50% worse than baseline
    CRITICAL_DEVIATION_PCT = 100.0  # 100% worse (2x baseline)

    METRICS = [
        {
            'name': 'avg_response_time_ms',
            'query': "SELECT AVG(response_time_ms) FROM usage_analytics WHERE response_time_ms IS NOT NULL AND request_timestamp > ?",
            'direction': 'higher_is_worse',
            'unit': 'ms',
            'recommendation': 'Consider switching to a faster model or check network latency.'
        },
        {
            'name': 'error_rate',
            'query': "SELECT CAST(SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS FLOAT) / MAX(COUNT(*), 1) FROM usage_analytics WHERE request_timestamp > ?",
            'direction': 'higher_is_worse',
            'unit': '%',
            'recommendation': 'Check API key validity and provider status pages.'
        },
        {
            'name': 'avg_tokens_per_request',
            'query': "SELECT AVG(tokens_used) FROM usage_analytics WHERE request_timestamp > ?",
            'direction': 'higher_is_worse',
            'unit': 'tokens',
            'recommendation': 'Review prompt sizes and consider compression.'
        },
        {
            'name': 'avg_cost_per_request',
            'query': "SELECT AVG(total_cost) FROM usage_analytics WHERE request_timestamp > ?",
            'direction': 'higher_is_worse',
            'unit': '$',
            'recommendation': 'Consider using cheaper models for simple tasks.'
        }
    ]

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(self.db_path)

    def _query_metric(self, conn: sqlite3.Connection, query: str, since: datetime) -> Optional[float]:
        """Execute a metric query and return the result."""
        try:
            cursor = conn.execute(query, (since.isoformat(),))
            row = cursor.fetchone()
            return row[0] if row and row[0] is not None else None
        except Exception as e:
            logger.error(f"Metric query failed: {e}")
            return None

    def analyze_trends(self) -> List[TrendAlert]:
        """
        Run trend analysis comparing recent performance (24h) against baseline (30d).
        Returns list of alerts for degraded metrics.
        """
        alerts = []
        now = datetime.now()

        # Time windows
        recent_since = now - timedelta(hours=24)
        baseline_since = now - timedelta(days=30)

        try:
            conn = self._get_connection()

            for metric in self.METRICS:
                try:
                    # Get recent value (last 24h)
                    recent_value = self._query_metric(conn, metric['query'], recent_since)

                    # Get baseline value (last 30d)
                    baseline_value = self._query_metric(conn, metric['query'], baseline_since)

                    if recent_value is None or baseline_value is None:
                        continue
                    if baseline_value == 0:
                        continue

                    # Calculate deviation
                    if metric['direction'] == 'higher_is_worse':
                        deviation_pct = ((recent_value - baseline_value) / baseline_value) * 100
                    else:
                        deviation_pct = ((baseline_value - recent_value) / baseline_value) * 100

                    # Check thresholds
                    severity = None
                    if deviation_pct >= self.CRITICAL_DEVIATION_PCT:
                        severity = 'CRITICAL'
                    elif deviation_pct >= self.WARNING_DEVIATION_PCT:
                        severity = 'WARNING'

                    if severity:
                        alert = TrendAlert(
                            id=str(uuid.uuid4()),
                            metric_name=metric['name'],
                            severity=severity,
                            current_value=round(recent_value, 4),
                            baseline_value=round(baseline_value, 4),
                            deviation_pct=round(deviation_pct, 1),
                            message=f"{metric['name']} degraded by {deviation_pct:.1f}% "
                                    f"(current: {recent_value:.2f}{metric['unit']}, "
                                    f"baseline: {baseline_value:.2f}{metric['unit']})",
                            recommendation=metric['recommendation'],
                            created_at=now.isoformat()
                        )
                        alerts.append(alert)
                        logger.warning(f"[TrendAnalyzer] {severity}: {alert.message}")

                except Exception as e:
                    logger.error(f"Failed to analyze metric {metric['name']}: {e}")

            conn.close()

        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")

        return alerts

    def save_alerts(self, alerts: List[TrendAlert]) -> int:
        """Persist trend alerts to the database."""
        if not alerts:
            return 0

        try:
            conn = self._get_connection()
            saved = 0
            for alert in alerts:
                try:
                    conn.execute("""
                        INSERT INTO trend_alerts
                        (id, metric_name, severity, current_value, baseline_value,
                         deviation_pct, message, recommendation, resolved, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """, (
                        alert.id, alert.metric_name, alert.severity,
                        alert.current_value, alert.baseline_value,
                        alert.deviation_pct, alert.message,
                        alert.recommendation, alert.created_at
                    ))
                    saved += 1
                except Exception as e:
                    logger.error(f"Failed to save alert {alert.id}: {e}")

            conn.commit()
            conn.close()
            return saved

        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")
            return 0

    def get_active_alerts(self) -> List[Dict]:
        """Get all unresolved trend alerts."""
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT * FROM trend_alerts WHERE resolved = 0 ORDER BY created_at DESC LIMIT 50"
            )
            columns = [desc[0] for desc in cursor.description]
            alerts = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.close()
            return alerts
        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            return []

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        try:
            conn = self._get_connection()
            conn.execute(
                "UPDATE trend_alerts SET resolved = 1, resolved_at = ? WHERE id = ?",
                (datetime.now().isoformat(), alert_id)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
            return False

    def run(self) -> List[TrendAlert]:
        """Run analysis and persist any alerts. Returns new alerts."""
        alerts = self.analyze_trends()
        if alerts:
            self.save_alerts(alerts)
            logger.info(f"[TrendAnalyzer] Generated {len(alerts)} alerts")
        else:
            logger.info("[TrendAnalyzer] No degradation detected")
        return alerts
