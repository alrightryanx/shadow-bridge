"""
Health dashboard routes for Ouroboros V2.

Provides API endpoints for the health dashboard:
- Health score
- Trend alerts
- Crash patterns
- Circuit breaker status (proxied from backend)
"""

from flask import Blueprint, render_template, jsonify, request
import sqlite3
import os
import json
import logging
import requests as http_requests

logger = logging.getLogger(__name__)

health_bp = Blueprint('health', __name__)

# Database path
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    '..', 'backend', 'data', 'shadow_ai.db'
)

BACKEND_URL = os.environ.get('SHADOW_BACKEND_URL', 'http://localhost:3000')


def _get_db():
    """Get a database connection."""
    return sqlite3.connect(DB_PATH)


@health_bp.route('/health')
def health_dashboard():
    """Render the health dashboard page."""
    return render_template('health.html')


@health_bp.route('/api/health/score')
def api_health_score():
    """Proxy health score from the Node.js backend."""
    try:
        resp = http_requests.get(f'{BACKEND_URL}/v1/health/score', timeout=5)
        return jsonify(resp.json())
    except Exception as e:
        logger.error(f"Failed to fetch health score: {e}")
        # Compute locally as fallback
        return _compute_local_health_score()


@health_bp.route('/api/health/alerts')
def api_health_alerts():
    """Get active trend alerts."""
    try:
        conn = _get_db()
        cursor = conn.execute("""
            SELECT id, metric_name, severity, current_value, baseline_value,
                   deviation_pct, message, recommendation, created_at
            FROM trend_alerts
            WHERE resolved = 0
            ORDER BY created_at DESC
            LIMIT 50
        """)
        columns = [desc[0] for desc in cursor.description]
        alerts = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return jsonify({'alerts': alerts})
    except Exception as e:
        logger.error(f"Failed to fetch alerts: {e}")
        return jsonify({'alerts': [], 'error': str(e)})


@health_bp.route('/api/health/patterns')
def api_health_patterns():
    """Get top crash patterns."""
    try:
        conn = _get_db()
        cursor = conn.execute("""
            SELECT id, error_type, file_name, method_name, line_range,
                   occurrence_count, first_seen, last_seen,
                   root_cause_category, resolved, resolution_summary
            FROM crash_patterns
            ORDER BY occurrence_count DESC
            LIMIT 20
        """)
        columns = [desc[0] for desc in cursor.description]
        patterns = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return jsonify({'patterns': patterns})
    except Exception as e:
        logger.error(f"Failed to fetch patterns: {e}")
        return jsonify({'patterns': [], 'error': str(e)})


@health_bp.route('/api/health/circuits')
def api_health_circuits():
    """Proxy circuit breaker status from backend."""
    try:
        # Try to get from the backend's infrastructure endpoint
        # We need an auth token - use admin token if available
        headers = {}
        admin_token = os.environ.get('SHADOW_ADMIN_TOKEN')
        if admin_token:
            headers['Authorization'] = f'Bearer {admin_token}'

        resp = http_requests.get(
            f'{BACKEND_URL}/v1/ai/infrastructure/status',
            headers=headers,
            timeout=5
        )
        data = resp.json()
        return jsonify({'circuits': data.get('circuit_breakers', {})})
    except Exception as e:
        logger.debug(f"Circuit breaker fetch failed (expected if backend not running): {e}")
        return jsonify({'circuits': {}})


@health_bp.route('/api/health/alerts/<alert_id>/resolve', methods=['POST'])
def api_resolve_alert(alert_id):
    """Mark an alert as resolved."""
    try:
        conn = _get_db()
        conn.execute(
            "UPDATE trend_alerts SET resolved = 1, resolved_at = datetime('now') WHERE id = ?",
            (alert_id,)
        )
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Failed to resolve alert: {e}")
        return jsonify({'success': False, 'error': str(e)})


def _compute_local_health_score():
    """Fallback local health score computation."""
    try:
        conn = _get_db()

        # API stats
        cursor = conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                   AVG(response_time_ms) as avg_response_time
            FROM usage_analytics
            WHERE request_timestamp > datetime('now', '-24 hours')
        """)
        row = cursor.fetchone()
        total = row[0] or 0
        successes = row[1] or 0
        avg_time = row[2] or 0

        # Agent stats
        cursor2 = conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active
            FROM agents
        """)
        agent_row = cursor2.fetchone()
        agent_total = agent_row[0] or 0
        agent_active = agent_row[1] or 0

        conn.close()

        success_rate = (successes / total * 100) if total > 0 else 100
        api_score = min(100, success_rate)
        perf_score = max(0, 100 - (avg_time / 300)) if avg_time else 100
        agent_score = (agent_active / agent_total * 100) if agent_total > 0 else 100
        overall = api_score * 0.40 + perf_score * 0.30 + agent_score * 0.30

        status = 'healthy' if overall >= 80 else ('degraded' if overall >= 50 else 'critical')

        return jsonify({
            'overall': round(overall, 1),
            'categories': {
                'api_reliability': round(api_score, 1),
                'response_performance': round(perf_score, 1),
                'agent_stability': round(agent_score, 1),
                'active_alerts': 0
            },
            'status': status,
            'raw': {
                'total_requests': total,
                'success_rate': round(success_rate, 1),
                'avg_response_time_ms': round(avg_time),
                'active_agents': agent_active,
                'total_agents': agent_total
            }
        })
    except Exception as e:
        logger.error(f"Local health score failed: {e}")
        return jsonify({'overall': 0, 'status': 'critical', 'error': str(e)})
