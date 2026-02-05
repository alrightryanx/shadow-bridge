"""
Ouroboros Dashboard routes.

Provides the Ouroboros self-healing pipeline dashboard with:
- Refiner process status and log viewing
- GitHub issue queue for ouroboros-labeled issues
- ADB device status
"""

from flask import Blueprint, render_template, jsonify, request
import json
import os
import sqlite3
import subprocess
import logging
import time
import uuid
import urllib.request
import urllib.error
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

ouroboros_bp = Blueprint('ouroboros', __name__)

# Paths
REFINER_STATE_PATH = r'C:\shadow\scripts\.ouroboros_refiner_state.json'
REFINER_LOG_PATH = r'C:\shadow\scripts\ouroboros_refiner.log'
ADB_PATH = r'C:\android\platform-tools\adb.exe'

# Database path (same as health_routes.py)
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    '..', 'backend', 'data', 'shadow_ai.db'
)


def _get_db():
    """Get a database connection."""
    return sqlite3.connect(DB_PATH)


def _ensure_deployed_fixes_table():
    """Create the deployed_fixes table if it doesn't exist."""
    try:
        conn = _get_db()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS deployed_fixes (
                id TEXT PRIMARY KEY,
                issue_number INTEGER NOT NULL,
                pattern_signature TEXT,
                fix_summary TEXT,
                ai_backend TEXT,
                deployed_at TEXT NOT NULL,
                verification_status TEXT DEFAULT 'pending',
                verified_at TEXT,
                notes TEXT
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to create deployed_fixes table: {e}")

# GitHub config
GITHUB_REPO = 'alrightryanx/shadow-android'
GITHUB_ISSUES_LABEL = 'ouroboros'

# Simple cache for GitHub API calls
_github_cache = {'data': None, 'timestamp': 0}
GITHUB_CACHE_TTL = 60  # seconds


@ouroboros_bp.route('/ouroboros')
def ouroboros_dashboard():
    """Render the Ouroboros pipeline dashboard."""
    return render_template('ouroboros.html')


@ouroboros_bp.route('/api/ouroboros/refiner/status')
def api_refiner_status():
    """Get Refiner process status, state file, and recent log activity."""
    result = {
        'alive': False,
        'last_run': None,
        'processed_count': 0,
        'processed_ids': [],
        'recent_activity': [],
    }

    # Read state file
    try:
        with open(REFINER_STATE_PATH, 'r') as f:
            state = json.load(f)
        result['last_run'] = state.get('last_run')
        result['processed_ids'] = state.get('processed_ids', [])
        result['processed_count'] = len(result['processed_ids'])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.debug(f"Could not read refiner state: {e}")

    # Check if refiner process is alive
    try:
        proc = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
            capture_output=True, text=True, timeout=5
        )
        result['alive'] = 'ouroboros_refiner' in proc.stdout.lower()
    except Exception as e:
        logger.debug(f"Could not check refiner process: {e}")

    # Parse recent log lines for activity summary
    try:
        lines = _read_last_lines(REFINER_LOG_PATH, 30)
        activity = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Parse log format: "2026-02-04 18:18:30,393 [INFO] message"
            parts = line.split('] ', 1)
            if len(parts) == 2:
                prefix = parts[0]
                message = parts[1]
                # Extract timestamp
                ts = prefix.split(' [')[0] if ' [' in prefix else ''
                # Extract level
                level = 'INFO'
                if '[ERROR]' in prefix:
                    level = 'ERROR'
                elif '[WARNING]' in prefix:
                    level = 'WARNING'
                elif '[DEBUG]' in prefix:
                    level = 'DEBUG'
                activity.append({
                    'timestamp': ts,
                    'level': level,
                    'message': message,
                })
        result['recent_activity'] = activity
    except Exception as e:
        logger.debug(f"Could not parse refiner log: {e}")

    return jsonify(result)


@ouroboros_bp.route('/api/ouroboros/refiner/log')
def api_refiner_log():
    """Return last N lines of the refiner log."""
    n = request.args.get('lines', 50, type=int)
    n = min(n, 500)  # Cap at 500 lines

    try:
        lines = _read_last_lines(REFINER_LOG_PATH, n)
        return jsonify({'lines': lines, 'count': len(lines)})
    except FileNotFoundError:
        return jsonify({'lines': [], 'count': 0, 'error': 'Log file not found'})
    except Exception as e:
        logger.error(f"Failed to read refiner log: {e}")
        return jsonify({'lines': [], 'count': 0, 'error': str(e)})


@ouroboros_bp.route('/api/ouroboros/github/issues')
def api_github_issues():
    """Fetch open ouroboros-labeled issues from GitHub (cached 60s)."""
    global _github_cache

    now = time.time()
    if _github_cache['data'] is not None and (now - _github_cache['timestamp']) < GITHUB_CACHE_TTL:
        return jsonify(_github_cache['data'])

    token = os.environ.get('GITHUB_TOKEN', '')
    url = f'https://api.github.com/repos/{GITHUB_REPO}/issues?labels={GITHUB_ISSUES_LABEL}&state=open&per_page=30'

    try:
        req = urllib.request.Request(url)
        req.add_header('Accept', 'application/vnd.github.v3+json')
        req.add_header('User-Agent', 'ShadowBridge-Ouroboros')
        if token:
            req.add_header('Authorization', f'token {token}')

        with urllib.request.urlopen(req, timeout=10) as resp:
            issues_raw = json.loads(resp.read().decode('utf-8'))

        issues = []
        for issue in issues_raw:
            issues.append({
                'number': issue.get('number'),
                'title': issue.get('title', ''),
                'state': issue.get('state', ''),
                'created_at': issue.get('created_at', ''),
                'updated_at': issue.get('updated_at', ''),
                'html_url': issue.get('html_url', ''),
                'labels': [l.get('name', '') for l in issue.get('labels', [])],
                'comments': issue.get('comments', 0),
            })

        result = {'issues': issues, 'count': len(issues)}
        _github_cache = {'data': result, 'timestamp': now}
        return jsonify(result)

    except urllib.error.HTTPError as e:
        logger.error(f"GitHub API error: {e.code} {e.reason}")
        return jsonify({'issues': [], 'count': 0, 'error': f'GitHub API: {e.code}'})
    except Exception as e:
        logger.error(f"Failed to fetch GitHub issues: {e}")
        return jsonify({'issues': [], 'count': 0, 'error': str(e)})


@ouroboros_bp.route('/api/ouroboros/adb/status')
def api_adb_status():
    """Run adb devices and return connected device list."""
    try:
        proc = subprocess.run(
            [ADB_PATH, 'devices'],
            capture_output=True, text=True, timeout=5
        )
        lines = proc.stdout.strip().split('\n')
        devices = []
        for line in lines[1:]:  # Skip header
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                devices.append({
                    'serial': parts[0],
                    'state': parts[1],
                })

        return jsonify({
            'devices': devices,
            'count': len(devices),
            'connected': any(d['state'] == 'device' for d in devices),
        })
    except FileNotFoundError:
        return jsonify({'devices': [], 'count': 0, 'connected': False, 'error': 'ADB not found'})
    except Exception as e:
        logger.error(f"Failed to get ADB status: {e}")
        return jsonify({'devices': [], 'count': 0, 'connected': False, 'error': str(e)})


@ouroboros_bp.route('/api/ouroboros/health-context')
def api_health_context():
    """Aggregated health data for the Ouroboros Refiner.

    Combines health score, matching crash patterns, and active trend alerts
    into a single response the Refiner can inject into AI prompts.
    """
    error_type = request.args.get('error_type', '')
    file_name = request.args.get('file_name', '')

    result = {
        'health_score': None,
        'crash_patterns': [],
        'trend_alerts': [],
    }

    try:
        conn = _get_db()

        # Health score - compute locally (same logic as health_routes fallback)
        try:
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

            success_rate = (successes / total * 100) if total > 0 else 100
            api_score = min(100, success_rate)
            perf_score = max(0, 100 - (avg_time / 300)) if avg_time else 100

            cursor2 = conn.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active
                FROM agents
            """)
            agent_row = cursor2.fetchone()
            agent_total = agent_row[0] or 0
            agent_active = agent_row[1] or 0
            agent_score = (agent_active / agent_total * 100) if agent_total > 0 else 100

            overall = api_score * 0.40 + perf_score * 0.30 + agent_score * 0.30
            status = 'healthy' if overall >= 80 else ('degraded' if overall >= 50 else 'critical')

            result['health_score'] = {
                'overall': round(overall, 1),
                'status': status,
                'api_reliability': round(api_score, 1),
                'response_performance': round(perf_score, 1),
                'agent_stability': round(agent_score, 1),
            }
        except Exception as e:
            logger.debug(f"Health score computation skipped: {e}")

        # Crash patterns - optionally filtered by error_type/file_name
        try:
            query = """
                SELECT id, error_type, file_name, method_name, line_range,
                       occurrence_count, first_seen, last_seen,
                       fix_attempts, root_cause_category, resolved, resolution_summary
                FROM crash_patterns
                WHERE 1=1
            """
            params = []
            if error_type:
                query += " AND error_type LIKE ?"
                params.append(f"%{error_type}%")
            if file_name:
                query += " AND file_name LIKE ?"
                params.append(f"%{file_name}%")
            query += " ORDER BY occurrence_count DESC LIMIT 20"

            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            result['crash_patterns'] = [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.debug(f"Crash patterns query skipped: {e}")

        # Active trend alerts
        try:
            cursor = conn.execute("""
                SELECT id, metric_name, severity, current_value, baseline_value,
                       deviation_pct, message, recommendation, created_at
                FROM trend_alerts
                WHERE resolved = 0
                ORDER BY created_at DESC
                LIMIT 20
            """)
            columns = [desc[0] for desc in cursor.description]
            result['trend_alerts'] = [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.debug(f"Trend alerts query skipped: {e}")

        conn.close()
    except Exception as e:
        logger.error(f"Failed to build health context: {e}")
        return jsonify({**result, 'error': str(e)})

    return jsonify(result)


@ouroboros_bp.route('/api/ouroboros/fix-deployed', methods=['POST'])
def api_fix_deployed():
    """Record a deployed fix for verification tracking.

    Accepts JSON: {issue_number, pattern_signature?, fix_summary?, ai_backend?, deployed_at?}
    """
    _ensure_deployed_fixes_table()

    data = request.get_json(silent=True)
    if not data or 'issue_number' not in data:
        return jsonify({'success': False, 'error': 'issue_number is required'}), 400

    fix_id = str(uuid.uuid4())[:12]
    deployed_at = data.get('deployed_at', datetime.now(timezone.utc).isoformat())

    try:
        conn = _get_db()
        conn.execute("""
            INSERT INTO deployed_fixes (id, issue_number, pattern_signature, fix_summary,
                                        ai_backend, deployed_at, verification_status)
            VALUES (?, ?, ?, ?, ?, ?, 'pending')
        """, (
            fix_id,
            int(data['issue_number']),
            data.get('pattern_signature', ''),
            data.get('fix_summary', ''),
            data.get('ai_backend', ''),
            deployed_at,
        ))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'fix_id': fix_id})
    except Exception as e:
        logger.error(f"Failed to record deployed fix: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@ouroboros_bp.route('/api/ouroboros/fix-verifications')
def api_fix_verifications():
    """Check verification status of deployed fixes.

    Joins deployed_fixes with crash_patterns to see if patterns recurred after fix.
    """
    _ensure_deployed_fixes_table()

    try:
        conn = _get_db()

        # Get all deployed fixes with optional pattern data
        cursor = conn.execute("""
            SELECT df.id, df.issue_number, df.pattern_signature, df.fix_summary,
                   df.ai_backend, df.deployed_at, df.verification_status,
                   df.verified_at, df.notes,
                   cp.occurrence_count, cp.last_seen, cp.resolved as pattern_resolved
            FROM deployed_fixes df
            LEFT JOIN crash_patterns cp ON df.pattern_signature = cp.id
            ORDER BY df.deployed_at DESC
            LIMIT 50
        """)
        columns = [desc[0] for desc in cursor.description]
        fixes = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()

        # Compute summary stats
        verified = sum(1 for f in fixes if f['verification_status'] == 'verified')
        pending = sum(1 for f in fixes if f['verification_status'] == 'pending')
        regressed = sum(1 for f in fixes if f['verification_status'] == 'regressed')

        return jsonify({
            'fixes': fixes,
            'summary': {
                'verified': verified,
                'pending': pending,
                'regressed': regressed,
                'total': len(fixes),
            }
        })
    except Exception as e:
        logger.error(f"Failed to fetch fix verifications: {e}")
        return jsonify({'fixes': [], 'summary': {'verified': 0, 'pending': 0, 'regressed': 0, 'total': 0}, 'error': str(e)})


@ouroboros_bp.route('/api/ouroboros/external/sync', methods=['POST'])
def api_external_sync():
    """Trigger the external vitals/crashlytics sync script."""
    try:
        provider = request.args.get('provider', 'all')
        script_path = r'C:\shadow\scripts\external_vitals_ingestor.py'
        
        # Run the script in the background
        # We use 'py' for Windows environments as per my instructions
        cmd = ['py', script_path, '--provider', provider]
        
        # Start the process without waiting
        subprocess.Popen(cmd)
        
        return jsonify({
            'success': True, 
            'message': f'External sync ({provider}) initiated in background.'
        })
    except Exception as e:
        logger.error(f"Failed to initiate external sync: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def _read_last_lines(filepath, n):
    """Read the last N lines of a file efficiently."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    return [line.rstrip('\n') for line in lines[-n:]]
