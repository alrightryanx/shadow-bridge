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
import subprocess
import logging
import time
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

ouroboros_bp = Blueprint('ouroboros', __name__)

# Paths
REFINER_STATE_PATH = r'C:\shadow\scripts\.ouroboros_refiner_state.json'
REFINER_LOG_PATH = r'C:\shadow\scripts\ouroboros_refiner.log'
ADB_PATH = r'C:\android\platform-tools\adb.exe'

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


def _read_last_lines(filepath, n):
    """Read the last N lines of a file efficiently."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    return [line.rstrip('\n') for line in lines[-n:]]
