"""
Ouroboros telemetry receiver endpoints.
Receives performance metrics from Shadow Android app for analysis and auto-improvement.
"""

from flask import Blueprint, request, jsonify
import re
import hmac
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

ouroboros_telemetry_bp = Blueprint('ouroboros_telemetry', __name__)

# Authorized devices (allowlist)
# In production, load from encrypted config file
AUTHORIZED_DEVICES = {
    "ZY22L3ZD8Q": {
        "name": "Dev Pixel",
        "token": None,  # Set during pairing
        "paired_at": None
    },
    "emulator-5554": {
        "name": "Android Emulator",
        "token": None,
        "paired_at": None
    }
}

# Whitelist for provider names
ALLOWED_PROVIDERS = {
    "claude-ssh", "SSH_CLAUDE_CODE",
    "gemini-api", "GEMINI_API",
    "openai-api", "OPENAI_API",
    "local-llm", "LOCAL_ON_DEVICE",
    "anthropic-api", "ANTHROPIC_API",
    "grok-api", "GROK_API"
}

# Whitelist for error types
ALLOWED_ERROR_TYPES = {
    "ConnectionTimeout", "RateLimitError", "AuthenticationError",
    "NetworkError", "ParseError", "UnknownError", "RequestCancelled",
    "UninitializedPropertyAccessException", "NullPointerException"
}

# Telemetry storage directory
TELEMETRY_DIR = Path(__file__).parent.parent.parent / "telemetry_data"
TELEMETRY_DIR.mkdir(exist_ok=True)


def validate_device_id(device_id: str) -> bool:
    """Validates device ID format (alphanumeric + dash only)."""
    if not device_id or len(device_id) > 50:
        return False
    return bool(re.match(r'^[a-zA-Z0-9\-]+$', device_id))


def validate_provider(provider: str) -> bool:
    """Validates provider name against whitelist."""
    return provider in ALLOWED_PROVIDERS


def validate_error_type(error_type: str) -> bool:
    """Validates error type against whitelist."""
    if not error_type:
        return True  # null/empty is valid
    return error_type in ALLOWED_ERROR_TYPES


def contains_injection_chars(value: str) -> bool:
    """Detects potential injection attack patterns."""
    if not value:
        return False

    dangerous_patterns = [
        r';.*?rm\s',           # Command injection
        r'\$\(',               # Command substitution
        r'`.*?`',              # Backtick execution
        r'\|\|',               # Command chaining
        r'&&',                 # Command chaining
        r'<script',            # XSS
        r'javascript:',        # XSS
        r'\.\./',              # Path traversal
        r'DROP\sTABLE',        # SQL injection
        r'--',                 # SQL comment
        r"';",                 # SQL injection
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            logger.warning(f"Injection pattern detected: {pattern} in value: {value[:50]}")
            return True

    return False


def validate_telemetry_payload(payload: dict) -> tuple[bool, str]:
    """
    Validates telemetry payload to prevent injection attacks.
    Returns (is_valid, error_message).
    """
    try:
        # Check required fields
        required_fields = ["device_id", "metrics"]
        if not all(field in payload for field in required_fields):
            return False, "Missing required fields"

        # Validate device ID format
        device_id = payload["device_id"]
        if not validate_device_id(device_id):
            return False, "Invalid device ID format"

        # Check if device is authorized
        # if device_id not in AUTHORIZED_DEVICES:
        #     logger.warning(f"Unauthorized device attempted telemetry: {device_id}")
        #     return False, "Device not authorized"

        # Validate metrics structure
        metrics = payload["metrics"]
        if not isinstance(metrics, dict):
            return False, "Invalid metrics format"

        # Validate provider name
        provider = metrics.get("provider", "")
        if not validate_provider(provider):
            logger.warning(f"Invalid provider: {provider}")
            return False, "Invalid provider name"

        # Validate error type if present
        error_type = metrics.get("error_type")
        if error_type and not validate_error_type(error_type):
            logger.warning(f"Invalid error type: {error_type}")
            return False, "Invalid error type"

        # Validate numeric fields
        for field in ["prompt_tokens", "response_tokens", "time_to_first_token", "total_time"]:
            if field in metrics:
                if not isinstance(metrics[field], (int, float)):
                    return False, f"Invalid type for {field}"
                if metrics[field] < 0:
                    return False, f"Negative value for {field}"

        # Check for injection attempts in string fields
        for field in ["provider", "error_type", "request_id"]:
            value = metrics.get(field, "")
            if value and contains_injection_chars(str(value)):
                return False, f"Injection attempt detected in {field}"

        return True, "Valid"

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False, "Validation exception"


@ouroboros_telemetry_bp.route('/api/telemetry/pair', methods=['POST'])
def pair_device():
    """
    Pairs a device with ShadowBridge for telemetry.
    Generates and stores auth token for device.
    """
    try:
        data = request.get_json()

        device_id = data.get('device_id')
        auth_token = data.get('auth_token')
        device_model = data.get('device_model', 'Unknown')

        # Validate device ID
        if not validate_device_id(device_id):
            return jsonify({"error": "Invalid device ID"}), 400

        # Check if device is in allowlist
        # if device_id not in AUTHORIZED_DEVICES:
        #     logger.warning(f"Pairing attempted by unauthorized device: {device_id}")
        #     return jsonify({"error": "Device not authorized"}), 403

        # Validate token format (64 hex chars)
        if not auth_token or not re.match(r'^[a-f0-9]{64}$', auth_token):
            return jsonify({"error": "Invalid token format"}), 400

        # Store token for device
        if device_id not in AUTHORIZED_DEVICES:
            AUTHORIZED_DEVICES[device_id] = {}

        AUTHORIZED_DEVICES[device_id]["token"] = auth_token
        AUTHORIZED_DEVICES[device_id]["paired_at"] = datetime.now().isoformat()
        AUTHORIZED_DEVICES[device_id]["model"] = device_model

        logger.info(f"Device paired successfully: {device_id} ({device_model})")

        return jsonify({
            "status": "paired",
            "device_id": device_id
        }), 200

    except Exception as e:
        logger.error(f"Pairing error: {e}")
        return jsonify({"error": "Pairing failed"}), 500


@ouroboros_telemetry_bp.route('/api/telemetry/ping', methods=['GET'])
def ping():
    """Simple ping endpoint to test connectivity."""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()}), 200


@ouroboros_telemetry_bp.route('/api/telemetry/response', methods=['POST'])
def receive_response_metrics():
    """
    Receives individual response metric from Android app.
    Validates and stores for analysis.
    """
    try:
        data = request.get_json()

        # Validate payload
        is_valid, error_msg = validate_telemetry_payload(data)
        if not is_valid:
            logger.warning(f"Invalid telemetry payload: {error_msg}")
            return jsonify({"error": error_msg}), 400

        # Extract and log metrics
        device_id = data["device_id"]
        metrics = data["metrics"]

        # Auto-register device (fix for server restarts clearing in-memory whitelist)
        if device_id and device_id not in AUTHORIZED_DEVICES:
            logger.info(f"Auto-registering device {device_id} from telemetry response")
            AUTHORIZED_DEVICES[device_id] = {
                "name": "Auto-Reconnected Device",
                "token": "reconnected-session",
                "paired_at": datetime.now().isoformat()
            }

        logger.info(f"Received telemetry from {device_id}: "
                   f"provider={metrics['provider']}, "
                   f"time={metrics['total_time']}ms, "
                   f"success={metrics['was_successful']}")

        # Store metrics to file (JSON Lines format)
        metrics_file = TELEMETRY_DIR / f"{device_id}_metrics.jsonl"
        with open(metrics_file, 'a') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "device_id": device_id,
                **metrics
            }) + "\n")

        # TODO: Trigger analysis if pattern detected (implement in ai_analyzer.py)
        # if is_slow_pattern(metrics):
        #     trigger_analysis(device_id, metrics)

        return jsonify({"status": "received"}), 200

    except Exception as e:
        logger.error(f"Error receiving telemetry: {e}")
        return jsonify({"error": "Internal server error"}), 500


@ouroboros_telemetry_bp.route('/api/telemetry/batch', methods=['POST'])
def receive_batch_metrics():
    """
    Receives batch of metrics from Android app.
    Used for syncing unsent telemetry.
    """
    try:
        data = request.get_json()

        device_id = data.get("device_id")
        metrics_list = data.get("metrics", [])

        # Validate device ID
        if not validate_device_id(device_id):
            return jsonify({"error": "Invalid device ID"}), 400

        # if device_id not in AUTHORIZED_DEVICES:
        #     return jsonify({"error": "Device not authorized"}), 403

        # Validate each metric
        valid_metrics = []
        for metric in metrics_list:
            payload = {"device_id": device_id, "metrics": metric}
            is_valid, _ = validate_telemetry_payload(payload)
            if is_valid:
                valid_metrics.append(metric)

        # Store valid metrics
        metrics_file = TELEMETRY_DIR / f"{device_id}_metrics.jsonl"
        with open(metrics_file, 'a') as f:
            for metric in valid_metrics:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "device_id": device_id,
                    **metric
                }) + "\n")

        logger.info(f"Received batch telemetry from {device_id}: "
                   f"{len(valid_metrics)}/{len(metrics_list)} valid metrics")

        return jsonify({
            "status": "received",
            "accepted": len(valid_metrics),
            "rejected": len(metrics_list) - len(valid_metrics)
        }), 200

    except Exception as e:
        logger.error(f"Error receiving batch telemetry: {e}")
        return jsonify({"error": "Internal server error"}), 500


@ouroboros_telemetry_bp.route('/api/telemetry/frustration', methods=['POST'])
def receive_frustration_pattern():
    """
    Receives frustration pattern detection from Android app.
    High priority - indicates user is experiencing issues.
    """
    try:
        data = request.get_json()

        device_id = data.get("device_id")
        pattern = data.get("pattern")
        severity = data.get("severity")
        description = data.get("description")

        # Validate device ID
        if not validate_device_id(device_id):
            return jsonify({"error": "Invalid device ID"}), 400

        # if device_id not in AUTHORIZED_DEVICES:
        #     return jsonify({"error": "Device not authorized"}), 403

        # Validate severity
        if severity not in ["low", "medium", "high", "critical"]:
            return jsonify({"error": "Invalid severity"}), 400

        logger.warning(f"FRUSTRATION PATTERN ({severity}): {device_id} - {description}")

        # Store frustration event
        frustration_file = TELEMETRY_DIR / f"{device_id}_frustration.jsonl"
        with open(frustration_file, 'a') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "device_id": device_id,
                "pattern": pattern,
                "severity": severity,
                "description": description,
                "events": data.get("events", [])
            }) + "\n")

        # TODO: Trigger immediate analysis for high/critical severity
        # if severity in ["high", "critical"]:
        #     trigger_urgent_analysis(device_id, data)

        return jsonify({"status": "received", "priority": "urgent"}), 200

    except Exception as e:
        logger.error(f"Error receiving frustration pattern: {e}")
        return jsonify({"error": "Internal server error"}), 500


@ouroboros_telemetry_bp.route('/api/telemetry/crash', methods=['POST'])
def receive_crash_report():
    """
    Receives crash report from Android app.
    CRITICAL PRIORITY - analyzes crash and creates GitHub issue with fix.
    """
    try:
        data = request.get_json()

        device_id = data.get("device_info", {}).get("device_id")

        # Validate device ID
        if not validate_device_id(device_id):
            return jsonify({"error": "Invalid device ID"}), 400

        # Auto-register device (fix for server restarts clearing in-memory whitelist)
        if device_id and device_id not in AUTHORIZED_DEVICES:
            logger.info(f"Auto-registering device {device_id} from crash report")
            AUTHORIZED_DEVICES[device_id] = {
                "name": "Auto-Reconnected Device",
                "token": "reconnected-session",
                "paired_at": datetime.now().isoformat()
            }

        # if device_id not in AUTHORIZED_DEVICES:
        #     return jsonify({"error": "Device not authorized"}), 403

        error_type = data.get("error", {}).get("type", "Unknown")
        file_name = data.get("error", {}).get("source_reference", {}).get("file_name", "Unknown")

        logger.critical(f"CRASH REPORT: {device_id} - {error_type} in {file_name}")

        # Store crash report
        crash_file = TELEMETRY_DIR / f"{device_id}_crashes.jsonl"
        with open(crash_file, 'a') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                **data
            }) + "\n")

        # Populate crash_patterns table for predictive engine
        _update_crash_patterns(data)

        # Trigger immediate analysis (background thread)
        import threading
        analysis_thread = threading.Thread(
            target=analyze_and_report_crash,
            args=(data,)
        )
        analysis_thread.daemon = True
        analysis_thread.start()

        return jsonify({
            "status": "received",
            "priority": "critical",
            "message": "Crash analysis initiated"
        }), 200

    except Exception as e:
        logger.error(f"Error receiving crash report: {e}")
        return jsonify({"error": "Internal server error"}), 500


def analyze_and_report_crash(crash_payload: dict):
    """
    Analyzes crash and creates GitHub issue (runs in background thread).
    """
    try:
        # Import here to avoid circular dependencies
        try:
            from ouroboros.crash_analyzer import CrashAnalyzer, GitHubReporter
        except ImportError:
            # Fallback for when running from within web/ directory
            import sys
            sys.path.append(str(Path(__file__).parent.parent.parent))
            from ouroboros.crash_analyzer import CrashAnalyzer, GitHubReporter

        # Get project root (shadow-android directory)
        project_root = str(Path(__file__).parent.parent.parent.parent / "shadow-android")

        # Analyze crash
        analyzer = CrashAnalyzer(project_root)
        analysis = analyzer.analyze_crash(crash_payload)

        if not analysis or not analysis.get("success"):
            logger.error(f"Crash analysis failed: {analysis.get('error') if analysis else 'Unknown'}")
            return

        # Create GitHub issue
        reporter = GitHubReporter(project_root)
        success = reporter.create_crash_issue(analysis)

        if success:
            logger.info("✓ Crash analysis complete - GitHub issue created")
        else:
            logger.error("✗ Failed to create GitHub issue")

    except Exception as e:
        logger.error(f"Crash analysis thread failed: {e}")
        import traceback
        traceback.print_exc()


def _update_crash_patterns(crash_data: dict):
    """
    Populate/update crash_patterns table from incoming crash reports.
    Enables the predictive engine to generate fix goals for recurring crashes.
    """
    import sqlite3
    import uuid

    db_path = Path("C:/shadow/backend/data/shadow_ai.db")
    if not db_path.exists():
        return

    try:
        error_info = crash_data.get("error", {})
        error_type = error_info.get("type", "Unknown")
        source_ref = error_info.get("source_reference", {})
        file_name = source_ref.get("file_name", "Unknown")
        method_name = source_ref.get("method_name", "Unknown")
        line_number = source_ref.get("line_number", 0)
        # Use a range around the line number for fuzzy matching
        line_range = f"{max(1, line_number - 5)}-{line_number + 5}" if line_number else ""

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check if this crash pattern already exists
        cursor.execute(
            """SELECT id, occurrence_count FROM crash_patterns
               WHERE error_type = ? AND file_name = ? AND method_name = ?""",
            [error_type, file_name, method_name],
        )
        existing = cursor.fetchone()

        now = datetime.now().isoformat()

        if existing:
            # Update occurrence count and last_seen
            cursor.execute(
                """UPDATE crash_patterns
                   SET occurrence_count = occurrence_count + 1,
                       last_seen = ?,
                       line_range = ?
                   WHERE id = ?""",
                [now, line_range, existing[0]],
            )
            logger.info(
                f"Updated crash pattern: {error_type} in {file_name}:{method_name} "
                f"(occurrences: {existing[1] + 1})"
            )
        else:
            # Insert new crash pattern
            pattern_id = str(uuid.uuid4())
            stack_trace = error_info.get("stack_trace", "")
            root_cause = _categorize_root_cause(error_type, stack_trace)

            cursor.execute(
                """INSERT INTO crash_patterns
                   (id, error_type, file_name, method_name, line_range,
                    occurrence_count, first_seen, last_seen, root_cause_category, resolved)
                   VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, 0)""",
                [pattern_id, error_type, file_name, method_name, line_range,
                 now, now, root_cause],
            )
            logger.info(f"New crash pattern: {error_type} in {file_name}:{method_name}")

        conn.commit()
        conn.close()

    except Exception as e:
        logger.warning(f"Failed to update crash_patterns: {e}")


def _categorize_root_cause(error_type: str, stack_trace: str) -> str:
    """Categorize the root cause of a crash based on error type and stack trace."""
    error_lower = error_type.lower()
    stack_lower = stack_trace.lower()

    if "nullpointer" in error_lower:
        return "null_reference"
    elif "outofmemory" in error_lower:
        return "memory"
    elif "stackoverflow" in error_lower:
        return "recursion"
    elif "classcast" in error_lower:
        return "type_mismatch"
    elif "index" in error_lower and "bound" in error_lower:
        return "bounds_check"
    elif "concurrent" in error_lower or "concurrent" in stack_lower:
        return "concurrency"
    elif "network" in error_lower or "socket" in error_lower or "connect" in error_lower:
        return "network"
    elif "io" in error_lower or "file" in error_lower:
        return "io"
    elif "security" in error_lower or "permission" in error_lower:
        return "security"
    elif "illegal" in error_lower and "state" in error_lower:
        return "state_management"
    else:
        return "unknown"
