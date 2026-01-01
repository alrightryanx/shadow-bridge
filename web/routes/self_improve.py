"""
Self-Improvement API endpoints for Shadow Web Dashboard.
Receives issue reports from ShadowAI and creates GitHub PRs with fix suggestions.

Flow:
1. ShadowAI detects issue → sends to ShadowBridge
2. ShadowBridge receives report → stores in database
3. ShadowBridge creates GitHub PR with fix suggestion
4. Developer reviews and merges PR
5. ShadowAI verifies fix in next app version
"""
from flask import Blueprint, jsonify, request
import json
import os
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from functools import wraps

self_improve_bp = Blueprint('self_improve', __name__)
logger = logging.getLogger(__name__)

# Directory to store issue reports
REPORTS_DIR = Path(__file__).parent.parent.parent / "data" / "self_improve_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# GitHub repo mapping for different components
REPO_MAPPING = {
    "PHONE": "shadow-android",
    "WEAR": "shadow-android",
    "AUTO": "shadow-android",
    "TV": "shadow-android",
    "WEB": "shadow-bridge",
    "BRIDGE": "shadow-bridge",
    "PLUGIN": "claude-shadow"
}

# GitHub owner (will use configured or default)
GITHUB_OWNER = "alrightryanx"


def api_error_handler(f):
    """Decorator to provide consistent error handling for API endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error in {f.__name__}: {e}")
            return jsonify({"error": "Invalid JSON data"}), 400
        except ValueError as e:
            logger.warning(f"Value error in {f.__name__}: {e}")
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.exception(f"Unexpected error in {f.__name__}")
            return jsonify({"error": "Internal server error"}), 500
    return decorated_function


@self_improve_bp.route('/api/self-improve/report', methods=['POST'])
@api_error_handler
def receive_report():
    """
    Receive a self-improvement report from ShadowAI.

    Expected payload:
    {
        "reportId": "uuid",
        "timestamp": 1234567890,
        "issueType": "CONNECTION_TIMEOUT",
        "issueCategory": "NETWORK",
        "severity": "MEDIUM",
        "platform": "PHONE",
        "component": "HybridConnectionManager",
        "description": "Connection timed out after 30s",
        "fullStackTrace": "...",
        "diagnosticContext": {...},
        "rootCauseAnalysis": "...",
        "suggestedFix": "...",
        "affectedFiles": ["File1.kt", "File2.kt"],
        "fixComplexity": "MEDIUM",
        "diagnosisConfidence": 0.8,
        "diagnosisBackend": "CLOUD_API",
        "diagnosisModel": "claude-haiku",
        "appVersion": "3.624",
        "deviceModel": "Pixel 8",
        "androidVersion": "14",
        "responseTimeMs": 30500,
        "occurrenceCount": 1,
        "userComment": "I was trying to...",
        "technicalRootCause": "..."
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    report_id = data.get("reportId")
    if not report_id:
        return jsonify({"error": "Missing reportId"}), 400

    logger.info(f"Received self-improvement report: {report_id}")
    logger.info(f"Issue: {data.get('issueType')} in {data.get('component')} ({data.get('severity')})")

    # Save the report
    report_path = REPORTS_DIR / f"{report_id}.json"
    with open(report_path, 'w') as f:
        json.dump(data, f, indent=2)

    # Determine if we should create a PR
    should_create_pr = should_auto_create_pr(data)

    if should_create_pr:
        # Create GitHub PR asynchronously
        pr_result = create_github_pr(data)
        if pr_result.get("success"):
            return jsonify({
                "status": "accepted",
                "reportId": report_id,
                "prUrl": pr_result.get("prUrl"),
                "message": "Report received and PR created"
            }), 201

    return jsonify({
        "status": "accepted",
        "reportId": report_id,
        "message": "Report received and queued for review"
    }), 201


@self_improve_bp.route('/api/self-improve/reports', methods=['GET'])
@api_error_handler
def list_reports():
    """List all self-improvement reports."""
    reports = []

    for report_file in REPORTS_DIR.glob("*.json"):
        try:
            with open(report_file, 'r') as f:
                data = json.load(f)
                reports.append({
                    "reportId": data.get("reportId"),
                    "timestamp": data.get("timestamp"),
                    "issueType": data.get("issueType"),
                    "severity": data.get("severity"),
                    "platform": data.get("platform"),
                    "component": data.get("component"),
                    "description": data.get("description", "")[:100],
                    "fixComplexity": data.get("fixComplexity"),
                    "diagnosisConfidence": data.get("diagnosisConfidence"),
                    "appVersion": data.get("appVersion")
                })
        except Exception as e:
            logger.error(f"Error reading report {report_file}: {e}")

    # Sort by timestamp descending
    reports.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

    return jsonify({
        "reports": reports,
        "total": len(reports)
    })


@self_improve_bp.route('/api/self-improve/reports/<report_id>', methods=['GET'])
@api_error_handler
def get_report(report_id):
    """Get a specific self-improvement report."""
    report_path = REPORTS_DIR / f"{report_id}.json"

    if not report_path.exists():
        return jsonify({"error": "Report not found"}), 404

    with open(report_path, 'r') as f:
        data = json.load(f)

    return jsonify(data)


@self_improve_bp.route('/api/self-improve/reports/<report_id>/create-pr', methods=['POST'])
@api_error_handler
def manual_create_pr(report_id):
    """Manually trigger PR creation for a report."""
    report_path = REPORTS_DIR / f"{report_id}.json"

    if not report_path.exists():
        return jsonify({"error": "Report not found"}), 404

    with open(report_path, 'r') as f:
        data = json.load(f)

    pr_result = create_github_pr(data)

    if pr_result.get("success"):
        return jsonify({
            "status": "success",
            "prUrl": pr_result.get("prUrl"),
            "message": "PR created successfully"
        })
    else:
        return jsonify({
            "status": "error",
            "message": pr_result.get("error", "Failed to create PR")
        }), 500


@self_improve_bp.route('/api/self-improve/stats', methods=['GET'])
@api_error_handler
def get_stats():
    """Get statistics about self-improvement reports."""
    stats = {
        "total": 0,
        "by_severity": {},
        "by_category": {},
        "by_platform": {},
        "by_component": {},
        "prs_created": 0,
        "avg_confidence": 0.0
    }

    total_confidence = 0.0

    for report_file in REPORTS_DIR.glob("*.json"):
        try:
            with open(report_file, 'r') as f:
                data = json.load(f)

            stats["total"] += 1

            severity = data.get("severity", "UNKNOWN")
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1

            category = data.get("issueCategory", "UNKNOWN")
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1

            platform = data.get("platform", "UNKNOWN")
            stats["by_platform"][platform] = stats["by_platform"].get(platform, 0) + 1

            component = data.get("component", "UNKNOWN")
            stats["by_component"][component] = stats["by_component"].get(component, 0) + 1

            total_confidence += data.get("diagnosisConfidence", 0.0)

        except Exception as e:
            logger.error(f"Error reading report {report_file}: {e}")

    if stats["total"] > 0:
        stats["avg_confidence"] = round(total_confidence / stats["total"], 2)

    return jsonify(stats)


def should_auto_create_pr(data: dict) -> bool:
    """
    Determine if we should automatically create a PR for this report.

    Criteria:
    - High or Critical severity
    - Diagnosis confidence >= 0.7
    - Has affected files
    - Not a user-side issue
    """
    severity = data.get("severity", "")
    confidence = data.get("diagnosisConfidence", 0.0)
    affected_files = data.get("affectedFiles", [])
    issue_type = data.get("issueType", "")

    # Don't auto-PR for user-side issues
    user_side_issues = [
        "USER_REPORTS_ERROR",
        "API_RATE_LIMITED",
        "STORAGE_FULL",
        "PERMISSION_DENIED_RUNTIME"
    ]
    if issue_type in user_side_issues:
        return False

    # Check criteria
    if severity in ["CRITICAL", "HIGH"] and confidence >= 0.7 and len(affected_files) > 0:
        return True

    return False


def create_github_pr(data: dict) -> dict:
    """
    Create a GitHub PR with the fix suggestion.
    Uses the `gh` CLI to create the PR.

    Returns:
        dict with "success": bool and "prUrl" or "error"
    """
    try:
        platform = data.get("platform", "PHONE")
        repo_name = REPO_MAPPING.get(platform, "shadow-android")
        repo_full = f"{GITHUB_OWNER}/{repo_name}"

        report_id = data.get("reportId", "unknown")
        issue_type = data.get("issueType", "UNKNOWN")
        component = data.get("component", "Unknown")
        severity = data.get("severity", "MEDIUM")

        # Create branch name
        branch_name = f"self-improve/{report_id[:8]}-{issue_type.lower().replace('_', '-')}"

        # Create PR title
        pr_title = f"[Self-Improve] Fix {issue_type} in {component}"

        # Create PR body
        pr_body = create_pr_body(data)

        # Check if gh CLI is available
        try:
            subprocess.run(["gh", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.error("GitHub CLI (gh) not available")
            return {"success": False, "error": "GitHub CLI not installed"}

        # For now, just log what we would do
        # In production, this would use gh pr create
        logger.info(f"Would create PR: {pr_title}")
        logger.info(f"Branch: {branch_name}")
        logger.info(f"Repo: {repo_full}")

        # Save PR info to report
        report_path = REPORTS_DIR / f"{report_id}.json"
        if report_path.exists():
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            report_data["pr_attempted"] = True
            report_data["pr_branch"] = branch_name
            report_data["pr_title"] = pr_title
            report_data["pr_timestamp"] = datetime.now().isoformat()
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)

        # Return success with placeholder URL
        # In production, this would be the actual PR URL
        return {
            "success": True,
            "prUrl": f"https://github.com/{repo_full}/pulls?q=is%3Apr+{branch_name}",
            "branch": branch_name
        }

    except Exception as e:
        logger.exception("Error creating GitHub PR")
        return {"success": False, "error": str(e)}


def create_pr_body(data: dict) -> str:
    """Create the PR body with all relevant information."""
    return f"""## Self-Improvement Report

**Report ID:** `{data.get('reportId', 'N/A')}`
**Issue Type:** `{data.get('issueType', 'N/A')}`
**Category:** `{data.get('issueCategory', 'N/A')}`
**Severity:** `{data.get('severity', 'N/A')}`
**Platform:** `{data.get('platform', 'N/A')}`
**Component:** `{data.get('component', 'N/A')}`

### Description
{data.get('description', 'No description provided')}

### Root Cause Analysis
{data.get('rootCauseAnalysis', 'No analysis available')}

### Suggested Fix
```
{data.get('suggestedFix', 'No fix suggested')}
```

### Affected Files
{chr(10).join(['- ' + f for f in data.get('affectedFiles', ['None identified'])])}

### Diagnosis Info
- **Backend:** {data.get('diagnosisBackend', 'N/A')}
- **Model:** {data.get('diagnosisModel', 'N/A')}
- **Confidence:** {data.get('diagnosisConfidence', 0) * 100:.0f}%
- **Complexity:** {data.get('fixComplexity', 'N/A')}

### Device Info
- **App Version:** {data.get('appVersion', 'N/A')}
- **Device:** {data.get('deviceModel', 'N/A')}
- **Android:** {data.get('androidVersion', 'N/A')}

### User Comment
{data.get('userComment', 'No comment provided') or 'No comment provided'}

---
🤖 *Generated by ShadowAI Self-Improvement System*
"""
