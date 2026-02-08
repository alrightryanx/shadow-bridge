"""Ouroboros proactive code health API routes for ShadowBridge.

Provides endpoints for the Ouroboros self-healing system to query
issues, health context, and report deployed fixes.
"""

import logging
from flask import Blueprint, jsonify, request

log = logging.getLogger(__name__)

ouroboros_bp = Blueprint("ouroboros", __name__)


@ouroboros_bp.route("/api/ouroboros/issues", methods=["GET"])
def get_issues():
    """GET /api/ouroboros/issues - List known issues for the refiner.

    Query params:
        status: Filter by status (open, fixed, ignored). Default: open
        project_id: Filter by project
        limit: Max results (default 20)
    """
    status = request.args.get("status", "open")
    project_id = request.args.get("project_id")
    limit = int(request.args.get("limit", 20))

    try:
        from web.services.code_health_monitor import get_code_health_monitor
        monitor = get_code_health_monitor()
        issues = monitor.get_issues(status=status, project_id=project_id, limit=limit)
        return jsonify(issues)
    except Exception as e:
        log.error(f"Failed to get ouroboros issues: {e}")
        return jsonify([])


@ouroboros_bp.route("/api/ouroboros/health-context", methods=["GET"])
def get_health_context():
    """GET /api/ouroboros/health-context - Health score + crash patterns.

    Returns aggregate health data for the Ouroboros refiner to use
    when deciding what to fix next.
    """
    try:
        from web.services.code_health_monitor import get_code_health_monitor
        monitor = get_code_health_monitor()
        context = monitor.get_health_context()
        return jsonify(context)
    except Exception as e:
        log.error(f"Failed to get health context: {e}")
        return jsonify({
            "health_score": 0.0,
            "crash_patterns": [],
            "error": str(e),
        })


@ouroboros_bp.route("/api/ouroboros/fix-deployed", methods=["POST"])
def fix_deployed():
    """POST /api/ouroboros/fix-deployed - Report a successful fix.

    Body:
        issue_id: ID of the fixed issue
        fix_description: What was done
        commit_hash: Optional git commit
    """
    data = request.get_json(silent=True) or {}
    issue_id = data.get("issue_id", "")
    fix_description = data.get("fix_description", "")
    commit_hash = data.get("commit_hash")

    if not issue_id:
        return jsonify({"error": "issue_id required"}), 400

    try:
        from web.services.code_health_monitor import get_code_health_monitor
        monitor = get_code_health_monitor()
        result = monitor.record_fix(issue_id, fix_description, commit_hash)
        return jsonify(result)
    except Exception as e:
        log.error(f"Failed to record fix: {e}")
        return jsonify({"error": str(e)}), 500


@ouroboros_bp.route("/api/ouroboros/code-health/<project_id>", methods=["GET"])
def get_code_health(project_id: str):
    """GET /api/ouroboros/code-health/<project_id> - Per-project health.

    Returns health analysis including dependency staleness, build health,
    doc freshness, and actionable suggestions.
    """
    try:
        from web.services.code_health_monitor import get_code_health_monitor
        monitor = get_code_health_monitor()
        health = monitor.get_project_health(project_id)
        return jsonify(health)
    except Exception as e:
        log.error(f"Failed to get code health for {project_id}: {e}")
        return jsonify({
            "project_id": project_id,
            "health_score": 0.0,
            "error": str(e),
        })
