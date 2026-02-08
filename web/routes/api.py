"""REST API routes for the task management system."""

import logging
import time
from flask import Blueprint, request, jsonify

from web.services.task_store import get_task_store

log = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__, url_prefix='/api')


# ---- Task Endpoints ----

@api_bp.route('/tasks', methods=['GET'])
def list_tasks():
    """List all tasks with optional status/executor filters."""
    store = get_task_store()
    status = request.args.get('status')
    executor = request.args.get('executor')
    tasks = store.list_tasks(status=status, executor=executor)
    return jsonify({"tasks": tasks, "count": len(tasks)})


@api_bp.route('/tasks/pending', methods=['GET'])
def get_pending_tasks():
    """Get pending tasks for an executor to claim."""
    store = get_task_store()
    executor = request.args.get('executor')
    tasks = store.get_pending_tasks(executor=executor)
    return jsonify({"tasks": tasks, "count": len(tasks)})


@api_bp.route('/tasks', methods=['POST'])
def create_task():
    """Create a new task."""
    store = get_task_store()
    data = request.get_json(silent=True) or {}
    task = store.create_task(data)

    # Broadcast task created via WebSocket
    try:
        from web.routes.websocket import broadcast_task_update
        broadcast_task_update(task["id"], "CREATED", task)
    except Exception:
        pass

    return jsonify(task), 201


@api_bp.route('/tasks/<task_id>', methods=['GET'])
def get_task(task_id):
    """Get task details by ID."""
    store = get_task_store()
    task = store.get_task(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task)


@api_bp.route('/tasks/<task_id>', methods=['PUT'])
def update_task(task_id):
    """Update task fields."""
    store = get_task_store()
    data = request.get_json(silent=True) or {}
    task = store.update_task(task_id, data)
    if not task:
        return jsonify({"error": "Task not found"}), 404

    try:
        from web.routes.websocket import broadcast_task_update
        broadcast_task_update(task_id, "UPDATED", task)
    except Exception:
        pass

    return jsonify(task)


@api_bp.route('/tasks/<task_id>/claim', methods=['POST'])
def claim_task(task_id):
    """Claim a pending task. Sets executor and status to IN_PROGRESS."""
    store = get_task_store()
    data = request.get_json(silent=True) or {}
    executor = data.get("executor", "bridge")
    task = store.claim_task(task_id, executor)
    if not task:
        return jsonify({"error": "Task not found or already claimed"}), 409

    try:
        from web.routes.websocket import broadcast_task_update
        broadcast_task_update(task_id, "CLAIMED", task)
    except Exception:
        pass

    return jsonify(task)


@api_bp.route('/tasks/<task_id>/checkpoint', methods=['POST'])
def save_checkpoint(task_id):
    """Save checkpoint data for a task."""
    store = get_task_store()
    data = request.get_json(silent=True) or {}
    task = store.save_checkpoint(task_id, data)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify({"success": True, "task_id": task_id})


@api_bp.route('/tasks/<task_id>/events', methods=['GET'])
def get_events(task_id):
    """Get execution events for a task, with optional ?since= timestamp filter."""
    store = get_task_store()
    since = float(request.args.get('since', 0))
    events = store.get_events(task_id, since=since)
    return jsonify({"events": events, "count": len(events)})


@api_bp.route('/tasks/<task_id>/events', methods=['POST'])
def post_event(task_id):
    """Post an execution event for a task."""
    store = get_task_store()
    task = store.get_task(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404

    data = request.get_json(silent=True) or {}
    event = store.add_event(task_id, data)

    try:
        from web.routes.websocket import broadcast_agent_event
        broadcast_agent_event(task_id, event)
    except Exception:
        pass

    return jsonify(event), 201


@api_bp.route('/tasks/<task_id>/complete', methods=['POST'])
def complete_task(task_id):
    """Mark a task as completed with optional output."""
    store = get_task_store()
    data = request.get_json(silent=True) or {}
    output = data.get("output")
    task = store.complete_task(task_id, output=output)
    if not task:
        return jsonify({"error": "Task not found"}), 404

    try:
        from web.routes.websocket import broadcast_task_update
        broadcast_task_update(task_id, "COMPLETED", task)
    except Exception:
        pass

    return jsonify(task)


# ---- Agent Endpoints ----

@api_bp.route('/agents', methods=['GET'])
def list_agents():
    """List all agents."""
    store = get_task_store()
    agents = store.list_agents()
    return jsonify({"agents": agents, "count": len(agents)})


# ---- Team Endpoints ----

@api_bp.route('/teams', methods=['GET'])
def list_teams():
    """List all teams."""
    store = get_task_store()
    teams = store.list_teams()
    return jsonify({"teams": teams, "count": len(teams)})


# ---- Analysis Endpoint ----

@api_bp.route('/projects/<project_id>/analysis', methods=['GET'])
def analyze_project(project_id):
    """Run codebase analysis on a project's working directory."""
    try:
        from codebase_analyzer import CodebaseAnalyzer

        # Look up project path from projects state
        project_dir = request.args.get('path')
        if not project_dir:
            # Try to find from stored projects
            try:
                import shadow_bridge_gui
                state = shadow_bridge_gui.load_projects_state()
                for device in state.get("devices", {}).values():
                    for project in device.get("projects", []):
                        if project.get("id") == project_id:
                            project_dir = project.get("workingDirectory")
                            break
            except Exception:
                pass

        if not project_dir:
            return jsonify({"error": "Project directory not found. Pass ?path=/dir"}), 404

        import os
        if not os.path.isdir(project_dir):
            return jsonify({"error": f"Directory not found: {project_dir}"}), 404

        analyzer = CodebaseAnalyzer()
        report = analyzer.analyze(project_dir)
        return jsonify(report.to_dict())
    except Exception as e:
        log.error(f"Analysis failed for project {project_id}: {e}")
        return jsonify({"error": str(e)}), 500


# ---- Daemon Endpoint ----

@api_bp.route('/daemon/status', methods=['GET'])
def daemon_status():
    """Get agent daemon status."""
    try:
        from agent_daemon import get_daemon
        daemon = get_daemon()
        return jsonify(daemon.status)
    except Exception as e:
        return jsonify({"running": False, "error": str(e)})


# ---- Status Endpoint ----

@api_bp.route('/status', methods=['GET'])
def get_status():
    """System status: version, uptime, counts."""
    store = get_task_store()
    stats = store.get_stats()
    try:
        import shadow_bridge_gui
        version = shadow_bridge_gui.APP_VERSION
    except Exception:
        version = "unknown"

    return jsonify({
        "version": version,
        "status": "running",
        "uptime_seconds": stats["uptime_seconds"],
        "tasks": {
            "total": stats["total_tasks"],
            "pending": stats["pending"],
            "in_progress": stats["in_progress"],
            "completed": stats["completed"],
        },
        "agents": stats["total_agents"],
        "teams": stats["total_teams"],
        "timestamp": time.time(),
    })
