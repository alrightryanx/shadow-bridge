"""REST API routes for the task management system."""

import logging
import time
from datetime import datetime
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

    # Track creator info
    data["source_ip"] = request.remote_addr
    if not data.get("created_by"):
        data["created_by"] = request.remote_addr

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


@api_bp.route('/tasks/<task_id>/heartbeat', methods=['POST'])
def heartbeat_task(task_id):
    """Heartbeat to extend lease on an active task."""
    store = get_task_store()
    task = store.heartbeat_task(task_id)
    if not task:
        return jsonify({"error": "Task not found or not in progress"}), 404
    return jsonify({"success": True, "task_id": task_id, "claimed_at": task.get("claimed_at")})


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

@api_bp.route('/websocket/clients', methods=['GET'])
def websocket_clients():
    """Get connected WebSocket clients and stale detection info."""
    from web.routes.websocket import get_connected_clients_summary, get_stale_clients, get_connected_client_count
    return jsonify({
        'clients': get_connected_clients_summary(),
        'stale_clients': get_stale_clients(),
        'total_connected': get_connected_client_count(),
        'timestamp': time.time(),
    })


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
            "failed": stats.get("failed", 0),
        },
        "agents": stats["total_agents"],
        "teams": stats["total_teams"],
        "timestamp": time.time(),
    })


# ---- Vector Store Endpoints ----

@api_bp.route('/vector/search', methods=['GET', 'POST'])
def vector_search():
    """Semantic search across vector store collections.

    POST body: {"query": "...", "collection": "shadowai_memory", "top_k": 5}
    GET params: ?q=...&limit=5&types=note,project
    Returns: {"results": [{"content": "...", "score": 0.95, "metadata": {...}}]}
    """
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        query = data.get('query', '')
        collection = data.get('collection')
        top_k = min(int(data.get('top_k', 5)), 50)
        source_types = None
    else:
        query = request.args.get('q', '')
        collection = request.args.get('collection')
        top_k = min(int(request.args.get('limit', request.args.get('top_k', 5))), 50)
        types_str = request.args.get('types', '')
        source_types = [t.strip() for t in types_str.split(',') if t.strip()] if types_str else None

    if not query:
        return jsonify({"error": "query field required"}), 400

    try:
        from web.services.vector_store import get_vector_store_v2
        store = get_vector_store_v2()
        if store is None:
            return jsonify({"error": "Vector store not available", "results": []}), 503

        collections = [collection] if collection else None
        results = store.hybrid_search(query, collections=collections, limit=top_k)

        # Filter by source_type if requested via GET ?types= parameter
        if source_types:
            results = [r for r in results if r.source_type in source_types]

        return jsonify({
            "results": [
                {
                    "id": r.id,
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata,
                    "source_type": r.source_type,
                    "source_id": r.source_id,
                    "title": r.title,
                    "collection": r.collection,
                }
                for r in results
            ],
            "count": len(results),
        })
    except Exception as e:
        log.error(f"Vector search failed: {e}")
        return jsonify({"error": str(e), "results": []}), 500


@api_bp.route('/vector/index', methods=['POST'])
def vector_index():
    """Index a document into the vector store.

    Body: {"content": "...", "metadata": {...}, "collection": "shadowai_memory",
           "source_type": "...", "source_id": "...", "title": "..."}
    """
    data = request.get_json(silent=True) or {}
    content = data.get('content', '')
    if not content:
        return jsonify({"error": "content field required"}), 400

    source_type = data.get('source_type', 'android')
    source_id = data.get('source_id', f"doc_{int(time.time())}")
    title = data.get('title', '')
    metadata = data.get('metadata', {})
    collection = data.get('collection', 'shadowai_memory')

    try:
        from web.services.vector_store import get_vector_store_v2
        store = get_vector_store_v2()
        if store is None:
            return jsonify({"error": "Vector store not available", "success": False}), 503

        success = store.index_document(
            source_type=source_type,
            source_id=source_id,
            title=title,
            content=content,
            metadata=metadata,
            collection=collection,
        )
        return jsonify({"success": success})
    except Exception as e:
        log.error(f"Vector index failed: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@api_bp.route('/vector/context', methods=['POST'])
def vector_context():
    """Build agent context from vector store search results.

    Body: {"query": "...", "max_tokens": 4000}
    Returns: {"context": "## Relevant Context\\n..."}
    """
    data = request.get_json(silent=True) or {}
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "query field required"}), 400

    max_tokens = min(int(data.get('max_tokens', 4000)), 16000)

    try:
        from web.services.vector_store import get_vector_store_v2
        store = get_vector_store_v2()
        if store is None:
            return jsonify({"error": "Vector store not available", "context": ""}), 503

        context = store.build_agent_context(query, max_tokens=max_tokens)
        return jsonify({"context": context})
    except Exception as e:
        log.error(f"Vector context build failed: {e}")
        return jsonify({"error": str(e), "context": ""}), 500


@api_bp.route('/vector/status', methods=['GET'])
def vector_status():
    """Get vector store availability and statistics."""
    try:
        from web.services.vector_store import get_vector_store_v2
        store = get_vector_store_v2()
        if store is None:
            return jsonify({"available": False, "error": "Vector store not initialized"})

        stats = store.get_collection_stats()
        total = sum(s["count"] for s in stats.values())
        return jsonify({
            "available": True,
            "document_count": total,
            "collections": stats,
            "embedding_model": "all-MiniLM-L6-v2",
        })
    except Exception as e:
        log.error(f"Vector status failed: {e}")
        return jsonify({"available": False, "error": str(e)})


# ---- Health & Tech Debt Endpoints ----

@api_bp.route('/health/ci-status', methods=['GET'])
def ci_status():
    """CI/CD health status endpoint."""
    try:
        from web.services.code_health_monitor import get_ci_status
        status = get_ci_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'UNKNOWN'}), 500


@api_bp.route('/health/tech-debt', methods=['GET'])
def tech_debt():
    """Technical debt scoring endpoint."""
    try:
        from web.services.code_health_monitor import calculate_tech_debt_score
        scores = calculate_tech_debt_score()
        return jsonify({
            'files': scores[:10],
            'total_files_analyzed': len(scores),
            'generated_at': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
