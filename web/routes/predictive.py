"""Predictive intelligence API routes for ShadowBridge.

These endpoints are called by BriefingEngine.kt on Android to fetch
predictions and engine status. Also provides action recording and
routine management endpoints.
"""

import logging
from flask import Blueprint, jsonify, request

log = logging.getLogger(__name__)

predictive_bp = Blueprint("predictive", __name__)


@predictive_bp.route("/api/predictive/predictions", methods=["GET"])
def get_predictions():
    """GET /api/predictive/predictions - Fetch predictions.

    Called by BriefingEngine.fetchPredictiveInsights().
    Returns a JSON array of prediction objects.

    Query params:
        limit: Max results (default 10)
        min_confidence: Minimum confidence threshold (default 0.0)
    """
    limit = int(request.args.get("limit", 10))
    min_confidence = float(request.args.get("min_confidence", 0.0))

    try:
        from web.services.predictive_engine import get_predictive_engine
        engine = get_predictive_engine()
        predictions = engine.get_predictions(limit=limit,
                                             min_confidence=min_confidence)
        return jsonify(predictions)
    except Exception as e:
        log.error(f"Failed to get predictions: {e}")
        return jsonify([])


@predictive_bp.route("/api/predictive/status", methods=["GET"])
def get_status():
    """GET /api/predictive/status - Engine status.

    Called by BriefingEngine.fetchPredictiveInsights().
    Returns shape: {loops: {total, completed, running}, patterns: {...}}
    """
    try:
        from web.services.predictive_engine import get_predictive_engine
        engine = get_predictive_engine()
        status = engine.get_status()
        return jsonify(status)
    except Exception as e:
        log.error(f"Failed to get predictive status: {e}")
        return jsonify({
            "loops": {"total": 0, "completed": 0, "running": 0},
            "patterns": {},
            "error": str(e),
        })


@predictive_bp.route("/api/predictive/record", methods=["POST"])
def record_action():
    """POST /api/predictive/record - Record a user action for learning.

    Called by Android after task execution or user actions.

    Body:
        action_type: Type of action (e.g., "task_generated", "review_submitted")
        project_id: Optional project context
        metadata: Optional additional context
    """
    data = request.get_json(silent=True) or {}
    action_type = data.get("action_type", "")
    project_id = data.get("project_id", "")
    metadata = data.get("metadata", {})

    if not action_type:
        return jsonify({"error": "action_type required"}), 400

    try:
        from web.services.predictive_engine import get_predictive_engine
        engine = get_predictive_engine()
        action = engine.record_user_action(action_type, project_id, metadata)
        return jsonify(action)
    except Exception as e:
        log.error(f"Failed to record action: {e}")
        return jsonify({"error": str(e)}), 500


@predictive_bp.route("/api/predictive/routines", methods=["GET"])
def get_routines():
    """GET /api/predictive/routines - List detected routines.

    Query params:
        status: Filter by status (detected, active, dismissed). Default: all.
    """
    status = request.args.get("status")

    try:
        from web.services.predictive_store import get_predictive_store
        store = get_predictive_store()
        routines = store.get_routines(status=status)
        return jsonify(routines)
    except Exception as e:
        log.error(f"Failed to get routines: {e}")
        return jsonify([])


@predictive_bp.route("/api/predictive/routines/<routine_id>/approve",
                     methods=["POST"])
def approve_routine(routine_id: str):
    """POST /api/predictive/routines/<id>/approve - Activate a routine.

    Converts a detected routine into an active automation.
    """
    try:
        from web.services.routine_detector import get_routine_detector
        detector = get_routine_detector()
        routine = detector.approve_routine(routine_id)
        if routine:
            return jsonify(routine)
        return jsonify({"error": "Routine not found"}), 404
    except Exception as e:
        log.error(f"Failed to approve routine: {e}")
        return jsonify({"error": str(e)}), 500


@predictive_bp.route("/api/predictive/predictions/<prediction_id>/resolve",
                     methods=["POST"])
def resolve_prediction(prediction_id: str):
    """POST /api/predictive/predictions/<id>/resolve - Feedback on prediction.

    Body:
        outcome: "accepted" or "rejected"
        feedback: Optional text feedback
    """
    data = request.get_json(silent=True) or {}
    outcome = data.get("outcome", "")
    feedback = data.get("feedback")

    if outcome not in ("accepted", "rejected", "expired"):
        return jsonify({"error": "outcome must be accepted, rejected, or expired"}), 400

    try:
        from web.services.predictive_engine import get_predictive_engine
        engine = get_predictive_engine()
        result = engine.resolve_prediction(prediction_id, outcome, feedback)
        if result:
            return jsonify(result)
        return jsonify({"error": "Prediction not found"}), 404
    except Exception as e:
        log.error(f"Failed to resolve prediction: {e}")
        return jsonify({"error": str(e)}), 500


@predictive_bp.route("/api/predictive/analyze", methods=["POST"])
def trigger_analysis():
    """POST /api/predictive/analyze - Trigger pattern analysis and prediction.

    Runs temporal pattern analysis, routine detection, and generates
    new predictions. Can be called on-demand or by scheduled jobs.
    """
    try:
        from web.services.predictive_engine import get_predictive_engine
        engine = get_predictive_engine()

        patterns = engine.analyze_temporal_patterns()
        routines = engine.detect_routines()
        predictions = engine.predict_next_actions()

        return jsonify({
            "patterns_found": len(patterns),
            "new_routines": len(routines),
            "predictions_generated": len(predictions),
        })
    except Exception as e:
        log.error(f"Failed to run analysis: {e}")
        return jsonify({"error": str(e)}), 500
