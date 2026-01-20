"""
Collaboration Routes - Shared sessions and real-time collaboration API

Provides REST API endpoints for:
- Shared session management
- Invitations
- Participant management
"""

from flask import Blueprint, request, jsonify, g
from typing import Optional

from ..services import collaboration_service
from ..services.collaboration_service import SessionRole
from .auth import login_required, login_optional

collab_bp = Blueprint("collaboration", __name__)


# =============================================================================
# Shared Sessions
# =============================================================================

@collab_bp.route("/sessions", methods=["GET"])
@login_required
def get_shared_sessions():
    """
    Get all shared sessions the current user is part of.

    Returns list of shared sessions where user is owner or participant.
    """
    sessions = collaboration_service.get_user_shared_sessions(g.current_user.id)
    return jsonify({
        "sessions": [s.to_dict() for s in sessions],
    })


@collab_bp.route("/sessions/<session_id>", methods=["GET"])
@login_required
def get_shared_session(session_id):
    """
    Get a specific shared session.

    Requires user to be owner or participant.
    """
    session = collaboration_service.get_shared_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    if not session.can_user_view(g.current_user.id):
        return jsonify({"error": "Access denied"}), 403

    return jsonify({
        "session": session.to_dict(),
    })


@collab_bp.route("/sessions", methods=["POST"])
@login_required
def create_shared_session():
    """
    Create a new shared session.

    Request body:
        session_id: The original session ID to share
        title: Display title for the shared session
        description: Optional description
        settings: Optional settings dict
    """
    data = request.get_json() or {}

    session_id = data.get("session_id")
    title = data.get("title")

    if not session_id or not title:
        return jsonify({"error": "session_id and title are required"}), 400

    # Check if session is already shared
    existing = collaboration_service.get_shared_session(session_id)
    if existing:
        return jsonify({"error": "Session is already shared"}), 409

    shared_session = collaboration_service.create_shared_session(
        session_id=session_id,
        owner_id=g.current_user.id,
        title=title,
        description=data.get("description"),
        settings=data.get("settings"),
    )

    return jsonify({
        "success": True,
        "session": shared_session.to_dict(),
    }), 201


@collab_bp.route("/sessions/<session_id>", methods=["PATCH"])
@login_required
def update_shared_session(session_id):
    """
    Update a shared session.

    Only owner can update.

    Request body (all optional):
        title: New title
        description: New description
        settings: New settings (merged)
        is_active: Active status
    """
    session = collaboration_service.get_shared_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    if session.owner_id != g.current_user.id:
        return jsonify({"error": "Only owner can update session"}), 403

    data = request.get_json() or {}

    if "title" in data:
        session.title = data["title"]
    if "description" in data:
        session.description = data["description"]
    if "settings" in data:
        session.settings.update(data["settings"])
    if "is_active" in data:
        session.is_active = data["is_active"]

    collaboration_service.update_shared_session(session)

    return jsonify({
        "success": True,
        "session": session.to_dict(),
    })


@collab_bp.route("/sessions/<session_id>", methods=["DELETE"])
@login_required
def delete_shared_session(session_id):
    """
    Delete (unshare) a shared session.

    Only owner can delete.
    """
    success = collaboration_service.delete_shared_session(session_id, g.current_user.id)

    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Session not found or access denied"}), 404


# =============================================================================
# Participants
# =============================================================================

@collab_bp.route("/sessions/<session_id>/participants", methods=["GET"])
@login_required
def get_participants(session_id):
    """Get all participants in a shared session."""
    session = collaboration_service.get_shared_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    if not session.can_user_view(g.current_user.id):
        return jsonify({"error": "Access denied"}), 403

    return jsonify({
        "participants": [p.to_dict() for p in session.participants],
        "owner_id": session.owner_id,
    })


@collab_bp.route("/sessions/<session_id>/participants/<user_id>", methods=["DELETE"])
@login_required
def remove_participant(session_id, user_id):
    """
    Remove a participant from a shared session.

    Owner can remove anyone. Users can remove themselves.
    """
    success = collaboration_service.remove_session_participant(
        session_id, user_id, g.current_user.id
    )

    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Failed to remove participant"}), 400


@collab_bp.route("/sessions/<session_id>/participants/<user_id>/role", methods=["PUT"])
@login_required
def update_participant_role(session_id, user_id):
    """
    Update a participant's role.

    Only owner can change roles.

    Request body:
        role: New role (owner, editor, viewer)
    """
    data = request.get_json() or {}
    role_str = data.get("role")

    if not role_str:
        return jsonify({"error": "role is required"}), 400

    try:
        role = SessionRole(role_str)
    except ValueError:
        return jsonify({"error": "Invalid role"}), 400

    success = collaboration_service.update_participant_role(
        session_id, user_id, role, g.current_user.id
    )

    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Failed to update role"}), 400


@collab_bp.route("/sessions/<session_id>/leave", methods=["POST"])
@login_required
def leave_session(session_id):
    """Leave a shared session (remove yourself)."""
    success = collaboration_service.remove_session_participant(
        session_id, g.current_user.id, g.current_user.id
    )

    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Cannot leave session (owner cannot leave)"}), 400


# =============================================================================
# Invitations
# =============================================================================

@collab_bp.route("/invitations", methods=["GET"])
@login_required
def get_my_invitations():
    """Get pending invitations for current user."""
    invitations = collaboration_service.get_user_pending_invitations(g.current_user.id)
    return jsonify({
        "invitations": [i.to_dict() for i in invitations],
    })


@collab_bp.route("/sessions/<session_id>/invitations", methods=["GET"])
@login_required
def get_session_invitations(session_id):
    """
    Get all invitations for a session.

    Only owner or editors can view.
    """
    session = collaboration_service.get_shared_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    if not session.can_user_edit(g.current_user.id):
        return jsonify({"error": "Access denied"}), 403

    invitations = collaboration_service.get_session_invitations(session_id)
    return jsonify({
        "invitations": [i.to_dict() for i in invitations],
    })


@collab_bp.route("/sessions/<session_id>/invitations", methods=["POST"])
@login_required
def create_invitation(session_id):
    """
    Create an invitation to join a shared session.

    Request body:
        invitee_id: User ID to invite (optional)
        invitee_email: Email to invite (optional, for unregistered)
        role: Role to grant (default: viewer)
        message: Personal message (optional)
    """
    data = request.get_json() or {}

    invitee_id = data.get("invitee_id")
    invitee_email = data.get("invitee_email")

    if not invitee_id and not invitee_email:
        return jsonify({"error": "invitee_id or invitee_email is required"}), 400

    role_str = data.get("role", "viewer")
    try:
        role = SessionRole(role_str)
    except ValueError:
        return jsonify({"error": "Invalid role"}), 400

    invitation = collaboration_service.create_invitation(
        session_id=session_id,
        inviter_id=g.current_user.id,
        invitee_id=invitee_id,
        invitee_email=invitee_email,
        role=role,
        message=data.get("message"),
    )

    if invitation:
        return jsonify({
            "success": True,
            "invitation": invitation.to_dict(),
        }), 201
    else:
        return jsonify({"error": "Failed to create invitation"}), 400


@collab_bp.route("/invitations/<invitation_id>/accept", methods=["POST"])
@login_required
def accept_invitation(invitation_id):
    """Accept an invitation and join the shared session."""
    success = collaboration_service.accept_invitation(invitation_id, g.current_user.id)

    if success:
        invitation = collaboration_service.get_invitation(invitation_id)
        return jsonify({
            "success": True,
            "session_id": invitation.session_id if invitation else None,
        })
    else:
        return jsonify({"error": "Failed to accept invitation"}), 400


@collab_bp.route("/invitations/<invitation_id>/decline", methods=["POST"])
@login_required
def decline_invitation(invitation_id):
    """Decline an invitation."""
    success = collaboration_service.decline_invitation(invitation_id, g.current_user.id)

    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Failed to decline invitation"}), 400


@collab_bp.route("/invitations/<invitation_id>", methods=["DELETE"])
@login_required
def revoke_invitation(invitation_id):
    """
    Revoke (cancel) an invitation.

    Only inviter or session owner can revoke.
    """
    success = collaboration_service.revoke_invitation(invitation_id, g.current_user.id)

    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Failed to revoke invitation"}), 400


# =============================================================================
# Permission Check
# =============================================================================

@collab_bp.route("/sessions/<session_id>/access", methods=["GET"])
@login_required
def check_access(session_id):
    """
    Check current user's access to a shared session.

    Returns role and permissions.
    """
    session = collaboration_service.get_shared_session(session_id)
    if not session:
        return jsonify({
            "has_access": False,
            "error": "Session not found or not shared",
        }), 404

    role = collaboration_service.get_user_role(session_id, g.current_user.id)

    if role:
        return jsonify({
            "has_access": True,
            "role": role.value,
            "can_edit": session.can_user_edit(g.current_user.id),
            "can_view": session.can_user_view(g.current_user.id),
            "is_owner": session.owner_id == g.current_user.id,
        })
    else:
        return jsonify({
            "has_access": False,
        })


# =============================================================================
# Stats
# =============================================================================

@collab_bp.route("/stats", methods=["GET"])
def get_collaboration_stats():
    """Get collaboration statistics (for dashboard)."""
    stats = collaboration_service.get_collaboration_stats()
    return jsonify(stats)
