"""
Collaboration Service - Shared sessions and real-time collaboration

Enables users to:
- Share AI sessions with other users
- Collaborate on conversations in real-time
- Manage session permissions and invitations
"""

import json
import threading
import secrets
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum

# File paths
SHADOWAI_DIR = Path.home() / ".shadowai"
SHARED_SESSIONS_FILE = SHADOWAI_DIR / "shared_sessions.json"
INVITATIONS_FILE = SHADOWAI_DIR / "session_invitations.json"

# Thread lock for file I/O
_file_lock = threading.Lock()


class SessionRole(str, Enum):
    """User role in a shared session."""
    OWNER = "owner"  # Full control, can delete session
    EDITOR = "editor"  # Can send messages
    VIEWER = "viewer"  # Read-only access


class InvitationStatus(str, Enum):
    """Status of a session invitation."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    EXPIRED = "expired"


@dataclass
class SessionParticipant:
    """A participant in a shared session."""
    user_id: str
    role: SessionRole
    joined_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_active: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "role": self.role.value,
            "joined_at": self.joined_at,
            "last_active": self.last_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionParticipant":
        return cls(
            user_id=data["user_id"],
            role=SessionRole(data["role"]),
            joined_at=data.get("joined_at", datetime.utcnow().isoformat()),
            last_active=data.get("last_active"),
        )


@dataclass
class SharedSession:
    """A session shared between multiple users."""
    id: str  # Original session ID
    owner_id: str  # User who created/owns the share
    title: str
    description: Optional[str] = None
    participants: List[SessionParticipant] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    is_active: bool = True
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "title": self.title,
            "description": self.description,
            "participants": [p.to_dict() for p in self.participants],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_active": self.is_active,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SharedSession":
        participants = [SessionParticipant.from_dict(p) for p in data.get("participants", [])]
        return cls(
            id=data["id"],
            owner_id=data["owner_id"],
            title=data["title"],
            description=data.get("description"),
            participants=participants,
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
            is_active=data.get("is_active", True),
            settings=data.get("settings", {}),
        )

    def get_participant(self, user_id: str) -> Optional[SessionParticipant]:
        """Get participant by user ID."""
        for p in self.participants:
            if p.user_id == user_id:
                return p
        return None

    def add_participant(self, user_id: str, role: SessionRole = SessionRole.VIEWER):
        """Add a participant to the session."""
        # Remove existing if present
        self.participants = [p for p in self.participants if p.user_id != user_id]
        self.participants.append(SessionParticipant(user_id=user_id, role=role))
        self.updated_at = datetime.utcnow().isoformat()

    def remove_participant(self, user_id: str):
        """Remove a participant from the session."""
        self.participants = [p for p in self.participants if p.user_id != user_id]
        self.updated_at = datetime.utcnow().isoformat()

    def update_participant_role(self, user_id: str, role: SessionRole):
        """Update a participant's role."""
        participant = self.get_participant(user_id)
        if participant:
            participant.role = role
            self.updated_at = datetime.utcnow().isoformat()

    def can_user_edit(self, user_id: str) -> bool:
        """Check if user can edit (send messages) in this session."""
        if user_id == self.owner_id:
            return True
        participant = self.get_participant(user_id)
        return participant and participant.role in [SessionRole.OWNER, SessionRole.EDITOR]

    def can_user_view(self, user_id: str) -> bool:
        """Check if user can view this session."""
        if user_id == self.owner_id:
            return True
        return self.get_participant(user_id) is not None


@dataclass
class SessionInvitation:
    """Invitation to join a shared session."""
    id: str
    session_id: str
    inviter_id: str
    invitee_id: Optional[str] = None  # Specific user
    invitee_email: Optional[str] = None  # Or email for unregistered users
    role: SessionRole = SessionRole.VIEWER
    status: InvitationStatus = InvitationStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: Optional[str] = None
    message: Optional[str] = None  # Personal message from inviter

    @staticmethod
    def generate_id() -> str:
        return f"inv_{secrets.token_urlsafe(16)}"

    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        return datetime.fromisoformat(self.expires_at) < datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "inviter_id": self.inviter_id,
            "invitee_id": self.invitee_id,
            "invitee_email": self.invitee_email,
            "role": self.role.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionInvitation":
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            inviter_id=data["inviter_id"],
            invitee_id=data.get("invitee_id"),
            invitee_email=data.get("invitee_email"),
            role=SessionRole(data.get("role", "viewer")),
            status=InvitationStatus(data.get("status", "pending")),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            expires_at=data.get("expires_at"),
            message=data.get("message"),
        )

    @classmethod
    def create(cls, session_id: str, inviter_id: str, invitee_id: Optional[str] = None,
               invitee_email: Optional[str] = None, role: SessionRole = SessionRole.VIEWER,
               message: Optional[str] = None, expires_hours: int = 72) -> "SessionInvitation":
        expires_at = (datetime.utcnow() + timedelta(hours=expires_hours)).isoformat()
        return cls(
            id=cls.generate_id(),
            session_id=session_id,
            inviter_id=inviter_id,
            invitee_id=invitee_id,
            invitee_email=invitee_email,
            role=role,
            message=message,
            expires_at=expires_at,
        )


def _ensure_dir():
    """Ensure the shadow data directory exists."""
    SHADOWAI_DIR.mkdir(parents=True, exist_ok=True)


def _read_json_file(path: Path) -> Dict[str, Any]:
    """Read JSON file with thread safety."""
    with _file_lock:
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {path}: {e}")
    return {}


def _write_json_file(path: Path, data: Dict[str, Any]):
    """Write JSON file with thread safety."""
    with _file_lock:
        try:
            _ensure_dir()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Error writing {path}: {e}")
            raise


# =============================================================================
# Shared Session Management
# =============================================================================

def get_all_shared_sessions() -> List[SharedSession]:
    """Get all shared sessions."""
    data = _read_json_file(SHARED_SESSIONS_FILE)
    sessions_data = data.get("sessions", [])
    return [SharedSession.from_dict(s) for s in sessions_data]


def get_shared_session(session_id: str) -> Optional[SharedSession]:
    """Get a shared session by ID."""
    sessions = get_all_shared_sessions()
    for session in sessions:
        if session.id == session_id:
            return session
    return None


def get_user_shared_sessions(user_id: str) -> List[SharedSession]:
    """Get all shared sessions that a user is part of."""
    sessions = get_all_shared_sessions()
    result = []
    for session in sessions:
        if session.owner_id == user_id or session.get_participant(user_id):
            result.append(session)
    return result


def create_shared_session(session_id: str, owner_id: str, title: str,
                          description: Optional[str] = None,
                          settings: Optional[Dict[str, Any]] = None) -> SharedSession:
    """
    Create a new shared session.

    Args:
        session_id: The original session ID to share
        owner_id: User who owns the share
        title: Display title for the shared session
        description: Optional description
        settings: Optional settings dict

    Returns:
        The created SharedSession
    """
    shared_session = SharedSession(
        id=session_id,
        owner_id=owner_id,
        title=title,
        description=description,
        settings=settings or {},
    )

    # Add owner as participant
    shared_session.add_participant(owner_id, SessionRole.OWNER)

    # Save
    data = _read_json_file(SHARED_SESSIONS_FILE)
    sessions_data = data.get("sessions", [])

    # Remove existing if present
    sessions_data = [s for s in sessions_data if s.get("id") != session_id]
    sessions_data.append(shared_session.to_dict())

    data["sessions"] = sessions_data
    _write_json_file(SHARED_SESSIONS_FILE, data)

    return shared_session


def update_shared_session(shared_session: SharedSession):
    """Update a shared session in storage."""
    data = _read_json_file(SHARED_SESSIONS_FILE)
    sessions_data = data.get("sessions", [])

    sessions_data = [s for s in sessions_data if s.get("id") != shared_session.id]
    sessions_data.append(shared_session.to_dict())

    data["sessions"] = sessions_data
    _write_json_file(SHARED_SESSIONS_FILE, data)


def delete_shared_session(session_id: str, user_id: str) -> bool:
    """
    Delete a shared session.

    Only the owner can delete a shared session.
    """
    session = get_shared_session(session_id)
    if not session or session.owner_id != user_id:
        return False

    data = _read_json_file(SHARED_SESSIONS_FILE)
    sessions_data = data.get("sessions", [])
    sessions_data = [s for s in sessions_data if s.get("id") != session_id]
    data["sessions"] = sessions_data
    _write_json_file(SHARED_SESSIONS_FILE, data)

    # Also delete all invitations for this session
    _cleanup_session_invitations(session_id)

    return True


def add_session_participant(session_id: str, user_id: str, role: SessionRole,
                            added_by: str) -> bool:
    """
    Add a participant to a shared session.

    Returns True if successful.
    """
    session = get_shared_session(session_id)
    if not session:
        return False

    # Check if adder has permission
    if not session.can_user_edit(added_by):
        return False

    session.add_participant(user_id, role)
    update_shared_session(session)

    return True


def remove_session_participant(session_id: str, user_id: str, removed_by: str) -> bool:
    """
    Remove a participant from a shared session.

    Returns True if successful.
    """
    session = get_shared_session(session_id)
    if not session:
        return False

    # Owner can remove anyone, users can remove themselves
    if removed_by != session.owner_id and removed_by != user_id:
        return False

    # Can't remove the owner
    if user_id == session.owner_id:
        return False

    session.remove_participant(user_id)
    update_shared_session(session)

    return True


def update_participant_role(session_id: str, user_id: str, new_role: SessionRole,
                            updated_by: str) -> bool:
    """
    Update a participant's role in a shared session.

    Only owner can change roles.
    """
    session = get_shared_session(session_id)
    if not session or session.owner_id != updated_by:
        return False

    session.update_participant_role(user_id, new_role)
    update_shared_session(session)

    return True


# =============================================================================
# Invitation Management
# =============================================================================

def get_all_invitations() -> List[SessionInvitation]:
    """Get all invitations."""
    data = _read_json_file(INVITATIONS_FILE)
    invites_data = data.get("invitations", [])
    return [SessionInvitation.from_dict(i) for i in invites_data]


def get_invitation(invitation_id: str) -> Optional[SessionInvitation]:
    """Get an invitation by ID."""
    invitations = get_all_invitations()
    for inv in invitations:
        if inv.id == invitation_id:
            return inv
    return None


def get_user_pending_invitations(user_id: str) -> List[SessionInvitation]:
    """Get pending invitations for a user."""
    invitations = get_all_invitations()
    return [
        inv for inv in invitations
        if inv.invitee_id == user_id and
           inv.status == InvitationStatus.PENDING and
           not inv.is_expired()
    ]


def get_session_invitations(session_id: str) -> List[SessionInvitation]:
    """Get all invitations for a session."""
    invitations = get_all_invitations()
    return [inv for inv in invitations if inv.session_id == session_id]


def create_invitation(session_id: str, inviter_id: str,
                      invitee_id: Optional[str] = None,
                      invitee_email: Optional[str] = None,
                      role: SessionRole = SessionRole.VIEWER,
                      message: Optional[str] = None) -> Optional[SessionInvitation]:
    """
    Create an invitation to join a shared session.

    Args:
        session_id: Session to invite to
        inviter_id: User creating the invitation
        invitee_id: User ID to invite (if known)
        invitee_email: Email to invite (for unregistered users)
        role: Role to grant upon acceptance
        message: Personal message from inviter

    Returns:
        Created invitation or None if failed
    """
    # Verify session exists and inviter has permission
    session = get_shared_session(session_id)
    if not session or not session.can_user_edit(inviter_id):
        return None

    # Need at least one identifier
    if not invitee_id and not invitee_email:
        return None

    # Check if already invited
    existing_invites = get_session_invitations(session_id)
    for inv in existing_invites:
        if inv.status == InvitationStatus.PENDING:
            if (invitee_id and inv.invitee_id == invitee_id) or \
               (invitee_email and inv.invitee_email == invitee_email):
                return None  # Already invited

    invitation = SessionInvitation.create(
        session_id=session_id,
        inviter_id=inviter_id,
        invitee_id=invitee_id,
        invitee_email=invitee_email,
        role=role,
        message=message,
    )

    # Save
    data = _read_json_file(INVITATIONS_FILE)
    invites_data = data.get("invitations", [])
    invites_data.append(invitation.to_dict())
    data["invitations"] = invites_data
    _write_json_file(INVITATIONS_FILE, data)

    return invitation


def accept_invitation(invitation_id: str, user_id: str) -> bool:
    """
    Accept an invitation and join the shared session.

    Returns True if successful.
    """
    invitation = get_invitation(invitation_id)
    if not invitation:
        return False

    # Verify this is for the right user
    if invitation.invitee_id and invitation.invitee_id != user_id:
        return False

    # Check status and expiration
    if invitation.status != InvitationStatus.PENDING or invitation.is_expired():
        return False

    # Add user to session
    success = add_session_participant(
        invitation.session_id,
        user_id,
        invitation.role,
        invitation.inviter_id,
    )

    if success:
        # Update invitation status
        invitation.status = InvitationStatus.ACCEPTED
        _update_invitation(invitation)

    return success


def decline_invitation(invitation_id: str, user_id: str) -> bool:
    """
    Decline an invitation.

    Returns True if successful.
    """
    invitation = get_invitation(invitation_id)
    if not invitation:
        return False

    # Verify this is for the right user
    if invitation.invitee_id and invitation.invitee_id != user_id:
        return False

    if invitation.status != InvitationStatus.PENDING:
        return False

    invitation.status = InvitationStatus.DECLINED
    _update_invitation(invitation)

    return True


def revoke_invitation(invitation_id: str, user_id: str) -> bool:
    """
    Revoke (cancel) an invitation.

    Only inviter or session owner can revoke.
    """
    invitation = get_invitation(invitation_id)
    if not invitation:
        return False

    session = get_shared_session(invitation.session_id)
    if not session:
        return False

    # Check permission
    if user_id != invitation.inviter_id and user_id != session.owner_id:
        return False

    # Remove invitation
    data = _read_json_file(INVITATIONS_FILE)
    invites_data = data.get("invitations", [])
    invites_data = [i for i in invites_data if i.get("id") != invitation_id]
    data["invitations"] = invites_data
    _write_json_file(INVITATIONS_FILE, data)

    return True


def _update_invitation(invitation: SessionInvitation):
    """Update an invitation in storage."""
    data = _read_json_file(INVITATIONS_FILE)
    invites_data = data.get("invitations", [])

    invites_data = [i for i in invites_data if i.get("id") != invitation.id]
    invites_data.append(invitation.to_dict())

    data["invitations"] = invites_data
    _write_json_file(INVITATIONS_FILE, data)


def _cleanup_session_invitations(session_id: str):
    """Remove all invitations for a session."""
    data = _read_json_file(INVITATIONS_FILE)
    invites_data = data.get("invitations", [])
    invites_data = [i for i in invites_data if i.get("session_id") != session_id]
    data["invitations"] = invites_data
    _write_json_file(INVITATIONS_FILE, data)


def cleanup_expired_invitations() -> int:
    """Remove all expired invitations. Returns count removed."""
    data = _read_json_file(INVITATIONS_FILE)
    invites_data = data.get("invitations", [])
    original_count = len(invites_data)

    valid_invites = []
    for i in invites_data:
        inv = SessionInvitation.from_dict(i)
        if not inv.is_expired():
            valid_invites.append(i)

    data["invitations"] = valid_invites
    _write_json_file(INVITATIONS_FILE, data)

    return original_count - len(valid_invites)


# =============================================================================
# Permission Helpers
# =============================================================================

def can_user_access_session(session_id: str, user_id: str) -> bool:
    """Check if user can access (view) a shared session."""
    session = get_shared_session(session_id)
    if not session:
        return False
    return session.can_user_view(user_id)


def can_user_edit_session(session_id: str, user_id: str) -> bool:
    """Check if user can edit (send messages) in a shared session."""
    session = get_shared_session(session_id)
    if not session:
        return False
    return session.can_user_edit(user_id)


def get_user_role(session_id: str, user_id: str) -> Optional[SessionRole]:
    """Get user's role in a shared session."""
    session = get_shared_session(session_id)
    if not session:
        return None

    if user_id == session.owner_id:
        return SessionRole.OWNER

    participant = session.get_participant(user_id)
    return participant.role if participant else None


# =============================================================================
# Statistics
# =============================================================================

def get_collaboration_stats() -> Dict[str, Any]:
    """Get collaboration statistics."""
    sessions = get_all_shared_sessions()
    invitations = get_all_invitations()

    active_sessions = [s for s in sessions if s.is_active]
    pending_invites = [i for i in invitations if i.status == InvitationStatus.PENDING]

    total_participants = sum(len(s.participants) for s in sessions)

    return {
        "total_shared_sessions": len(sessions),
        "active_shared_sessions": len(active_sessions),
        "total_participants": total_participants,
        "pending_invitations": len(pending_invites),
    }
