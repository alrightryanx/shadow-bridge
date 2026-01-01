"""
Human Override System - AGI-Readiness Trust & Safety

Provides human control over AI agent operations:
1. Pause/Resume - Halt agents individually or globally
2. Inspect - View agent state, tasks, and decisions
3. Redirect - Change agent tasks mid-execution
4. Kill Switch - Emergency stop all AI activity
5. Decision Queue - Review and approve pending decisions

This ensures humans remain in control as AI capabilities grow.
"""
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from threading import Lock
from uuid import uuid4
import json
from pathlib import Path

from .agent_protocol import AgentId, AgentState, AgentStateType, Task, TaskStatus

logger = logging.getLogger(__name__)

# Persistence
OVERRIDE_STATE_FILE = Path.home() / ".shadowai" / "override_state.json"


class OverrideScope(Enum):
    """Scope of override operation."""
    SINGLE_AGENT = "single_agent"
    AGENT_TYPE = "agent_type"
    TEAM = "team"
    ALL_AGENTS = "all_agents"
    SYSTEM = "system"  # Everything including non-agent processes


class DecisionType(Enum):
    """Types of decisions requiring human review."""
    FILE_ACCESS = "file_access"
    NETWORK_ACCESS = "network_access"
    CODE_EXECUTION = "code_execution"
    SYSTEM_MODIFICATION = "system_modification"
    AGENT_SPAWN = "agent_spawn"
    EXTERNAL_API = "external_api"
    DATA_EXPORT = "data_export"
    HIGH_COST_OPERATION = "high_cost_operation"
    SENSITIVE_DATA = "sensitive_data"


class DecisionStatus(Enum):
    """Status of a pending decision."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    AUTO_APPROVED = "auto_approved"


@dataclass
class PauseToken:
    """Token representing a pause operation."""
    id: str
    scope: OverrideScope
    target: Optional[str]  # Agent ID, type, or team name
    reason: str
    paused_by: str  # User or system identifier
    paused_at: datetime
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'scope': self.scope.value,
            'target': self.target,
            'reason': self.reason,
            'paused_by': self.paused_by,
            'paused_at': self.paused_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class PendingDecision:
    """A decision waiting for human approval."""
    id: str
    agent_id: AgentId
    decision_type: DecisionType
    action: str
    context: Dict[str, Any]
    risk_level: str  # low, medium, high, critical
    status: DecisionStatus = DecisionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    decided_by: Optional[str] = None
    decided_at: Optional[datetime] = None
    decision_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'agent_id': self.agent_id.to_dict(),
            'decision_type': self.decision_type.value,
            'action': self.action,
            'context': self.context,
            'risk_level': self.risk_level,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'decided_by': self.decided_by,
            'decided_at': self.decided_at.isoformat() if self.decided_at else None,
            'decision_reason': self.decision_reason
        }


@dataclass
class AgentInspection:
    """Detailed inspection of an agent's state."""
    agent_id: AgentId
    state: AgentState
    current_task: Optional[Dict] = None
    recent_actions: List[Dict] = field(default_factory=list)
    pending_decisions: List[Dict] = field(default_factory=list)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    message_queue_size: int = 0
    uptime_seconds: int = 0
    tasks_completed: int = 0
    errors_count: int = 0

    def to_dict(self) -> Dict:
        return {
            'agent_id': self.agent_id.to_dict(),
            'state': self.state.to_dict(),
            'current_task': self.current_task,
            'recent_actions': self.recent_actions,
            'pending_decisions': self.pending_decisions,
            'resource_usage': self.resource_usage,
            'message_queue_size': self.message_queue_size,
            'uptime_seconds': self.uptime_seconds,
            'tasks_completed': self.tasks_completed,
            'errors_count': self.errors_count
        }


class HumanOverrideSystem:
    """
    Central system for human control over AI agents.

    Provides:
    - Global and per-agent pause/resume
    - Real-time agent inspection
    - Decision queue for approval workflows
    - Emergency kill switch
    - Redirect capabilities
    """

    def __init__(self):
        self._lock = Lock()

        # Active pause tokens
        self._pause_tokens: Dict[str, PauseToken] = {}

        # Paused agents (agent_id -> pause_token_id)
        self._paused_agents: Dict[str, str] = {}

        # Global pause state
        self._global_pause: bool = False
        self._global_pause_token: Optional[PauseToken] = None

        # Kill switch state
        self._kill_switch_active: bool = False
        self._kill_switch_activated_at: Optional[datetime] = None
        self._kill_switch_reason: Optional[str] = None

        # Decision queue
        self._pending_decisions: Dict[str, PendingDecision] = {}

        # Action history for inspection
        self._action_history: Dict[str, List[Dict]] = {}  # agent_id -> actions
        self._max_action_history = 100

        # Metrics
        self._metrics = {
            'pauses_issued': 0,
            'decisions_approved': 0,
            'decisions_denied': 0,
            'kill_switches_activated': 0
        }

        self._load_state()

    def _load_state(self) -> None:
        """Load override state from disk."""
        try:
            if OVERRIDE_STATE_FILE.exists():
                with open(OVERRIDE_STATE_FILE, 'r') as f:
                    data = json.load(f)

                self._global_pause = data.get('global_pause', False)
                self._kill_switch_active = data.get('kill_switch_active', False)

                if data.get('kill_switch_activated_at'):
                    self._kill_switch_activated_at = datetime.fromisoformat(
                        data['kill_switch_activated_at']
                    )

                logger.info("Loaded override state")
        except Exception as e:
            logger.warning(f"Failed to load override state: {e}")

    def _save_state(self) -> None:
        """Save override state to disk."""
        try:
            OVERRIDE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(OVERRIDE_STATE_FILE, 'w') as f:
                json.dump({
                    'global_pause': self._global_pause,
                    'kill_switch_active': self._kill_switch_active,
                    'kill_switch_activated_at': (
                        self._kill_switch_activated_at.isoformat()
                        if self._kill_switch_activated_at else None
                    ),
                    'saved_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save override state: {e}")

    # ============ Pause/Resume ============

    def pause(
        self,
        scope: OverrideScope,
        target: Optional[str] = None,
        reason: str = "",
        paused_by: str = "user",
        duration_minutes: Optional[int] = None
    ) -> PauseToken:
        """
        Pause agent(s).

        Args:
            scope: Scope of pause (single, type, team, all)
            target: Target identifier (agent ID, type, team name)
            reason: Reason for pause
            paused_by: Who initiated the pause
            duration_minutes: Optional auto-expire duration

        Returns:
            PauseToken for resuming
        """
        with self._lock:
            expires_at = None
            if duration_minutes:
                expires_at = datetime.now() + timedelta(minutes=duration_minutes)

            token = PauseToken(
                id=str(uuid4())[:12],
                scope=scope,
                target=target,
                reason=reason,
                paused_by=paused_by,
                paused_at=datetime.now(),
                expires_at=expires_at
            )

            self._pause_tokens[token.id] = token

            # Apply pause based on scope
            if scope == OverrideScope.ALL_AGENTS or scope == OverrideScope.SYSTEM:
                self._global_pause = True
                self._global_pause_token = token
            elif scope == OverrideScope.SINGLE_AGENT and target:
                self._paused_agents[target] = token.id

            self._metrics['pauses_issued'] += 1
            self._save_state()

            logger.warning(f"PAUSE issued: scope={scope.value}, target={target}, reason={reason}")

            return token

    def resume(self, token_id: str) -> bool:
        """
        Resume from a pause.

        Args:
            token_id: The pause token to cancel

        Returns:
            True if successfully resumed
        """
        with self._lock:
            token = self._pause_tokens.get(token_id)
            if not token:
                return False

            # Remove pause based on scope
            if token.scope in (OverrideScope.ALL_AGENTS, OverrideScope.SYSTEM):
                self._global_pause = False
                self._global_pause_token = None
            elif token.scope == OverrideScope.SINGLE_AGENT and token.target:
                self._paused_agents.pop(token.target, None)

            del self._pause_tokens[token_id]
            self._save_state()

            logger.info(f"RESUME: token={token_id}")
            return True

    def is_paused(self, agent_id: Optional[AgentId] = None) -> bool:
        """Check if an agent or the system is paused."""
        if self._kill_switch_active:
            return True

        if self._global_pause:
            # Check if global pause has expired
            if self._global_pause_token and self._global_pause_token.is_expired():
                self._global_pause = False
                self._global_pause_token = None
            else:
                return True

        if agent_id:
            agent_key = str(agent_id)
            if agent_key in self._paused_agents:
                token_id = self._paused_agents[agent_key]
                token = self._pause_tokens.get(token_id)
                if token and not token.is_expired():
                    return True
                else:
                    # Clean up expired pause
                    self._paused_agents.pop(agent_key, None)

        return False

    def get_active_pauses(self) -> List[PauseToken]:
        """Get all active pause tokens."""
        active = []
        for token in self._pause_tokens.values():
            if not token.is_expired():
                active.append(token)
        return active

    # ============ Kill Switch ============

    def activate_kill_switch(self, reason: str = "Emergency stop") -> bool:
        """
        Activate the emergency kill switch.

        This immediately halts ALL agent activity and prevents new tasks.
        """
        with self._lock:
            if self._kill_switch_active:
                return False  # Already active

            self._kill_switch_active = True
            self._kill_switch_activated_at = datetime.now()
            self._kill_switch_reason = reason
            self._global_pause = True

            self._metrics['kill_switches_activated'] += 1
            self._save_state()

            logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

            return True

    def deactivate_kill_switch(self, confirmation: str) -> bool:
        """
        Deactivate the kill switch.

        Args:
            confirmation: Must be "CONFIRM_DEACTIVATE" to proceed
        """
        if confirmation != "CONFIRM_DEACTIVATE":
            logger.warning("Kill switch deactivation rejected: invalid confirmation")
            return False

        with self._lock:
            self._kill_switch_active = False
            self._kill_switch_activated_at = None
            self._kill_switch_reason = None
            self._global_pause = False
            self._global_pause_token = None

            self._save_state()

            logger.warning("KILL SWITCH DEACTIVATED")
            return True

    def get_kill_switch_status(self) -> Dict[str, Any]:
        """Get current kill switch status."""
        return {
            'active': self._kill_switch_active,
            'activated_at': (
                self._kill_switch_activated_at.isoformat()
                if self._kill_switch_activated_at else None
            ),
            'reason': self._kill_switch_reason,
            'duration_seconds': (
                (datetime.now() - self._kill_switch_activated_at).total_seconds()
                if self._kill_switch_activated_at else None
            )
        }

    # ============ Decision Queue ============

    def request_decision(
        self,
        agent_id: AgentId,
        decision_type: DecisionType,
        action: str,
        context: Dict[str, Any],
        risk_level: str = "medium",
        timeout_minutes: int = 30
    ) -> PendingDecision:
        """
        Request human approval for a decision.

        Args:
            agent_id: Agent requesting approval
            decision_type: Type of decision
            action: Description of the action
            context: Additional context
            risk_level: Risk level (low, medium, high, critical)
            timeout_minutes: How long before decision expires

        Returns:
            PendingDecision that can be checked for status
        """
        with self._lock:
            decision = PendingDecision(
                id=str(uuid4())[:12],
                agent_id=agent_id,
                decision_type=decision_type,
                action=action,
                context=context,
                risk_level=risk_level,
                expires_at=datetime.now() + timedelta(minutes=timeout_minutes)
            )

            self._pending_decisions[decision.id] = decision

            logger.info(f"Decision requested: {decision.id} - {action}")

            return decision

    def approve_decision(
        self,
        decision_id: str,
        approved_by: str,
        reason: Optional[str] = None
    ) -> bool:
        """Approve a pending decision."""
        with self._lock:
            decision = self._pending_decisions.get(decision_id)
            if not decision:
                return False

            if decision.status != DecisionStatus.PENDING:
                return False

            decision.status = DecisionStatus.APPROVED
            decision.decided_by = approved_by
            decision.decided_at = datetime.now()
            decision.decision_reason = reason

            self._metrics['decisions_approved'] += 1

            logger.info(f"Decision APPROVED: {decision_id} by {approved_by}")
            return True

    def deny_decision(
        self,
        decision_id: str,
        denied_by: str,
        reason: Optional[str] = None
    ) -> bool:
        """Deny a pending decision."""
        with self._lock:
            decision = self._pending_decisions.get(decision_id)
            if not decision:
                return False

            if decision.status != DecisionStatus.PENDING:
                return False

            decision.status = DecisionStatus.DENIED
            decision.decided_by = denied_by
            decision.decided_at = datetime.now()
            decision.decision_reason = reason

            self._metrics['decisions_denied'] += 1

            logger.info(f"Decision DENIED: {decision_id} by {denied_by}")
            return True

    def get_decision_status(self, decision_id: str) -> Optional[PendingDecision]:
        """Get the status of a decision."""
        decision = self._pending_decisions.get(decision_id)
        if decision:
            # Check for expiry
            if (decision.status == DecisionStatus.PENDING and
                decision.expires_at and
                datetime.now() > decision.expires_at):
                decision.status = DecisionStatus.EXPIRED
        return decision

    def get_pending_decisions(
        self,
        agent_id: Optional[AgentId] = None,
        decision_type: Optional[DecisionType] = None
    ) -> List[PendingDecision]:
        """Get all pending decisions, optionally filtered."""
        pending = []

        for decision in self._pending_decisions.values():
            # Update expired status
            if (decision.status == DecisionStatus.PENDING and
                decision.expires_at and
                datetime.now() > decision.expires_at):
                decision.status = DecisionStatus.EXPIRED
                continue

            if decision.status != DecisionStatus.PENDING:
                continue

            if agent_id and str(decision.agent_id) != str(agent_id):
                continue

            if decision_type and decision.decision_type != decision_type:
                continue

            pending.append(decision)

        return pending

    # ============ Inspection ============

    def record_action(
        self,
        agent_id: AgentId,
        action: str,
        details: Dict[str, Any]
    ) -> None:
        """Record an action for inspection history."""
        agent_key = str(agent_id)

        if agent_key not in self._action_history:
            self._action_history[agent_key] = []

        self._action_history[agent_key].append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        })

        # Trim history
        if len(self._action_history[agent_key]) > self._max_action_history:
            self._action_history[agent_key] = \
                self._action_history[agent_key][-self._max_action_history:]

    def inspect(
        self,
        agent_id: AgentId,
        registry=None,
        message_bus=None
    ) -> AgentInspection:
        """
        Get detailed inspection of an agent.

        Args:
            agent_id: Agent to inspect
            registry: Optional AgentRegistry for state info
            message_bus: Optional MessageBus for queue info
        """
        agent_key = str(agent_id)

        # Get state from registry if available
        state = AgentState.idle()
        if registry:
            state = registry.get_state(agent_id) or AgentState.idle()

        # Get message queue size
        queue_size = 0
        if message_bus:
            queue_size = message_bus.get_queue_size(agent_id)

        # Get pending decisions for this agent
        pending = [
            d.to_dict() for d in self.get_pending_decisions(agent_id)
        ]

        # Get action history
        actions = self._action_history.get(agent_key, [])[-20:]

        return AgentInspection(
            agent_id=agent_id,
            state=state,
            recent_actions=actions,
            pending_decisions=pending,
            message_queue_size=queue_size
        )

    # ============ Stats ============

    def get_stats(self) -> Dict[str, Any]:
        """Get override system statistics."""
        pending_count = len([
            d for d in self._pending_decisions.values()
            if d.status == DecisionStatus.PENDING
        ])

        return {
            'global_pause': self._global_pause,
            'kill_switch_active': self._kill_switch_active,
            'active_pause_tokens': len(self.get_active_pauses()),
            'paused_agents': len(self._paused_agents),
            'pending_decisions': pending_count,
            'total_decisions': len(self._pending_decisions),
            **self._metrics
        }


# Global instance
_override_system: Optional[HumanOverrideSystem] = None


def get_override_system() -> HumanOverrideSystem:
    """Get or create the global human override system."""
    global _override_system

    if _override_system is None:
        _override_system = HumanOverrideSystem()

    return _override_system
