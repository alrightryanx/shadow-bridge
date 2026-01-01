"""
Safety Audit Integration - AGI-Readiness Trust & Safety

Provides comprehensive audit logging for safety-relevant AI operations:
1. Safety event logging (permissions, overrides, rate limits)
2. Real-time safety monitoring
3. Compliance reporting
4. Risk scoring
5. Integration with all safety components

This creates a complete audit trail for AI safety compliance.
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from collections import defaultdict
import json
from pathlib import Path

from .agent_protocol import AgentId, MessageType

logger = logging.getLogger(__name__)

# Persistence
AUDIT_FILE = Path.home() / ".shadowai" / "safety_audit.jsonl"
RISK_SCORES_FILE = Path.home() / ".shadowai" / "risk_scores.json"
MAX_AUDIT_ENTRIES = 10000


class SafetyEventType(Enum):
    """Types of safety events to audit."""
    # Permission events
    PERMISSION_CHECK = "permission_check"
    PERMISSION_DENIED = "permission_denied"
    PERMISSION_ESCALATED = "permission_escalated"

    # Override events
    PAUSE_ISSUED = "pause_issued"
    RESUME_ISSUED = "resume_issued"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    KILL_SWITCH_DEACTIVATED = "kill_switch_deactivated"

    # Decision events
    DECISION_REQUESTED = "decision_requested"
    DECISION_APPROVED = "decision_approved"
    DECISION_DENIED = "decision_denied"
    DECISION_EXPIRED = "decision_expired"

    # Rate limit events
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    BURST_LIMITED = "burst_limited"

    # Trust events
    TRUST_LEVEL_CHANGED = "trust_level_changed"
    TRUST_VIOLATION = "trust_violation"

    # Agent events
    AGENT_REGISTERED = "agent_registered"
    AGENT_UNREGISTERED = "agent_unregistered"
    AGENT_STATE_CHANGE = "agent_state_change"
    AGENT_ERROR = "agent_error"

    # High-risk operations
    SENSITIVE_OPERATION = "sensitive_operation"
    EXTERNAL_ACCESS = "external_access"
    CODE_EXECUTION = "code_execution"
    DATA_EXPORT = "data_export"


class RiskLevel(Enum):
    """Risk levels for events and agents."""
    MINIMAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SafetyEvent:
    """A safety-relevant event for audit."""
    id: str
    timestamp: datetime
    event_type: SafetyEventType
    risk_level: RiskLevel

    # Event context
    agent_id: Optional[str] = None
    action: Optional[str] = None
    resource: Optional[str] = None
    decision: Optional[str] = None

    # Details
    details: Dict[str, Any] = field(default_factory=dict)

    # Outcome
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'risk_level': self.risk_level.name,
            'agent_id': self.agent_id,
            'action': self.action,
            'resource': self.resource,
            'decision': self.decision,
            'details': self.details,
            'success': self.success,
            'error_message': self.error_message
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SafetyEvent':
        return cls(
            id=data['id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            event_type=SafetyEventType(data['event_type']),
            risk_level=RiskLevel[data['risk_level']],
            agent_id=data.get('agent_id'),
            action=data.get('action'),
            resource=data.get('resource'),
            decision=data.get('decision'),
            details=data.get('details', {}),
            success=data.get('success', True),
            error_message=data.get('error_message')
        )


@dataclass
class AgentRiskScore:
    """Risk score for an agent."""
    agent_id: str
    current_score: float  # 0.0 - 100.0
    events_24h: int
    violations_24h: int
    high_risk_actions_24h: int
    last_updated: datetime

    def to_dict(self) -> Dict:
        return {
            'agent_id': self.agent_id,
            'current_score': self.current_score,
            'events_24h': self.events_24h,
            'violations_24h': self.violations_24h,
            'high_risk_actions_24h': self.high_risk_actions_24h,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class SafetyReport:
    """Safety audit report."""
    generated_at: datetime
    period_start: datetime
    period_end: datetime

    # Summary
    total_events: int
    events_by_type: Dict[str, int]
    events_by_risk: Dict[str, int]

    # Risk metrics
    agents_with_violations: int
    high_risk_events: int
    average_risk_score: float

    # Top concerns
    top_risk_agents: List[Dict]
    recent_high_risk_events: List[Dict]

    def to_dict(self) -> Dict:
        return {
            'generated_at': self.generated_at.isoformat(),
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'total_events': self.total_events,
            'events_by_type': self.events_by_type,
            'events_by_risk': self.events_by_risk,
            'agents_with_violations': self.agents_with_violations,
            'high_risk_events': self.high_risk_events,
            'average_risk_score': self.average_risk_score,
            'top_risk_agents': self.top_risk_agents,
            'recent_high_risk_events': self.recent_high_risk_events
        }


class SafetyAudit:
    """
    Central safety audit service.

    Integrates with:
    - HumanOverrideSystem: Logs pauses, resumes, kill switches, decisions
    - PermissionPolicy: Logs permission checks, denials, escalations
    - RateLimiter: Logs rate limit hits and quota violations
    - AgentRegistry: Logs agent lifecycle and state changes

    Provides:
    - Comprehensive audit trail
    - Real-time risk monitoring
    - Compliance reporting
    - Agent risk scoring
    """

    def __init__(self):
        self._lock = Lock()

        # In-memory event buffer
        self._events: List[SafetyEvent] = []
        self._max_events = MAX_AUDIT_ENTRIES

        # Agent risk scores
        self._risk_scores: Dict[str, AgentRiskScore] = {}

        # Real-time counters
        self._counters = defaultdict(int)
        self._last_reset = datetime.now()

        # Metrics
        self._metrics = {
            'total_events_logged': 0,
            'high_risk_events': 0,
            'critical_events': 0,
            'last_critical_event': None
        }

        self._load_recent_events()
        self._load_risk_scores()

    def _load_recent_events(self) -> None:
        """Load recent events from disk."""
        try:
            if AUDIT_FILE.exists():
                events = []
                with open(AUDIT_FILE, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            event = SafetyEvent.from_dict(data)
                            events.append(event)
                        except:
                            continue

                # Keep only recent events
                cutoff = datetime.now() - timedelta(days=7)
                self._events = [e for e in events if e.timestamp > cutoff][-self._max_events:]

                logger.info(f"Loaded {len(self._events)} safety events")
        except Exception as e:
            logger.warning(f"Failed to load safety events: {e}")

    def _load_risk_scores(self) -> None:
        """Load agent risk scores from disk."""
        try:
            if RISK_SCORES_FILE.exists():
                with open(RISK_SCORES_FILE, 'r') as f:
                    data = json.load(f)

                for agent_id, score_data in data.get('scores', {}).items():
                    self._risk_scores[agent_id] = AgentRiskScore(
                        agent_id=agent_id,
                        current_score=score_data['current_score'],
                        events_24h=score_data['events_24h'],
                        violations_24h=score_data['violations_24h'],
                        high_risk_actions_24h=score_data['high_risk_actions_24h'],
                        last_updated=datetime.fromisoformat(score_data['last_updated'])
                    )
        except Exception as e:
            logger.warning(f"Failed to load risk scores: {e}")

    def _persist_event(self, event: SafetyEvent) -> None:
        """Persist a single event to disk."""
        try:
            AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(AUDIT_FILE, 'a') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to persist safety event: {e}")

    def _save_risk_scores(self) -> None:
        """Save risk scores to disk."""
        try:
            RISK_SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(RISK_SCORES_FILE, 'w') as f:
                json.dump({
                    'scores': {
                        k: v.to_dict() for k, v in self._risk_scores.items()
                    },
                    'saved_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save risk scores: {e}")

    def log_event(
        self,
        event_type: SafetyEventType,
        risk_level: RiskLevel = RiskLevel.LOW,
        agent_id: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        decision: Optional[str] = None,
        details: Optional[Dict] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> SafetyEvent:
        """
        Log a safety event.

        Args:
            event_type: Type of safety event
            risk_level: Risk level of the event
            agent_id: Agent involved (if any)
            action: Action being performed
            resource: Resource being accessed
            decision: Decision made (allow/deny/etc.)
            details: Additional event details
            success: Whether the operation succeeded
            error_message: Error message if failed

        Returns:
            The logged SafetyEvent
        """
        with self._lock:
            from uuid import uuid4

            event = SafetyEvent(
                id=str(uuid4())[:12],
                timestamp=datetime.now(),
                event_type=event_type,
                risk_level=risk_level,
                agent_id=agent_id,
                action=action,
                resource=resource,
                decision=decision,
                details=details or {},
                success=success,
                error_message=error_message
            )

            # Add to buffer
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]

            # Persist
            self._persist_event(event)

            # Update metrics
            self._metrics['total_events_logged'] += 1
            if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                self._metrics['high_risk_events'] += 1
            if risk_level == RiskLevel.CRITICAL:
                self._metrics['critical_events'] += 1
                self._metrics['last_critical_event'] = event.id

            # Update counters
            self._counters[event_type.value] += 1
            self._counters[f"risk_{risk_level.name.lower()}"] += 1

            # Update agent risk score
            if agent_id:
                self._update_risk_score(agent_id, event)

            # Log high-risk events
            if risk_level.value >= RiskLevel.HIGH.value:
                logger.warning(
                    f"HIGH-RISK EVENT: {event_type.value} by {agent_id} - {action}"
                )

            return event

    def _update_risk_score(self, agent_id: str, event: SafetyEvent) -> None:
        """Update risk score for an agent based on an event."""
        now = datetime.now()
        day_ago = now - timedelta(hours=24)

        # Get or create risk score
        if agent_id not in self._risk_scores:
            self._risk_scores[agent_id] = AgentRiskScore(
                agent_id=agent_id,
                current_score=0.0,
                events_24h=0,
                violations_24h=0,
                high_risk_actions_24h=0,
                last_updated=now
            )

        score = self._risk_scores[agent_id]

        # Count events in last 24h for this agent
        agent_events = [
            e for e in self._events
            if e.agent_id == agent_id and e.timestamp > day_ago
        ]

        score.events_24h = len(agent_events)
        score.violations_24h = len([
            e for e in agent_events
            if e.event_type in (
                SafetyEventType.PERMISSION_DENIED,
                SafetyEventType.TRUST_VIOLATION,
                SafetyEventType.RATE_LIMITED,
                SafetyEventType.QUOTA_EXCEEDED
            )
        ])
        score.high_risk_actions_24h = len([
            e for e in agent_events
            if e.risk_level.value >= RiskLevel.HIGH.value
        ])

        # Calculate risk score (0-100)
        base_score = 0.0

        # Points for violations
        base_score += score.violations_24h * 5

        # Points for high-risk actions
        base_score += score.high_risk_actions_24h * 10

        # Points for volume (too many events can be suspicious)
        if score.events_24h > 1000:
            base_score += (score.events_24h - 1000) / 100

        # Cap at 100
        score.current_score = min(100.0, base_score)
        score.last_updated = now

        self._save_risk_scores()

    # ============ Convenience methods for common events ============

    def log_permission_check(
        self,
        agent_id: str,
        action: str,
        resource: str,
        decision: str,
        rule_id: Optional[str] = None
    ) -> SafetyEvent:
        """Log a permission check."""
        event_type = SafetyEventType.PERMISSION_CHECK
        if decision == "deny":
            event_type = SafetyEventType.PERMISSION_DENIED
        elif decision == "require_approval":
            event_type = SafetyEventType.PERMISSION_ESCALATED

        risk = RiskLevel.LOW
        if decision == "deny":
            risk = RiskLevel.MEDIUM

        return self.log_event(
            event_type=event_type,
            risk_level=risk,
            agent_id=agent_id,
            action=action,
            resource=resource,
            decision=decision,
            details={'rule_id': rule_id} if rule_id else {}
        )

    def log_pause(
        self,
        scope: str,
        target: Optional[str],
        reason: str,
        paused_by: str
    ) -> SafetyEvent:
        """Log a pause event."""
        return self.log_event(
            event_type=SafetyEventType.PAUSE_ISSUED,
            risk_level=RiskLevel.MEDIUM,
            action="pause",
            details={
                'scope': scope,
                'target': target,
                'reason': reason,
                'paused_by': paused_by
            }
        )

    def log_kill_switch(self, reason: str, activated: bool) -> SafetyEvent:
        """Log a kill switch event."""
        return self.log_event(
            event_type=(
                SafetyEventType.KILL_SWITCH_ACTIVATED if activated
                else SafetyEventType.KILL_SWITCH_DEACTIVATED
            ),
            risk_level=RiskLevel.CRITICAL if activated else RiskLevel.HIGH,
            action="kill_switch",
            details={'reason': reason, 'activated': activated}
        )

    def log_decision(
        self,
        agent_id: str,
        decision_type: str,
        action: str,
        decision: str,
        decided_by: Optional[str] = None
    ) -> SafetyEvent:
        """Log a decision event."""
        event_map = {
            'approved': SafetyEventType.DECISION_APPROVED,
            'denied': SafetyEventType.DECISION_DENIED,
            'expired': SafetyEventType.DECISION_EXPIRED,
            'pending': SafetyEventType.DECISION_REQUESTED
        }

        return self.log_event(
            event_type=event_map.get(decision, SafetyEventType.DECISION_REQUESTED),
            risk_level=RiskLevel.MEDIUM,
            agent_id=agent_id,
            action=action,
            decision=decision,
            details={
                'decision_type': decision_type,
                'decided_by': decided_by
            }
        )

    def log_rate_limit(
        self,
        agent_id: str,
        action: str,
        limit_type: str,
        config_id: str
    ) -> SafetyEvent:
        """Log a rate limit event."""
        event_map = {
            'rate_limited': SafetyEventType.RATE_LIMITED,
            'quota_exceeded': SafetyEventType.QUOTA_EXCEEDED,
            'burst_limited': SafetyEventType.BURST_LIMITED
        }

        return self.log_event(
            event_type=event_map.get(limit_type, SafetyEventType.RATE_LIMITED),
            risk_level=RiskLevel.MEDIUM,
            agent_id=agent_id,
            action=action,
            details={'limit_type': limit_type, 'config_id': config_id}
        )

    def log_trust_change(
        self,
        agent_id: str,
        old_level: str,
        new_level: str,
        changed_by: str
    ) -> SafetyEvent:
        """Log a trust level change."""
        return self.log_event(
            event_type=SafetyEventType.TRUST_LEVEL_CHANGED,
            risk_level=RiskLevel.MEDIUM,
            agent_id=agent_id,
            action="trust_change",
            details={
                'old_level': old_level,
                'new_level': new_level,
                'changed_by': changed_by
            }
        )

    # ============ Query methods ============

    def get_events(
        self,
        event_types: Optional[List[SafetyEventType]] = None,
        risk_levels: Optional[List[RiskLevel]] = None,
        agent_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[SafetyEvent]:
        """Get events with optional filters."""
        events = self._events

        if event_types:
            events = [e for e in events if e.event_type in event_types]

        if risk_levels:
            events = [e for e in events if e.risk_level in risk_levels]

        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]

        if since:
            events = [e for e in events if e.timestamp > since]

        return events[-limit:]

    def get_risk_score(self, agent_id: str) -> Optional[AgentRiskScore]:
        """Get risk score for an agent."""
        return self._risk_scores.get(agent_id)

    def get_all_risk_scores(self) -> List[AgentRiskScore]:
        """Get all agent risk scores, sorted by score descending."""
        scores = list(self._risk_scores.values())
        scores.sort(key=lambda s: s.current_score, reverse=True)
        return scores

    def generate_report(
        self,
        hours: int = 24
    ) -> SafetyReport:
        """Generate a safety report for the specified period."""
        now = datetime.now()
        period_start = now - timedelta(hours=hours)

        # Get events in period
        events = [e for e in self._events if e.timestamp > period_start]

        # Count by type
        events_by_type = defaultdict(int)
        for e in events:
            events_by_type[e.event_type.value] += 1

        # Count by risk
        events_by_risk = defaultdict(int)
        for e in events:
            events_by_risk[e.risk_level.name] += 1

        # High risk events
        high_risk = [
            e for e in events
            if e.risk_level.value >= RiskLevel.HIGH.value
        ]

        # Agents with violations
        violation_types = {
            SafetyEventType.PERMISSION_DENIED,
            SafetyEventType.TRUST_VIOLATION,
            SafetyEventType.RATE_LIMITED,
            SafetyEventType.QUOTA_EXCEEDED
        }
        agents_with_violations = set(
            e.agent_id for e in events
            if e.event_type in violation_types and e.agent_id
        )

        # Calculate average risk score
        risk_scores = [
            s.current_score for s in self._risk_scores.values()
        ]
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0

        # Top risk agents
        top_agents = self.get_all_risk_scores()[:10]

        return SafetyReport(
            generated_at=now,
            period_start=period_start,
            period_end=now,
            total_events=len(events),
            events_by_type=dict(events_by_type),
            events_by_risk=dict(events_by_risk),
            agents_with_violations=len(agents_with_violations),
            high_risk_events=len(high_risk),
            average_risk_score=avg_risk,
            top_risk_agents=[a.to_dict() for a in top_agents],
            recent_high_risk_events=[
                e.to_dict() for e in sorted(
                    high_risk,
                    key=lambda x: x.timestamp,
                    reverse=True
                )[:20]
            ]
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get safety audit statistics."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(hours=24)

        events_last_hour = len([e for e in self._events if e.timestamp > hour_ago])
        events_last_day = len([e for e in self._events if e.timestamp > day_ago])

        return {
            'total_events_in_memory': len(self._events),
            'events_last_hour': events_last_hour,
            'events_last_24h': events_last_day,
            'agents_tracked': len(self._risk_scores),
            'counters': dict(self._counters),
            **self._metrics
        }


# Global instance
_safety_audit: Optional[SafetyAudit] = None


def get_safety_audit() -> SafetyAudit:
    """Get or create the global safety audit service."""
    global _safety_audit

    if _safety_audit is None:
        _safety_audit = SafetyAudit()

    return _safety_audit
