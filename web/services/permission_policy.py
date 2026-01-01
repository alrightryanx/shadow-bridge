"""
Permission Policy System - AGI-Readiness Trust & Safety

Defines permission boundaries for AI agents:
1. Action-based permissions (what can they do)
2. Resource-based permissions (what can they access)
3. Trust levels (how much autonomy do they have)
4. Context-aware policies (permissions vary by situation)
5. Escalation rules (when to require human approval)

This ensures AI agents operate within defined safety boundaries.
"""
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import Lock
import json
from pathlib import Path

from .agent_protocol import AgentId, AgentDescriptor, CapabilityCategory
from .human_override import DecisionType

logger = logging.getLogger(__name__)

# Persistence
POLICY_FILE = Path.home() / ".shadowai" / "permission_policies.json"


class TrustLevel(Enum):
    """Trust levels for agents."""
    UNTRUSTED = 0       # New/unknown agent, requires approval for everything
    LIMITED = 1         # Limited trust, requires approval for sensitive ops
    STANDARD = 2        # Standard trust, can perform routine operations
    ELEVATED = 3        # Elevated trust, can perform most operations
    TRUSTED = 4         # Fully trusted, minimal restrictions
    SYSTEM = 5          # System-level trust (internal processes only)


class ActionCategory(Enum):
    """Categories of actions for permission checks."""
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    DELETE_FILE = "delete_file"
    EXECUTE_CODE = "execute_code"
    NETWORK_ACCESS = "network_access"
    SPAWN_AGENT = "spawn_agent"
    MODIFY_CONFIG = "modify_config"
    ACCESS_SECRETS = "access_secrets"
    EXTERNAL_API = "external_api"
    DATABASE_WRITE = "database_write"
    SEND_MESSAGE = "send_message"
    ESCALATE = "escalate"


class PermissionDecision(Enum):
    """Result of permission check."""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    RATE_LIMIT = "rate_limit"


@dataclass
class ResourcePattern:
    """Pattern for matching resources."""
    pattern: str            # Glob-style pattern (e.g., "/home/user/*", "*.py")
    resource_type: str      # file, api, database, etc.
    sensitivity: str = "normal"  # low, normal, high, critical

    def matches(self, resource: str) -> bool:
        """Check if resource matches this pattern."""
        import fnmatch
        return fnmatch.fnmatch(resource.lower(), self.pattern.lower())

    def to_dict(self) -> Dict:
        return {
            'pattern': self.pattern,
            'resource_type': self.resource_type,
            'sensitivity': self.sensitivity
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ResourcePattern':
        return cls(
            pattern=data['pattern'],
            resource_type=data['resource_type'],
            sensitivity=data.get('sensitivity', 'normal')
        )


@dataclass
class PermissionRule:
    """A single permission rule."""
    id: str
    name: str
    description: str

    # Matching criteria
    trust_levels: Set[TrustLevel]
    action_categories: Set[ActionCategory]
    resource_patterns: List[ResourcePattern]
    agent_types: Optional[Set[str]] = None  # None = all types

    # Decision
    decision: PermissionDecision = PermissionDecision.ALLOW

    # Conditions
    require_reason: bool = False
    max_operations_per_hour: Optional[int] = None
    time_restrictions: Optional[Dict[str, Any]] = None  # e.g., {"business_hours_only": True}

    # Escalation
    escalation_type: Optional[DecisionType] = None

    # Metadata
    enabled: bool = True
    priority: int = 50  # Higher = evaluated first
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'trust_levels': [tl.value for tl in self.trust_levels],
            'action_categories': [ac.value for ac in self.action_categories],
            'resource_patterns': [rp.to_dict() for rp in self.resource_patterns],
            'agent_types': list(self.agent_types) if self.agent_types else None,
            'decision': self.decision.value,
            'require_reason': self.require_reason,
            'max_operations_per_hour': self.max_operations_per_hour,
            'time_restrictions': self.time_restrictions,
            'escalation_type': self.escalation_type.value if self.escalation_type else None,
            'enabled': self.enabled,
            'priority': self.priority,
            'created_at': self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PermissionRule':
        return cls(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            trust_levels={TrustLevel(tl) for tl in data['trust_levels']},
            action_categories={ActionCategory(ac) for ac in data['action_categories']},
            resource_patterns=[ResourcePattern.from_dict(rp) for rp in data['resource_patterns']],
            agent_types=set(data['agent_types']) if data.get('agent_types') else None,
            decision=PermissionDecision(data['decision']),
            require_reason=data.get('require_reason', False),
            max_operations_per_hour=data.get('max_operations_per_hour'),
            time_restrictions=data.get('time_restrictions'),
            escalation_type=DecisionType(data['escalation_type']) if data.get('escalation_type') else None,
            enabled=data.get('enabled', True),
            priority=data.get('priority', 50),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now()
        )


@dataclass
class PermissionCheckResult:
    """Result of a permission check."""
    decision: PermissionDecision
    matched_rule: Optional[PermissionRule] = None
    reason: str = ""
    escalation_type: Optional[DecisionType] = None
    rate_limit_remaining: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            'decision': self.decision.value,
            'matched_rule': self.matched_rule.id if self.matched_rule else None,
            'reason': self.reason,
            'escalation_type': self.escalation_type.value if self.escalation_type else None,
            'rate_limit_remaining': self.rate_limit_remaining
        }


class PermissionPolicy:
    """
    Central permission policy system for AI agents.

    Provides:
    - Rule-based permission checking
    - Trust level management
    - Resource pattern matching
    - Rate limiting integration
    - Escalation to human override
    """

    def __init__(self):
        self._lock = Lock()

        # Permission rules
        self._rules: Dict[str, PermissionRule] = {}

        # Agent trust levels
        self._trust_levels: Dict[str, TrustLevel] = {}

        # Operation counters for rate limiting
        self._operation_counts: Dict[str, List[datetime]] = {}

        # Audit log
        self._check_history: List[Dict] = []
        self._max_history = 1000

        # Metrics
        self._metrics = {
            'checks_performed': 0,
            'decisions_allow': 0,
            'decisions_deny': 0,
            'decisions_escalate': 0
        }

        self._load_policies()
        self._create_default_rules()

    def _load_policies(self) -> None:
        """Load policies from disk."""
        try:
            if POLICY_FILE.exists():
                with open(POLICY_FILE, 'r') as f:
                    data = json.load(f)

                for rule_data in data.get('rules', []):
                    rule = PermissionRule.from_dict(rule_data)
                    self._rules[rule.id] = rule

                for agent_id, level in data.get('trust_levels', {}).items():
                    self._trust_levels[agent_id] = TrustLevel(level)

                logger.info(f"Loaded {len(self._rules)} permission rules")
        except Exception as e:
            logger.warning(f"Failed to load policies: {e}")

    def _save_policies(self) -> None:
        """Save policies to disk."""
        try:
            POLICY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(POLICY_FILE, 'w') as f:
                json.dump({
                    'rules': [r.to_dict() for r in self._rules.values()],
                    'trust_levels': {k: v.value for k, v in self._trust_levels.items()},
                    'saved_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save policies: {e}")

    def _create_default_rules(self) -> None:
        """Create default security rules if none exist."""
        if self._rules:
            return

        # Rule 1: Block access to secrets for untrusted agents
        self.add_rule(PermissionRule(
            id="block-secrets-untrusted",
            name="Block Secrets Access (Untrusted)",
            description="Untrusted agents cannot access secrets",
            trust_levels={TrustLevel.UNTRUSTED, TrustLevel.LIMITED},
            action_categories={ActionCategory.ACCESS_SECRETS},
            resource_patterns=[ResourcePattern("*", "secret")],
            decision=PermissionDecision.DENY
        ))

        # Rule 2: Require approval for code execution by limited agents
        self.add_rule(PermissionRule(
            id="approve-code-exec",
            name="Require Approval for Code Execution",
            description="Limited agents need approval to execute code",
            trust_levels={TrustLevel.LIMITED},
            action_categories={ActionCategory.EXECUTE_CODE},
            resource_patterns=[ResourcePattern("*", "code")],
            decision=PermissionDecision.REQUIRE_APPROVAL,
            escalation_type=DecisionType.CODE_EXECUTION
        ))

        # Rule 3: Rate limit external API calls
        self.add_rule(PermissionRule(
            id="rate-limit-external-api",
            name="Rate Limit External APIs",
            description="Limit external API calls to 100/hour",
            trust_levels={TrustLevel.LIMITED, TrustLevel.STANDARD},
            action_categories={ActionCategory.EXTERNAL_API},
            resource_patterns=[ResourcePattern("*", "api")],
            decision=PermissionDecision.ALLOW,
            max_operations_per_hour=100
        ))

        # Rule 4: Require approval for sensitive file writes
        self.add_rule(PermissionRule(
            id="approve-sensitive-writes",
            name="Approve Sensitive File Writes",
            description="Require approval for writing to sensitive locations",
            trust_levels={TrustLevel.UNTRUSTED, TrustLevel.LIMITED, TrustLevel.STANDARD},
            action_categories={ActionCategory.WRITE_FILE, ActionCategory.DELETE_FILE},
            resource_patterns=[
                ResourcePattern("/etc/*", "file", "critical"),
                ResourcePattern("*.env", "file", "critical"),
                ResourcePattern("*credentials*", "file", "critical"),
                ResourcePattern("*password*", "file", "critical"),
                ResourcePattern("*secret*", "file", "critical")
            ],
            decision=PermissionDecision.REQUIRE_APPROVAL,
            escalation_type=DecisionType.FILE_ACCESS
        ))

        # Rule 5: Allow standard operations for trusted agents
        self.add_rule(PermissionRule(
            id="allow-trusted",
            name="Allow Trusted Agent Operations",
            description="Trusted agents can perform most operations",
            trust_levels={TrustLevel.TRUSTED, TrustLevel.SYSTEM},
            action_categories=set(ActionCategory),
            resource_patterns=[ResourcePattern("*", "*")],
            decision=PermissionDecision.ALLOW,
            priority=10  # Lower priority, checked last
        ))

        # Rule 6: Block agent spawning for most agents
        self.add_rule(PermissionRule(
            id="restrict-agent-spawn",
            name="Restrict Agent Spawning",
            description="Only elevated+ agents can spawn new agents",
            trust_levels={TrustLevel.UNTRUSTED, TrustLevel.LIMITED, TrustLevel.STANDARD},
            action_categories={ActionCategory.SPAWN_AGENT},
            resource_patterns=[ResourcePattern("*", "agent")],
            decision=PermissionDecision.REQUIRE_APPROVAL,
            escalation_type=DecisionType.AGENT_SPAWN
        ))

        self._save_policies()
        logger.info("Created default permission rules")

    def add_rule(self, rule: PermissionRule) -> None:
        """Add or update a permission rule."""
        with self._lock:
            self._rules[rule.id] = rule
            self._save_policies()

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a permission rule."""
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                self._save_policies()
                return True
            return False

    def get_rule(self, rule_id: str) -> Optional[PermissionRule]:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def list_rules(self) -> List[PermissionRule]:
        """List all rules sorted by priority."""
        return sorted(self._rules.values(), key=lambda r: -r.priority)

    def set_trust_level(self, agent_id: AgentId, level: TrustLevel) -> None:
        """Set the trust level for an agent."""
        with self._lock:
            self._trust_levels[str(agent_id)] = level
            self._save_policies()

    def get_trust_level(self, agent_id: AgentId) -> TrustLevel:
        """Get the trust level for an agent (default: UNTRUSTED)."""
        return self._trust_levels.get(str(agent_id), TrustLevel.UNTRUSTED)

    def check_permission(
        self,
        agent_id: AgentId,
        action: ActionCategory,
        resource: str,
        resource_type: str = "file",
        context: Optional[Dict[str, Any]] = None
    ) -> PermissionCheckResult:
        """
        Check if an agent has permission to perform an action.

        Args:
            agent_id: The agent requesting permission
            action: The action category
            resource: The resource being accessed
            resource_type: Type of resource (file, api, database, etc.)
            context: Optional additional context

        Returns:
            PermissionCheckResult with decision and details
        """
        with self._lock:
            self._metrics['checks_performed'] += 1

            trust_level = self.get_trust_level(agent_id)
            agent_key = str(agent_id)

            # Check rules in priority order
            for rule in sorted(self._rules.values(), key=lambda r: -r.priority):
                if not rule.enabled:
                    continue

                # Check trust level match
                if trust_level not in rule.trust_levels:
                    continue

                # Check action match
                if action not in rule.action_categories:
                    continue

                # Check resource pattern match
                resource_matched = False
                for pattern in rule.resource_patterns:
                    if pattern.resource_type in (resource_type, "*") and pattern.matches(resource):
                        resource_matched = True
                        break

                if not resource_matched:
                    continue

                # Rule matches - check rate limiting if applicable
                if rule.max_operations_per_hour:
                    remaining = self._check_rate_limit(
                        agent_key,
                        rule.id,
                        rule.max_operations_per_hour
                    )

                    if remaining <= 0:
                        result = PermissionCheckResult(
                            decision=PermissionDecision.RATE_LIMIT,
                            matched_rule=rule,
                            reason="Rate limit exceeded",
                            rate_limit_remaining=0
                        )
                        self._log_check(agent_id, action, resource, result)
                        return result

                # Apply rule decision
                result = PermissionCheckResult(
                    decision=rule.decision,
                    matched_rule=rule,
                    reason=rule.description,
                    escalation_type=rule.escalation_type,
                    rate_limit_remaining=self._get_rate_limit_remaining(
                        agent_key, rule.id, rule.max_operations_per_hour
                    ) if rule.max_operations_per_hour else None
                )

                # Update metrics
                if rule.decision == PermissionDecision.ALLOW:
                    self._metrics['decisions_allow'] += 1
                    if rule.max_operations_per_hour:
                        self._record_operation(agent_key, rule.id)
                elif rule.decision == PermissionDecision.DENY:
                    self._metrics['decisions_deny'] += 1
                elif rule.decision == PermissionDecision.REQUIRE_APPROVAL:
                    self._metrics['decisions_escalate'] += 1

                self._log_check(agent_id, action, resource, result)
                return result

            # No rule matched - default deny for untrusted, allow for others
            if trust_level == TrustLevel.UNTRUSTED:
                result = PermissionCheckResult(
                    decision=PermissionDecision.DENY,
                    reason="No matching rule, default deny for untrusted agents"
                )
                self._metrics['decisions_deny'] += 1
            else:
                result = PermissionCheckResult(
                    decision=PermissionDecision.ALLOW,
                    reason="No matching rule, default allow"
                )
                self._metrics['decisions_allow'] += 1

            self._log_check(agent_id, action, resource, result)
            return result

    def _check_rate_limit(self, agent_key: str, rule_id: str, max_per_hour: int) -> int:
        """Check rate limit and return remaining operations."""
        key = f"{agent_key}:{rule_id}"
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)

        # Get operations in last hour
        ops = self._operation_counts.get(key, [])
        ops = [ts for ts in ops if ts > hour_ago]
        self._operation_counts[key] = ops

        return max_per_hour - len(ops)

    def _get_rate_limit_remaining(self, agent_key: str, rule_id: str, max_per_hour: Optional[int]) -> Optional[int]:
        """Get remaining rate limit operations."""
        if not max_per_hour:
            return None
        return self._check_rate_limit(agent_key, rule_id, max_per_hour)

    def _record_operation(self, agent_key: str, rule_id: str) -> None:
        """Record an operation for rate limiting."""
        key = f"{agent_key}:{rule_id}"
        if key not in self._operation_counts:
            self._operation_counts[key] = []
        self._operation_counts[key].append(datetime.now())

    def _log_check(
        self,
        agent_id: AgentId,
        action: ActionCategory,
        resource: str,
        result: PermissionCheckResult
    ) -> None:
        """Log a permission check for audit."""
        self._check_history.append({
            'timestamp': datetime.now().isoformat(),
            'agent_id': str(agent_id),
            'action': action.value,
            'resource': resource,
            'decision': result.decision.value,
            'rule_id': result.matched_rule.id if result.matched_rule else None,
            'reason': result.reason
        })

        # Trim history
        if len(self._check_history) > self._max_history:
            self._check_history = self._check_history[-self._max_history:]

    def get_check_history(self, limit: int = 100) -> List[Dict]:
        """Get recent permission check history."""
        return self._check_history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get permission policy statistics."""
        return {
            'total_rules': len(self._rules),
            'enabled_rules': len([r for r in self._rules.values() if r.enabled]),
            'agents_with_trust': len(self._trust_levels),
            'trust_level_distribution': {
                level.name: len([a for a, l in self._trust_levels.items() if l == level])
                for level in TrustLevel
            },
            **self._metrics
        }


# Global instance
_permission_policy: Optional[PermissionPolicy] = None


def get_permission_policy() -> PermissionPolicy:
    """Get or create the global permission policy."""
    global _permission_policy

    if _permission_policy is None:
        _permission_policy = PermissionPolicy()

    return _permission_policy
