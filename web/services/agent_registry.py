"""
Agent Registry Service - AGI-Readiness Infrastructure

Manages agent registration, discovery, and health monitoring:
1. Register/unregister agents
2. Capability-based discovery
3. Load balancing across agents
4. Health monitoring and failover

This is the directory service for the multi-agent system.
"""
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict
from threading import Lock
import json
from pathlib import Path

from .agent_protocol import (
    AgentId, AgentDescriptor, AgentState, AgentStateType,
    Capability, CapabilityCategory, AgentMessage, MessageType,
    Task, TaskStatus, create_standard_agent
)

logger = logging.getLogger(__name__)

# Persistence
REGISTRY_FILE = Path.home() / ".shadowai" / "agent_registry.json"

# Health check settings
HEARTBEAT_TIMEOUT_SECONDS = 60
UNHEALTHY_THRESHOLD = 3  # Missed heartbeats before marked unhealthy


class AgentRegistry:
    """
    Central registry for all agents in the system.

    Provides:
    - Agent registration and lifecycle management
    - Capability-based agent discovery
    - Load balancing for task assignment
    - Health monitoring
    """

    def __init__(self):
        self._agents: Dict[str, AgentDescriptor] = {}
        self._states: Dict[str, AgentState] = {}
        self._heartbeats: Dict[str, datetime] = {}
        self._task_counts: Dict[str, int] = defaultdict(int)
        self._metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._lock = Lock()
        self._message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)

        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        try:
            if REGISTRY_FILE.exists():
                with open(REGISTRY_FILE, 'r', encoding="utf-8") as f:
                    data = json.load(f)

                for agent_data in data.get('agents', []):
                    descriptor = AgentDescriptor.from_dict(agent_data)
                    self._agents[str(descriptor.id)] = descriptor
                    self._states[str(descriptor.id)] = AgentState.idle()

                logger.info(f"Loaded {len(self._agents)} agents from registry")
        except Exception as e:
            logger.warning(f"Failed to load registry: {e}")

    def _save_registry(self) -> None:
        """Save registry to disk."""
        try:
            REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(REGISTRY_FILE, 'w', encoding="utf-8") as f:
                json.dump({
                    'agents': [a.to_dict() for a in self._agents.values()],
                    'saved_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def register(self, agent: AgentDescriptor) -> AgentId:
        """
        Register a new agent.

        Args:
            agent: Agent descriptor

        Returns:
            The agent's ID
        """
        with self._lock:
            agent_key = str(agent.id)

            if agent_key in self._agents:
                logger.warning(f"Agent {agent_key} already registered, updating")

            self._agents[agent_key] = agent
            self._states[agent_key] = AgentState.idle()
            self._heartbeats[agent_key] = datetime.now()
            self._task_counts[agent_key] = 0

            self._save_registry()
            logger.info(f"Registered agent: {agent.name} ({agent.agent_type})")

            return agent.id

    def unregister(self, agent_id: AgentId) -> bool:
        """
        Unregister an agent.

        Args:
            agent_id: ID of agent to remove

        Returns:
            True if agent was found and removed
        """
        with self._lock:
            agent_key = str(agent_id)

            if agent_key not in self._agents:
                return False

            del self._agents[agent_key]
            self._states.pop(agent_key, None)
            self._heartbeats.pop(agent_key, None)
            self._task_counts.pop(agent_key, None)
            self._metrics.pop(agent_key, None)

            self._save_registry()
            logger.info(f"Unregistered agent: {agent_key}")

            return True

    def get(self, agent_id: AgentId) -> Optional[AgentDescriptor]:
        """Get an agent by ID."""
        return self._agents.get(str(agent_id))

    def get_state(self, agent_id: AgentId) -> Optional[AgentState]:
        """Get the current state of an agent."""
        return self._states.get(str(agent_id))

    def update_state(self, agent_id: AgentId, state: AgentState) -> None:
        """Update an agent's state."""
        agent_key = str(agent_id)
        if agent_key in self._agents:
            self._states[agent_key] = state
            self._heartbeats[agent_key] = datetime.now()

    def list_all(self) -> List[AgentDescriptor]:
        """List all registered agents."""
        return list(self._agents.values())

    def discover(
        self,
        capability: CapabilityCategory,
        min_confidence: float = 0.5
    ) -> List[AgentDescriptor]:
        """
        Find agents with a specific capability.

        Args:
            capability: Required capability
            min_confidence: Minimum confidence level

        Returns:
            List of agents with the capability, sorted by confidence
        """
        results = []

        for agent in self._agents.values():
            confidence = agent.get_capability_confidence(capability)
            if confidence >= min_confidence:
                results.append((agent, confidence))

        # Sort by confidence (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return [agent for agent, _ in results]

    def discover_multi(
        self,
        capabilities: List[CapabilityCategory],
        require_all: bool = True
    ) -> List[AgentDescriptor]:
        """
        Find agents with multiple capabilities.

        Args:
            capabilities: Required capabilities
            require_all: If True, agent must have ALL capabilities

        Returns:
            List of matching agents
        """
        results = []

        for agent in self._agents.values():
            matching = sum(1 for cap in capabilities if agent.has_capability(cap))

            if require_all and matching == len(capabilities):
                results.append(agent)
            elif not require_all and matching > 0:
                results.append(agent)

        return results

    def find_best_for_task(self, task: Task) -> Optional[AgentDescriptor]:
        """
        Find the best available agent for a task.

        Considers:
        - Required capabilities
        - Current load
        - Agent health
        - Average response time

        Args:
            task: The task to assign

        Returns:
            Best available agent, or None if no suitable agent found
        """
        candidates = self.discover_multi(task.required_capabilities, require_all=True)

        if not candidates:
            # Try with ANY required capability
            candidates = self.discover_multi(task.required_capabilities, require_all=False)

        if not candidates:
            return None

        # Score each candidate
        scored = []
        for agent in candidates:
            score = self._calculate_assignment_score(agent, task)
            if score > 0:
                scored.append((agent, score))

        if not scored:
            return None

        # Return highest scoring agent
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def _calculate_assignment_score(self, agent: AgentDescriptor, task: Task) -> float:
        """Calculate a score for assigning a task to an agent."""
        agent_key = str(agent.id)
        state = self._states.get(agent_key)

        # Check if agent is available
        if state and state.state_type not in (AgentStateType.IDLE, AgentStateType.WAITING):
            return 0.0

        # Check health
        if not self._is_healthy(agent.id):
            return 0.0

        # Check concurrent task limit
        current_tasks = self._task_counts.get(agent_key, 0)
        if current_tasks >= agent.max_concurrent_tasks:
            return 0.0

        # Base score from capability confidence
        confidence_sum = sum(
            agent.get_capability_confidence(cap)
            for cap in task.required_capabilities
        )
        base_score = confidence_sum / len(task.required_capabilities)

        # Penalty for current load
        load_factor = 1.0 - (current_tasks / agent.max_concurrent_tasks) * 0.5

        # Bonus for faster response time (normalized to 0-1)
        speed_factor = 1.0 - min(agent.average_response_time_ms / 10000, 1.0) * 0.3

        return base_score * load_factor * speed_factor

    def heartbeat(self, agent_id: AgentId) -> None:
        """Record a heartbeat from an agent."""
        agent_key = str(agent_id)
        if agent_key in self._agents:
            self._heartbeats[agent_key] = datetime.now()

    def _is_healthy(self, agent_id: AgentId) -> bool:
        """Check if an agent is healthy (recent heartbeat)."""
        agent_key = str(agent_id)
        last_heartbeat = self._heartbeats.get(agent_key)

        if not last_heartbeat:
            return False

        age = (datetime.now() - last_heartbeat).total_seconds()
        return age < HEARTBEAT_TIMEOUT_SECONDS

    def get_healthy_agents(self) -> List[AgentDescriptor]:
        """Get all healthy agents."""
        return [
            agent for agent in self._agents.values()
            if self._is_healthy(agent.id)
        ]

    def get_unhealthy_agents(self) -> List[AgentDescriptor]:
        """Get all unhealthy agents."""
        return [
            agent for agent in self._agents.values()
            if not self._is_healthy(agent.id)
        ]

    def increment_task_count(self, agent_id: AgentId) -> None:
        """Increment the active task count for an agent."""
        agent_key = str(agent_id)
        self._task_counts[agent_key] = self._task_counts.get(agent_key, 0) + 1

    def decrement_task_count(self, agent_id: AgentId) -> None:
        """Decrement the active task count for an agent."""
        agent_key = str(agent_id)
        self._task_counts[agent_key] = max(0, self._task_counts.get(agent_key, 0) - 1)

    def record_metric(self, agent_id: AgentId, metric: str, value: float) -> None:
        """Record a performance metric for an agent."""
        agent_key = str(agent_id)
        self._metrics[agent_key][metric] = value

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total = len(self._agents)
        healthy = len(self.get_healthy_agents())
        unhealthy = total - healthy

        by_type = defaultdict(int)
        for agent in self._agents.values():
            by_type[agent.agent_type] += 1

        total_capacity = sum(a.max_concurrent_tasks for a in self._agents.values())
        current_load = sum(self._task_counts.values())

        return {
            'total_agents': total,
            'healthy_agents': healthy,
            'unhealthy_agents': unhealthy,
            'by_type': dict(by_type),
            'total_capacity': total_capacity,
            'current_load': current_load,
            'load_percentage': (current_load / total_capacity * 100) if total_capacity > 0 else 0
        }

    def get_agent_details(self) -> List[Dict[str, Any]]:
        """Get detailed info for all agents."""
        details = []
        for agent in self._agents.values():
            agent_key = str(agent.id)
            state = self._states.get(agent_key, AgentState.idle())

            details.append({
                **agent.to_dict(),
                'state': state.to_dict(),
                'healthy': self._is_healthy(agent.id),
                'active_tasks': self._task_counts.get(agent_key, 0),
                'metrics': dict(self._metrics.get(agent_key, {}))
            })

        return details

    def create_standard_team(self) -> List[AgentId]:
        """Create a standard team of agents."""
        agents = [
            create_standard_agent("SENIOR_DEVELOPER", "Senior Dev"),
            create_standard_agent("JUNIOR_DEVELOPER", "Junior Dev"),
            create_standard_agent("TESTER", "QA Tester"),
            create_standard_agent("CODE_REVIEWER", "Code Reviewer"),
            create_standard_agent("DOCUMENTATION_WRITER", "Doc Writer"),
            create_standard_agent("DEBUGGER", "Bug Hunter"),
        ]

        return [self.register(agent) for agent in agents]


# Global instance
_registry: Optional[AgentRegistry] = None


def get_registry() -> AgentRegistry:
    """Get or create the global agent registry."""
    global _registry

    if _registry is None:
        _registry = AgentRegistry()

    return _registry
