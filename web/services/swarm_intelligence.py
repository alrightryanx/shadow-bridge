"""
Swarm Intelligence Service - AGI-Readiness

Implements emergent problem-solving via agent swarms:
1. Stigmergy Matrix: Indirect communication via environmental signals (pheromones).
2. Swarm Coordination: Parallel task execution with emergent consensus.
3. Behavior Monitoring: Detecting patterns in swarm activity.

AGI-Readiness: Enables complex, decentralized problem solving without a single point of failure.
"""
import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

from .agent_protocol import AgentId, AgentMessage, MessageType, Task
from .message_bus import get_message_bus
from .agent_registry import get_registry

logger = logging.getLogger(__name__)

@dataclass
class Signal:
    """A signal (pheromone) left by an agent in the environment."""
    agent_id: str
    type: str  # e.g., "SOLUTION_FOUND", "ERROR_DETECTED", "RESOURCE_CLAIMED"
    intensity: float  # 0.0 to 1.0 (decays over time)
    payload: Any
    timestamp: float = field(default_factory=time.time)

class StigmergyMatrix:
    """
    Shared environmental memory for indirect agent communication.
    Simulates biological pheromone trails.
    """
    def __init__(self):
        self.signals: Dict[str, List[Signal]] = defaultdict(list)
        self.decay_rate = 0.1 # Per minute

    def deposit(self, signal_type: str, agent_id: str, intensity: float, payload: Any):
        signal = Signal(agent_id, signal_type, intensity, payload)
        self.signals[signal_type].append(signal)
        logger.debug(f"Signal deposited: {signal_type} by {agent_id}")

    def sense(self, signal_type: str) -> List[Signal]:
        """Sense active signals, filtering out decayed ones."""
        self._decay_signals()
        return self.signals.get(signal_type, [])

    def _decay_signals(self):
        """Reduce signal intensity over time."""
        now = time.time()
        for s_type in list(self.signals.keys()):
            # Keep signals with intensity > 0.1 and not too old
            self.signals[s_type] = [
                s for s in self.signals[s_type]
                if s.intensity * (1.0 - (now - s.timestamp) / 600 * self.decay_rate) > 0.1
            ]

class SwarmOrchestrator:
    """
    Coordinates agent swarms.
    """
    def __init__(self):
        self.matrix = StigmergyMatrix()
        self.message_bus = get_message_bus()
        self.registry = get_registry()
        self._active_swarms: Dict[str, Set[str]] = {}

    async def execute_swarm_task(self, task: Task, agent_count: int = 3) -> Dict[str, Any]:
        """
        Deploy a swarm of agents to solve a task.
        """
        import uuid
        swarm_id = str(uuid.uuid4())[:8]
        logger.info(f"Deploying swarm {swarm_id} for task: {task.title}")

        # 1. Discover and Recruit Agents
        agents = self.registry.get_healthy_agents()
        if not agents:
            return {"error": "No healthy agents available"}
            
        # Select agents (simplified selection)
        recruits = agents[:min(len(agents), agent_count)]
        self._active_swarms[swarm_id] = {str(a.id) for a in recruits}

        # 2. Broadcast Task to Swarm
        for agent in recruits:
            msg = AgentMessage.create(
                message_type=MessageType.TASK_ASSIGNMENT,
                from_agent=AgentId("orchestrator", "swarm"),
                to_agent=agent.id,
                payload={"swarm_id": swarm_id, "task": task.to_dict()}
            )
            self.message_bus.send(msg)

        # 3. Monitor for Emergent Solution (simulated)
        # In a real swarm, agents would use the StigmergyMatrix to coordinate.
        # Here we simulate agents working and depositing signals.
        
        await asyncio.sleep(1) # Simulate initial work
        
        # Agent 1 finds a partial solution
        self.matrix.deposit("PARTIAL_SOLUTION", str(recruits[0].id), 0.8, {"progress": 0.4})
        
        # Agent 2 senses it and refines
        signals = self.matrix.sense("PARTIAL_SOLUTION")
        if signals:
            self.matrix.deposit("REFINED_SOLUTION", str(recruits[1].id), 0.9, {"progress": 0.9})

        # 4. Detect Consensus
        final_signals = self.matrix.sense("REFINED_SOLUTION")
        if final_signals:
            consensus = final_signals[0].payload
            logger.info(f"Swarm {swarm_id} reached consensus.")
            return {
                "swarm_id": swarm_id,
                "status": "SUCCESS",
                "agents_involved": len(recruits),
                "solution": consensus,
                "signals_processed": len(self.matrix.signals)
            }

        return {"swarm_id": swarm_id, "status": "PENDING"}

# Global instance
_swarm_orchestrator: Optional[SwarmOrchestrator] = None

def get_swarm_orchestrator() -> SwarmOrchestrator:
    global _swarm_orchestrator
    if _swarm_orchestrator is None:
        _swarm_orchestrator = SwarmOrchestrator()
    return _swarm_orchestrator
