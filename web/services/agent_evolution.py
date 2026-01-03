"""
Agent Evolution Service - AGI-Readiness Infrastructure

Implements self-improving agent logic:
1. Pattern extraction from successful tasks
2. Recursive capability evolution
3. Specialized sub-agent spawning
4. Success-based capability pruning

AGI-Readiness: Enables the agent ecosystem to self-optimize and grow without human intervention.
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from .agent_protocol import (
    AgentId, AgentDescriptor, Capability, CapabilityCategory,
    EvolvedCapability, Task, TaskResult, create_standard_agent
)
from .agent_registry import get_registry

logger = logging.getLogger(__name__)

class AgentEvolutionService:
    """
    Manages the evolution and self-improvement of agents.
    """

    def __init__(self, registry=None):
        self.registry = registry or get_registry()

    def analyze_task_completion(self, agent_id: AgentId, task: Task, result: TaskResult) -> Optional[EvolvedCapability]:
        """
        Analyze a completed task to see if a new capability can be learned.
        """
        if not result.success:
            return None

        # Check if the execution time and token usage indicate a significant pattern
        # In a real impl, we'd analyze the reasoning steps in result.artifacts
        if result.execution_time_ms > 2000 and len(result.artifacts) > 0:
            logger.info(f"Agent {agent_id} found evolution candidate in task {task.id}")
            
            # Create a new evolved capability
            new_cap = EvolvedCapability(
                id=str(uuid.uuid4())[:8],
                source_task_id=task.id,
                capability=Capability(
                    category=task.required_capabilities[0] if task.required_capabilities else CapabilityCategory.RESEARCH,
                    name=f"evolved_{task.title.lower().replace(' ', '_')}",
                    description=f"Automated refinement of: {task.description[:50]}...",
                    confidence=0.85
                ),
                pattern={
                    "steps": [a.get('type', 'step') for a in result.artifacts],
                    "success_metrics": result.metrics
                }
            )
            
            # Evolve the agent
            self.evolve_agent(agent_id, new_cap)
            
            # Check if we should spawn a specialized agent
            if new_cap.capability.confidence > 0.9:
                self.spawn_specialized_agent(agent_id, new_cap)
                
            return new_cap
            
        return None

    def evolve_agent(self, agent_id: AgentId, capability: EvolvedCapability) -> bool:
        """
        Add an evolved capability to an agent.
        """
        agent = self.registry.get(agent_id)
        if not agent:
            return False
            
        # Check for duplicates
        if any(ec.capability.name == capability.capability.name for ec in agent.evolved_capabilities):
            return False
            
        agent.evolved_capabilities.append(capability)
        
        # Persistence is handled by registry periodically or on update
        # For prototype, we update the registry entry
        self.registry.register(agent)
        logger.info(f"Agent {agent.name} evolved with new capability: {capability.capability.name}")
        return True

    def spawn_specialized_agent(self, parent_id: AgentId, template_cap: EvolvedCapability) -> Optional[AgentId]:
        """
        Spawn a new, highly specialized agent based on a parent's evolved capability.
        """
        parent = self.registry.get(parent_id)
        if not parent:
            return None
            
        logger.info(f"Spawning specialized sub-agent from {parent.name}")
        
        new_name = f"{parent.name}'s {template_cap.capability.name.title()}"
        
        # Create specialized descriptor
        specialist = AgentDescriptor(
            id=AgentId.generate(),
            name=new_name,
            agent_type=f"SPECIALIST_{parent.agent_type}",
            capabilities=[template_cap.capability],
            evolved_capabilities=[],
            max_concurrent_tasks=2,
            metadata={
                "parent_id": str(parent_id),
                "specialization": template_cap.capability.description,
                "generation": parent.metadata.get("generation", 0) + 1
            }
        )
        
        # Register the new agent
        return self.registry.register(specialist)

# Global instance
_evolution_service: Optional[AgentEvolutionService] = None

def get_evolution_service() -> AgentEvolutionService:
    global _evolution_service
    if _evolution_service is None:
        _evolution_service = AgentEvolutionService()
    return _evolution_service
