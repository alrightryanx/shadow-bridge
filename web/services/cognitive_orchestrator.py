"""
Cognitive Orchestrator - The "Brain" of ShadowAI

Integrates all cognitive services into a cohesive decision-making pipeline:
1. Receives User Query
2. Metacognitive Assessment (Risk/Confidence)
3. Strategy Selection (Direct, Debate, Clarify)
4. Context Assembly (Predictive/Reactive)
5. Execution Planning

AGI-Readiness: This is the central control loop for autonomous behavior.
"""
import logging
import asyncio
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict

from .metacognition import get_metacognitive_layer, CognitiveAssessment
from .debate_system import get_debate_system
from .context_window import get_context_service
from .llm_gateway import get_llm_gateway

logger = logging.getLogger(__name__)

@dataclass
class CognitivePlan:
    """The decided course of action."""
    strategy: str  # DIRECT, DEBATE, CLARIFY, SWARM, RECURSIVE, STAFF, BLOCKED
    reasoning: str
    risk_level: str
    confidence: float
    execution_steps: list
    context_used: Dict
    debate_session_id: Optional[str] = None
    task_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

class CognitiveOrchestrator:
    """
    Orchestrates the cognitive pipeline.
    """

    def __init__(self):
        self.meta_layer = get_metacognitive_layer()
        self.debate_system = get_debate_system()
        self.context_service = get_context_service()
        self.llm = get_llm_gateway()

    async def process_query(self, query: str, client_context: Optional[Dict] = None) -> CognitivePlan:
        """
        Process a user query through the cognitive architecture.
        """
        logger.info(f"Cognitive Orchestrator processing: '{query}'")

        # 1. Metacognitive Assessment
        assessment = await self.meta_layer.assess_query(query, client_context)
        logger.info(f"Assessment: Risk={assessment.risk_level}, Conf={assessment.confidence}")

        # 2. Context Assembly
        context_window = self.context_service.build_context_window(query)
        
        # 3. Intelligent Routing (AGI Brain)
        routing_decision = await self._route_intelligently(query, assessment, context_window)
        
        # 4. Strategy Execution Plan
        plan = await self._formulate_plan(query, assessment, context_window, routing_decision)
        
        return plan

    async def _route_intelligently(self, query: str, assessment: CognitiveAssessment, context: Any) -> Dict[str, Any]:
        """Use LLM to decide the best strategy for the task."""
        system_prompt = """You are the Supervisor Brain of ShadowAI, an AGI Orchestrator.
Your goal is to route user queries to the most appropriate cognitive strategy.

STRATEGIES:
1. DIRECT: Simple tasks, questions, or commands that can be answered or executed immediately.
2. CLARIFY: Queries that are too ambiguous or missing critical information.
3. SWARM: Complex technical tasks that benefit from parallel processing or multiple perspectives (e.g., optimizing code, complex research).
4. RECURSIVE: Deep thinking tasks, logic puzzles, or tasks requiring extreme precision where self-correction is needed.
5. STAFF: Long-running, multi-step projects (e.g., "Build an app", "Write and publish a book").
6. DEBATE: High-risk decisions or ethical dilemmas where multiple viewpoints are needed before acting.

Return your decision in JSON format:
{
  "strategy": "STRATEGY_NAME",
  "reasoning": "Brief explanation of why this strategy was chosen",
  "confidence": 0.0-1.0
}"""
        
        prompt = f"User Query: {query}\nRisk Level: {assessment.risk_level}\nAmbiguity: {assessment.ambiguity_detected}\nContext: {context.summary if hasattr(context, 'summary') else 'None'}"
        
        try:
            decision = await self.llm.ask_json(prompt, system_prompt)
            # Validate strategy
            valid_strategies = ["DIRECT", "CLARIFY", "SWARM", "RECURSIVE", "STAFF", "DEBATE"]
            if decision.get("strategy") not in valid_strategies:
                decision["strategy"] = "DIRECT" # Default
            return decision
        except Exception as e:
            logger.error(f"Routing error: {e}")
            return {"strategy": "DIRECT", "reasoning": "Fallback to direct due to routing error.", "confidence": 0.5}

    async def _formulate_plan(self, query: str, assessment: CognitiveAssessment, context: Any, routing: Dict[str, Any]) -> CognitivePlan:
        """Refine the plan based on routing decision and assessment."""
        
        strategy = routing.get("strategy", "DIRECT")
        reasoning = routing.get("reasoning", "Standard execution.")
        confidence = routing.get("confidence", assessment.confidence)

        # Force DEBATE if risk is HIGH regardless of LLM decision
        if assessment.risk_level == "HIGH" and strategy != "DEBATE":
            strategy = "DEBATE"
            reasoning = "High risk detected by metacognitive layer. Overriding to DEBATE strategy for safety."

        # Map strategies to execution steps
        steps = []
        debate_id = None
        
        if strategy == "DEBATE":
            debate_topic = f"Safety and efficacy of: {query}"
            session = await self.debate_system.conduct_debate(topic=debate_topic, rounds=2)
            debate_id = session.session_id
            steps = ["Conduct Multi-Model Debate", "Synthesize Consensus", "Safe Execution"]
        elif strategy == "SWARM":
            steps = ["Recruit Agent Swarm", "Broadcase Task", "Monitor Stigmergy Matrix", "Aggregate Result"]
        elif strategy == "RECURSIVE":
            steps = ["Initialize Reasoning Loop", "Recursive Refinement", "Convergence Check"]
        elif strategy == "STAFF":
            steps = ["Initialize Agent Staff", "Supervisor Assignment", "Multi-Agent Workflow"]
        elif strategy == "CLARIFY":
            steps = ["Formulate Clarification Question", "Wait for User Input"]
        else: # DIRECT
            steps = ["Assemble Context", "Execute Direct Command"]

        return CognitivePlan(
            strategy=strategy,
            reasoning=reasoning,
            risk_level=assessment.risk_level,
            confidence=confidence,
            execution_steps=steps,
            context_used={"token_estimate": getattr(context, 'token_estimate', 0)},
            debate_session_id=debate_id
        )

# Global instance
_orchestrator: Optional[CognitiveOrchestrator] = None

def get_cognitive_orchestrator() -> CognitiveOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = CognitiveOrchestrator()
    return _orchestrator
