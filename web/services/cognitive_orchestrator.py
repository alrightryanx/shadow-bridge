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

logger = logging.getLogger(__name__)

@dataclass
class CognitivePlan:
    """The decided course of action."""
    strategy: str  # DIRECT, DEBATE, CLARIFY, BLOCKED
    reasoning: str
    risk_level: str
    confidence: float
    execution_steps: list
    context_used: Dict
    debate_session_id: Optional[str] = None

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

    async def process_query(self, query: str, client_context: Optional[Dict] = None) -> CognitivePlan:
        """
        Process a user query through the cognitive architecture.
        """
        logger.info(f"Cognitive Orchestrator processing: '{query}'")

        # 1. Metacognitive Assessment
        # "Think about thinking" - is this dangerous? Ambiguous?
        assessment = self.meta_layer.assess_query(query, client_context)
        logger.info(f"Assessment: Risk={assessment.risk_level}, Conf={assessment.confidence}")

        # 2. Context Assembly
        # Build the context window based on the query
        context_window = self.context_service.build_context_window(query)
        
        # 3. Strategy Selection & Planning
        plan = await self._formulate_plan(query, assessment, context_window)
        
        return plan

    async def _formulate_plan(self, query: str, assessment: CognitiveAssessment, context: Any) -> CognitivePlan:
        """Decide on a strategy based on assessment."""
        
        # Strategy: DEBATE
        # Triggered by High Risk or Low Confidence (but not due to simple ambiguity)
        if assessment.risk_level == "HIGH" or (assessment.confidence < 0.6 and not assessment.ambiguity_detected):
            logger.info("Triggering Debate System due to Risk/Confidence")
            
            # Start a debate to refine the approach
            debate_topic = f"How to safely handle: {query}"
            session = await self.debate_system.conduct_debate(topic=debate_topic, rounds=2)
            
            return CognitivePlan(
                strategy="DEBATE",
                reasoning=f"Query deemed {assessment.risk_level} risk. Debate conducted to ensure safety.",
                risk_level=assessment.risk_level,
                confidence=assessment.confidence, # Initial confidence
                execution_steps=["Review Debate Consensus", "Execute Synthesized Plan"],
                context_used={"active_items": len(context.active_context or [])},
                debate_session_id=session.session_id
            )

        # Strategy: CLARIFY
        # Triggered by Ambiguity or Missing Info
        if assessment.ambiguity_detected or assessment.missing_info:
            return CognitivePlan(
                strategy="CLARIFY",
                reasoning=assessment.reasoning,
                risk_level=assessment.risk_level,
                confidence=assessment.confidence,
                execution_steps=["Ask User for Clarification"],
                context_used={}
            )

        # Strategy: DIRECT
        # Standard execution path
        steps = ["Load Context", "Execute Command"]
        
        # Predictive Pre-loading hook (already handled in context service/API, but could be reinforced here)
        # In a full system, we might refine the steps here.

        return CognitivePlan(
            strategy="DIRECT",
            reasoning="Query is clear and low risk. Proceeding with execution.",
            risk_level=assessment.risk_level,
            confidence=assessment.confidence,
            execution_steps=steps,
            context_used={"token_estimate": context.token_estimate}
        )

# Global instance
_orchestrator: Optional[CognitiveOrchestrator] = None

def get_cognitive_orchestrator() -> CognitiveOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = CognitiveOrchestrator()
    return _orchestrator
