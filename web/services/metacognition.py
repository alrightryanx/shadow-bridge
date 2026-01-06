"""
Metacognitive Awareness Layer - AGI-Readiness

Enables the AI to "think about its own thinking":
1. Assess confidence in its understanding
2. Detect ambiguity or missing information
3. Critique its own plans before execution
4. Propose alternative strategies

AGI-Readiness: Essential for safe, autonomous operation.
"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json

from .llm_gateway import get_llm_gateway

logger = logging.getLogger(__name__)

@dataclass
class CognitiveAssessment:
    """Assessment of a query or plan."""
    query: str
    confidence: float  # 0.0 to 1.0
    ambiguity_detected: bool
    missing_info: List[str]
    risk_level: str  # LOW, MEDIUM, HIGH
    reasoning: str
    suggested_strategy: str

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class SelfCritique:
    """Critique of a generated response or plan."""
    original_plan: str
    critique_points: List[str]
    score: float  # 0.0 to 1.0
    improved_plan: Optional[str]

    def to_dict(self) -> Dict:
        return asdict(self)

class MetacognitiveLayer:
    """
    Layer for higher-order cognitive functions.
    """

    def __init__(self):
        self.llm = get_llm_gateway()

    async def assess_query(self, query: str, context: Optional[Dict] = None) -> CognitiveAssessment:
        """
        Assess a user query before execution using LLM.
        """
        system_prompt = """You are the Metacognitive Layer of ShadowAI.
Your goal is to "think about the thinking" required for this query.
Assess the RISK, AMBIGUITY, and MISSING INFORMATION.

Return JSON:
{
  "confidence": 0.0-1.0,
  "ambiguity_detected": true/false,
  "missing_info": ["item1", ...],
  "risk_level": "LOW/MEDIUM/HIGH",
  "reasoning": "Why this assessment?",
  "suggested_strategy": "DIRECT/DEBATE/CLARIFY"
}"""
        
        prompt = f"Assess Query: {query}\nContext: {json.dumps(context) if context else 'None'}"
        
        try:
            assessment_data = await self.llm.ask_json(prompt, system_prompt)
            return CognitiveAssessment(
                query=query,
                confidence=assessment_data.get("confidence", 0.5),
                ambiguity_detected=assessment_data.get("ambiguity_detected", False),
                missing_info=assessment_data.get("missing_info", []),
                risk_level=assessment_data.get("risk_level", "LOW"),
                reasoning=assessment_data.get("reasoning", "LLM Assessment"),
                suggested_strategy=assessment_data.get("suggested_strategy", "DIRECT")
            )
        except Exception as e:
            logger.error(f"Metacognitive assessment error: {e}")
            # Fallback to heuristic
            return self._heuristic_assessment(query, context)

    def _heuristic_assessment(self, query: str, context: Optional[Dict] = None) -> CognitiveAssessment:
        """Fallback heuristic assessment."""
        lower_q = query.lower()
        risk = "HIGH" if "delete" in lower_q or "remove" in lower_q else "LOW"
        return CognitiveAssessment(
            query=query,
            confidence=0.5,
            ambiguity_detected=False,
            missing_info=[],
            risk_level=risk,
            reasoning="Fallback heuristic used.",
            suggested_strategy="DIRECT"
        )

    async def critique_plan(self, plan: str, goal: str) -> SelfCritique:
        """
        Self-critique a generated plan using LLM.
        """
        system_prompt = """Critique the following plan for achieving the goal.
Be critical and identify potential failure points.

Return JSON:
{
  "critique_points": ["point1", ...],
  "score": 0.0-1.0,
  "improved_plan": "..."
}"""
        
        prompt = f"Goal: {goal}\nPlan: {plan}"
        
        try:
            critique_data = await self.llm.ask_json(prompt, system_prompt)
            return SelfCritique(
                original_plan=plan,
                critique_points=critique_data.get("critique_points", []),
                score=critique_data.get("score", 0.5),
                improved_plan=critique_data.get("improved_plan")
            )
        except Exception as e:
            logger.error(f"Critique error: {e}")
            return SelfCritique(plan, ["Critique system error"], 0.5, None)

# Global instance
_metacognitive_layer: Optional[MetacognitiveLayer] = None

def get_metacognitive_layer() -> MetacognitiveLayer:
    global _metacognitive_layer
    if _metacognitive_layer is None:
        _metacognitive_layer = MetacognitiveLayer()
    return _metacognitive_layer
