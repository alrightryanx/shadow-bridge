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

    def __init__(self, ai_service=None):
        self.ai_service = ai_service  # To use LLM for assessment

    def assess_query(self, query: str, context: Optional[Dict] = None) -> CognitiveAssessment:
        """
        Assess a user query before execution.
        
        In a full implementation, this would call a fast LLM to evaluate.
        For prototype, we use heuristics.
        """
        # Heuristic assessment
        confidence = 0.9
        ambiguity = False
        missing = []
        risk = "LOW"
        reasoning = "Query appears straightforward."
        strategy = "DIRECT_EXECUTION"

        lower_q = query.lower()

        # Ambiguity detection
        if " it " in lower_q or " that " in lower_q or " this " in lower_q:
            confidence -= 0.2
            ambiguity = True
            reasoning = "Query contains ambiguous pronouns ('it', 'that'). Context required."
            strategy = "CLARIFY_OR_INFER"

        # Missing info detection
        if "email" in lower_q and "@" not in lower_q and "last" not in lower_q:
             confidence -= 0.3
             missing.append("recipient_email")
             reasoning = "User asked to email but didn't specify recipient."
             strategy = "ASK_CLARIFICATION"

        # Risk assessment
        if "delete" in lower_q or "remove" in lower_q or "kill" in lower_q:
            risk = "MEDIUM"
            if "all" in lower_q or "system" in lower_q:
                risk = "HIGH"
                confidence -= 0.1
                reasoning = "Destructive command detected. High risk."
                strategy = "REQUIRE_CONFIRMATION"

        # Context improvement
        if context and context.get("active_context"):
            confidence += 0.1
            reasoning += " Active context available."

        return CognitiveAssessment(
            query=query,
            confidence=min(1.0, max(0.1, confidence)),
            ambiguity_detected=ambiguity,
            missing_info=missing,
            risk_level=risk,
            reasoning=reasoning,
            suggested_strategy=strategy
        )

    def critique_plan(self, plan: str, goal: str) -> SelfCritique:
        """
        Self-critique a generated plan.
        """
        critiques = []
        score = 0.8
        
        # Heuristic critique
        if len(plan.split()) < 5:
            critiques.append("Plan is too brief.")
            score -= 0.2
        
        if "sudo" in plan:
            critiques.append("Plan uses elevated privileges.")
            score -= 0.1
            
        return SelfCritique(
            original_plan=plan,
            critique_points=critiques,
            score=min(1.0, max(0.1, score)),
            improved_plan=None # In real impl, would generate improvement
        )

# Global instance
_metacognitive_layer: Optional[MetacognitiveLayer] = None

def get_metacognitive_layer() -> MetacognitiveLayer:
    global _metacognitive_layer
    if _metacognitive_layer is None:
        _metacognitive_layer = MetacognitiveLayer()
    return _metacognitive_layer
