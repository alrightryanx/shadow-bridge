"""
Multi-Model Debate System - AGI-Readiness

Implements a system where multiple AI personas debate a topic 
to reach a higher-quality consensus:
1. Propose: Initial solutions from different perspectives.
2. Critique: Each persona critiques the others' solutions.
3. Refine: Personas update their solutions based on feedback.
4. Synthesize: A moderator synthesizes the final consensus.

AGI-Readiness: Reduces bias and increases reasoning quality through adversarial collaboration.
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
import time

logger = logging.getLogger(__name__)

@dataclass
class DebatePersona:
    """A specific persona for the debate."""
    name: str
    role: str
    description: str
    temperature: float = 0.7

@dataclass
class DebateTurn:
    """A single turn in the debate."""
    persona_name: str
    content: str
    turn_type: str  # PROPOSAL, CRITIQUE, REFINEMENT
    timestamp: float = field(default_factory=time.time)

@dataclass
class DebateSession:
    """A full debate session."""
    session_id: str
    topic: str
    rounds: int
    personas: List[DebatePersona]
    turns: List[DebateTurn] = field(default_factory=list)
    consensus: Optional[str] = None
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "rounds": self.rounds,
            "personas": [p.__dict__ for p in self.personas],
            "turns": [t.__dict__ for t in self.turns],
            "consensus": self.consensus,
            "status": self.status
        }

class MultiModelDebateSystem:
    """
    Orchestrates a debate between multiple AI personas.
    """

    def __init__(self, ai_provider=None):
        self.ai_provider = ai_provider
        self.default_personas = [
            DebatePersona(
                name="Optimist",
                role="VISIONARY",
                description="Focuses on potential, innovation, and positive outcomes."
            ),
            DebatePersona(
                name="Skeptic",
                role="CRITIC",
                description="Focuses on risks, edge cases, and potential failures."
            ),
            DebatePersona(
                name="Realist",
                role="PRAGMATIST",
                description="Focuses on feasibility, resources, and practical implementation."
            )
        ]
        self._sessions: Dict[str, DebateSession] = {}

    async def conduct_debate(
        self,
        topic: str,
        rounds: int = 2,
        personas: Optional[List[DebatePersona]] = None
    ) -> DebateSession:
        """
        Conduct a debate session.
        """
        import uuid
        session_id = str(uuid.uuid4())[:8]
        active_personas = personas or self.default_personas
        
        session = DebateSession(
            session_id=session_id,
            topic=topic,
            rounds=rounds,
            personas=active_personas,
            status="RUNNING"
        )
        self._sessions[session_id] = session

        try:
            # Round 1: Initial Proposals
            logger.info(f"Debate {session_id}: Round 1 (Proposals)")
            proposals = []
            for persona in active_personas:
                # In real impl, call AI provider with persona-specific prompt
                content = self._simulate_response(persona, topic, "PROPOSAL")
                turn = DebateTurn(persona.name, content, "PROPOSAL")
                session.turns.append(turn)
                proposals.append(turn)

            # Subsequent Rounds: Critique and Refinement
            for r in range(2, rounds + 1):
                logger.info(f"Debate {session_id}: Round {r} (Critique & Refinement)")
                
                # Critique Phase
                critiques = []
                for persona in active_personas:
                    others_work = [p.content for p in proposals if p.persona_name != persona.name]
                    content = self._simulate_response(persona, topic, "CRITIQUE", context=others_work)
                    turn = DebateTurn(persona.name, content, "CRITIQUE")
                    session.turns.append(turn)
                    critiques.append(turn)
                
                # Refinement Phase
                new_proposals = []
                for persona in active_personas:
                    my_critiques = [c.content for c in critiques if c.persona_name != persona.name]
                    content = self._simulate_response(persona, topic, "REFINEMENT", context=my_critiques)
                    turn = DebateTurn(persona.name, content, "REFINEMENT")
                    session.turns.append(turn)
                    new_proposals.append(turn)
                
                proposals = new_proposals

            # Final Phase: Synthesis
            logger.info(f"Debate {session_id}: Synthesis")
            session.consensus = self._simulate_synthesis(topic, session.turns)
            session.status = "COMPLETED"

        except Exception as e:
            logger.error(f"Debate {session_id} failed: {e}")
            session.status = "FAILED"
            raise

        return session

    def _simulate_response(self, persona: DebatePersona, topic: str, turn_type: str, context: Optional[List[str]] = None) -> str:
        """
        Simulate an AI response for the prototype.
        In production, this calls the LLM with a specialized prompt.
        """
        if turn_type == "PROPOSAL":
            if persona.name == "Optimist":
                return f"[Optimist View] This is a brilliant idea for '{topic}'. We should scale it globally using cutting-edge tech."
            if persona.name == "Skeptic":
                return f"[Skeptic View] I have concerns about '{topic}'. The security risks and scaling costs are too high."
            return f"[Realist View] For '{topic}', we need a phased approach with clear KPIs and a solid budget."
        
        if turn_type == "CRITIQUE":
            return f"[{persona.name} Critique] I've reviewed the other proposals. They miss the point about {context[0][:20]}..."
            
        if turn_type == "REFINEMENT":
            return f"[{persona.name} Refined] Based on the feedback, I've updated the plan for '{topic}' to be more robust."
            
        return "Thinking..."

    def _simulate_synthesis(self, topic: str, turns: List[DebateTurn]) -> str:
        """
        Simulate the moderator's synthesis.
        """
        return f"Consensus for '{topic}': We will proceed with a high-innovation strategy (Optimist) but with a mandatory zero-trust security audit (Skeptic) and a 3-month trial period (Realist)."

    def get_session(self, session_id: str) -> Optional[DebateSession]:
        return self._sessions.get(session_id)

# Global instance
_debate_system: Optional[MultiModelDebateSystem] = None

def get_debate_system() -> MultiModelDebateSystem:
    global _debate_system
    if _debate_system is None:
        _debate_system = MultiModelDebateSystem()
    return _debate_system
