import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
import time

from .llm_gateway import get_llm_gateway

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

    def __init__(self):
        self.llm = get_llm_gateway()
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
        Conduct a debate session using the LLM Gateway.
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
                content = await self._call_persona(persona, topic, "PROPOSAL")
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
                    content = await self._call_persona(persona, topic, "CRITIQUE", context=others_work)
                    turn = DebateTurn(persona.name, content, "CRITIQUE")
                    session.turns.append(turn)
                    critiques.append(turn)
                
                # Refinement Phase
                new_proposals = []
                for persona in active_personas:
                    my_critiques = [c.content for c in critiques if c.persona_name != persona.name]
                    content = await self._call_persona(persona, topic, "REFINEMENT", context=my_critiques)
                    turn = DebateTurn(persona.name, content, "REFINEMENT")
                    session.turns.append(turn)
                    new_proposals.append(turn)
                
                proposals = new_proposals

            # Final Phase: Synthesis
            logger.info(f"Debate {session_id}: Synthesis")
            session.consensus = await self._call_moderator(topic, session.turns)
            session.status = "COMPLETED"

        except Exception as e:
            logger.error(f"Debate {session_id} failed: {e}")
            session.status = "FAILED"
            raise

        return session

    async def _call_persona(self, persona: DebatePersona, topic: str, turn_type: str, context: Optional[List[str]] = None) -> str:
        """Call the LLM with a persona-specific prompt."""
        system_prompt = f"You are the {persona.name} persona ({persona.role}). {persona.description}"
        
        if turn_type == "PROPOSAL":
            prompt = f"Provide a detailed proposal for the topic: '{topic}'"
        elif turn_type == "CRITIQUE":
            others = "\n\n".join(context or [])
            prompt = f"Critique the following proposals for the topic '{topic}':\n\n{others}\n\nFocus on your unique perspective."
        elif turn_type == "REFINEMENT":
            critiques = "\n\n".join(context or [])
            prompt = f"Refine your proposal for '{topic}' based on these critiques:\n\n{critiques}"
        else:
            prompt = topic

        return await self.llm.ask(prompt, system_prompt)

    async def _call_moderator(self, topic: str, turns: List[DebateTurn]) -> str:
        """Call the LLM to synthesize the final consensus."""
        history = "\n\n".join([f"{t.persona_name} ({t.turn_type}): {t.content}" for t in turns])
        prompt = f"Synthesize a final consensus for the topic '{topic}' based on this debate history:\n\n{history}"
        
        return await self.llm.ask(prompt, "You are a neutral moderator focused on finding the best path forward.")

    def get_session(self, session_id: str) -> Optional[DebateSession]:
        return self._sessions.get(session_id)

    def get_session(self, session_id: str) -> Optional[DebateSession]:
        return self._sessions.get(session_id)

# Global instance
_debate_system: Optional[MultiModelDebateSystem] = None

def get_debate_system() -> MultiModelDebateSystem:
    global _debate_system
    if _debate_system is None:
        _debate_system = MultiModelDebateSystem()
    return _debate_system
