"""
Agent Protocol Specification - AGI-Readiness Infrastructure

Defines the standard protocol for agent communication:
1. Agent lifecycle states
2. Message types and formats
3. Capability declarations
4. Task assignment protocol

This enables multi-agent coordination and prepares for AGI orchestration.
"""
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from uuid import uuid4
import json

logger = logging.getLogger(__name__)


# ============ Agent Identity ============

@dataclass
class AgentId:
    """Unique agent identifier."""
    id: str
    namespace: str = "shadowai"

    def __str__(self) -> str:
        return f"{self.namespace}:{self.id}"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other) -> bool:
        if isinstance(other, AgentId):
            return str(self) == str(other)
        return False

    @classmethod
    def generate(cls, namespace: str = "shadowai") -> 'AgentId':
        return cls(id=str(uuid4())[:8], namespace=namespace)

    def to_dict(self) -> Dict:
        return {'id': self.id, 'namespace': self.namespace}

    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentId':
        return cls(id=data['id'], namespace=data.get('namespace', 'shadowai'))


# ============ Agent State ============

class AgentStateType(Enum):
    """Agent lifecycle states."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    WAITING = "waiting"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class AgentState:
    """Current state of an agent."""
    state_type: AgentStateType
    task_id: Optional[str] = None
    reason: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    started_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            'state_type': self.state_type.value,
            'task_id': self.task_id,
            'reason': self.reason,
            'progress': self.progress,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None
        }

    @classmethod
    def idle(cls) -> 'AgentState':
        return cls(state_type=AgentStateType.IDLE)

    @classmethod
    def running(cls, task_id: str, progress: float = 0.0) -> 'AgentState':
        return cls(
            state_type=AgentStateType.RUNNING,
            task_id=task_id,
            progress=progress,
            started_at=datetime.now(),
            last_activity=datetime.now()
        )


# ============ Capabilities ============

class CapabilityCategory(Enum):
    """Categories of agent capabilities."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    DATA_ANALYSIS = "data_analysis"
    COMMUNICATION = "communication"
    PLANNING = "planning"
    DEBUGGING = "debugging"
    DEPLOYMENT = "deployment"
    SECURITY = "security"
    DESIGN = "design"


@dataclass
class Capability:
    """A specific capability an agent can provide."""
    category: CapabilityCategory
    name: str
    description: str
    confidence: float = 1.0  # 0.0 to 1.0 - how well the agent performs this
    requirements: List[str] = field(default_factory=list)  # Prerequisites

    def to_dict(self) -> Dict:
        return {
            'category': self.category.value,
            'name': self.name,
            'description': self.description,
            'confidence': self.confidence,
            'requirements': self.requirements
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Capability':
        return cls(
            category=CapabilityCategory(data['category']),
            name=data['name'],
            description=data['description'],
            confidence=data.get('confidence', 1.0),
            requirements=data.get('requirements', [])
        )


# ============ Agent Descriptor ============

@dataclass
class AgentDescriptor:
    """Complete description of an agent."""
    id: AgentId
    name: str
    agent_type: str  # SENIOR_DEVELOPER, TESTER, etc.
    capabilities: List[Capability]
    max_concurrent_tasks: int = 1
    cost_per_token: float = 0.0
    average_response_time_ms: int = 1000
    backend: Optional[str] = None  # Which LLM backend
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'id': self.id.to_dict(),
            'name': self.name,
            'agent_type': self.agent_type,
            'capabilities': [c.to_dict() for c in self.capabilities],
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'cost_per_token': self.cost_per_token,
            'average_response_time_ms': self.average_response_time_ms,
            'backend': self.backend,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentDescriptor':
        return cls(
            id=AgentId.from_dict(data['id']),
            name=data['name'],
            agent_type=data['agent_type'],
            capabilities=[Capability.from_dict(c) for c in data.get('capabilities', [])],
            max_concurrent_tasks=data.get('max_concurrent_tasks', 1),
            cost_per_token=data.get('cost_per_token', 0.0),
            average_response_time_ms=data.get('average_response_time_ms', 1000),
            backend=data.get('backend'),
            metadata=data.get('metadata', {})
        )

    def has_capability(self, category: CapabilityCategory) -> bool:
        return any(c.category == category for c in self.capabilities)

    def get_capability_confidence(self, category: CapabilityCategory) -> float:
        for c in self.capabilities:
            if c.category == category:
                return c.confidence
        return 0.0


# ============ Message Protocol ============

class MessageType(Enum):
    """Types of inter-agent messages."""
    # Task lifecycle
    TASK_ASSIGNMENT = "task_assignment"
    TASK_ACCEPTED = "task_accepted"
    TASK_REJECTED = "task_rejected"
    PROGRESS_UPDATE = "progress_update"
    TASK_RESULT = "task_result"
    TASK_ERROR = "task_error"

    # Coordination
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"
    HANDOFF_REQUEST = "handoff_request"
    HANDOFF_RESPONSE = "handoff_response"

    # Human interaction
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_RESPONSE = "approval_response"
    CLARIFICATION_REQUEST = "clarification_request"
    CLARIFICATION_RESPONSE = "clarification_response"

    # System
    HEARTBEAT = "heartbeat"
    STATUS_REQUEST = "status_request"
    STATUS_RESPONSE = "status_response"
    SHUTDOWN = "shutdown"


@dataclass
class AgentMessage:
    """Message between agents or orchestrator."""
    id: str
    message_type: MessageType
    from_agent: AgentId
    to_agent: Optional[AgentId]  # None = broadcast
    payload: Dict[str, Any]
    timestamp: datetime
    reply_to: Optional[str] = None  # Message ID being replied to
    priority: int = 5  # 1-10, 1 = highest
    ttl_seconds: int = 300  # Time to live

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'message_type': self.message_type.value,
            'from_agent': self.from_agent.to_dict(),
            'to_agent': self.to_agent.to_dict() if self.to_agent else None,
            'payload': self.payload,
            'timestamp': self.timestamp.isoformat(),
            'reply_to': self.reply_to,
            'priority': self.priority,
            'ttl_seconds': self.ttl_seconds
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentMessage':
        return cls(
            id=data['id'],
            message_type=MessageType(data['message_type']),
            from_agent=AgentId.from_dict(data['from_agent']),
            to_agent=AgentId.from_dict(data['to_agent']) if data.get('to_agent') else None,
            payload=data['payload'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            reply_to=data.get('reply_to'),
            priority=data.get('priority', 5),
            ttl_seconds=data.get('ttl_seconds', 300)
        )

    @classmethod
    def create(
        cls,
        message_type: MessageType,
        from_agent: AgentId,
        to_agent: Optional[AgentId],
        payload: Dict[str, Any],
        **kwargs
    ) -> 'AgentMessage':
        return cls(
            id=str(uuid4())[:12],
            message_type=message_type,
            from_agent=from_agent,
            to_agent=to_agent,
            payload=payload,
            timestamp=datetime.now(),
            **kwargs
        )

    def is_expired(self) -> bool:
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl_seconds


# ============ Task Protocol ============

class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """A task to be executed by an agent."""
    id: str
    title: str
    description: str
    required_capabilities: List[CapabilityCategory]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[AgentId] = None
    parent_task_id: Optional[str] = None  # For subtasks
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'required_capabilities': [c.value for c in self.required_capabilities],
            'priority': self.priority.value,
            'status': self.status.value,
            'assigned_to': self.assigned_to.to_dict() if self.assigned_to else None,
            'parent_task_id': self.parent_task_id,
            'dependencies': self.dependencies,
            'created_at': self.created_at.isoformat(),
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'context': self.context,
            'result': self.result,
            'error': self.error
        }

    @classmethod
    def create(
        cls,
        title: str,
        description: str,
        required_capabilities: List[CapabilityCategory],
        **kwargs
    ) -> 'Task':
        return cls(
            id=str(uuid4())[:12],
            title=title,
            description=description,
            required_capabilities=required_capabilities,
            **kwargs
        )


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    success: bool
    output: Any
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    execution_time_ms: int = 0
    tokens_used: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


# ============ Protocol Validation ============

def validate_message(message: AgentMessage) -> List[str]:
    """Validate a message against the protocol."""
    errors = []

    if not message.id:
        errors.append("Message ID is required")

    if not message.from_agent:
        errors.append("From agent is required")

    if message.is_expired():
        errors.append("Message has expired")

    # Validate payload based on message type
    if message.message_type == MessageType.TASK_ASSIGNMENT:
        if 'task' not in message.payload:
            errors.append("TASK_ASSIGNMENT requires 'task' in payload")

    elif message.message_type == MessageType.PROGRESS_UPDATE:
        if 'progress' not in message.payload:
            errors.append("PROGRESS_UPDATE requires 'progress' in payload")
        elif not (0 <= message.payload['progress'] <= 1):
            errors.append("Progress must be between 0 and 1")

    elif message.message_type == MessageType.APPROVAL_REQUEST:
        if 'action' not in message.payload:
            errors.append("APPROVAL_REQUEST requires 'action' in payload")

    return errors


def validate_task(task: Task) -> List[str]:
    """Validate a task definition."""
    errors = []

    if not task.id:
        errors.append("Task ID is required")

    if not task.title:
        errors.append("Task title is required")

    if not task.required_capabilities:
        errors.append("At least one required capability must be specified")

    if task.deadline and task.deadline < datetime.now():
        errors.append("Task deadline is in the past")

    return errors


# ============ Standard Agent Types ============

STANDARD_AGENT_TYPES = {
    "EXECUTIVE_ASSISTANT": {
        "capabilities": [CapabilityCategory.PLANNING, CapabilityCategory.COMMUNICATION],
        "description": "Coordinates tasks and manages communication"
    },
    "SENIOR_DEVELOPER": {
        "capabilities": [CapabilityCategory.CODE_GENERATION, CapabilityCategory.CODE_REVIEW, CapabilityCategory.DEBUGGING],
        "description": "Writes and reviews complex code"
    },
    "JUNIOR_DEVELOPER": {
        "capabilities": [CapabilityCategory.CODE_GENERATION],
        "description": "Writes code under guidance"
    },
    "TESTER": {
        "capabilities": [CapabilityCategory.TESTING],
        "description": "Writes and runs tests"
    },
    "CODE_REVIEWER": {
        "capabilities": [CapabilityCategory.CODE_REVIEW, CapabilityCategory.SECURITY],
        "description": "Reviews code for quality and security"
    },
    "DOCUMENTATION_WRITER": {
        "capabilities": [CapabilityCategory.DOCUMENTATION],
        "description": "Writes documentation and comments"
    },
    "RESEARCHER": {
        "capabilities": [CapabilityCategory.RESEARCH, CapabilityCategory.DATA_ANALYSIS],
        "description": "Researches solutions and analyzes data"
    },
    "DEBUGGER": {
        "capabilities": [CapabilityCategory.DEBUGGING, CapabilityCategory.TESTING],
        "description": "Identifies and fixes bugs"
    },
    "DESIGNER": {
        "capabilities": [CapabilityCategory.DESIGN],
        "description": "Designs UI/UX and architecture"
    },
    "DEVOPS": {
        "capabilities": [CapabilityCategory.DEPLOYMENT, CapabilityCategory.SECURITY],
        "description": "Handles deployment and infrastructure"
    }
}


def create_standard_agent(
    agent_type: str,
    name: str,
    backend: Optional[str] = None
) -> AgentDescriptor:
    """Create an agent descriptor for a standard agent type."""
    if agent_type not in STANDARD_AGENT_TYPES:
        raise ValueError(f"Unknown agent type: {agent_type}")

    type_info = STANDARD_AGENT_TYPES[agent_type]
    capabilities = [
        Capability(
            category=cap,
            name=f"{cap.value}_{agent_type.lower()}",
            description=f"{cap.value} capability for {agent_type}"
        )
        for cap in type_info["capabilities"]
    ]

    return AgentDescriptor(
        id=AgentId.generate(),
        name=name,
        agent_type=agent_type,
        capabilities=capabilities,
        backend=backend,
        metadata={"description": type_info["description"]}
    )
