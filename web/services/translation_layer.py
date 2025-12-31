"""
Translation Layer - AGI-Readiness Infrastructure

Converts between agent-speak and human-readable formats:
1. Technical agent messages -> Plain English summaries
2. Complex task results -> Understandable explanations
3. Agent coordination events -> User notifications
4. Error messages -> Actionable guidance

This bridges the gap between AI operations and human understanding.
"""
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
import re

from .agent_protocol import (
    AgentId, AgentMessage, MessageType, AgentState, AgentStateType,
    Task, TaskStatus, TaskResult, CapabilityCategory
)

logger = logging.getLogger(__name__)


@dataclass
class HumanReadableMessage:
    """A message formatted for human consumption."""
    title: str
    summary: str
    details: Optional[str] = None
    action_required: bool = False
    suggested_actions: List[str] = None
    severity: str = "info"  # info, success, warning, error
    timestamp: datetime = None
    original_message_id: Optional[str] = None

    def __post_init__(self):
        if self.suggested_actions is None:
            self.suggested_actions = []
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'summary': self.summary,
            'details': self.details,
            'action_required': self.action_required,
            'suggested_actions': self.suggested_actions,
            'severity': self.severity,
            'timestamp': self.timestamp.isoformat(),
            'original_message_id': self.original_message_id
        }


class TranslationLayer:
    """
    Translates between agent-speak and human-readable formats.

    Handles:
    - Message type translations
    - Technical jargon simplification
    - Context-aware explanations
    - Action recommendations
    """

    def __init__(self):
        # Custom translators for specific message types
        self._translators: Dict[MessageType, Callable] = {
            MessageType.TASK_ASSIGNMENT: self._translate_task_assignment,
            MessageType.TASK_ACCEPTED: self._translate_task_accepted,
            MessageType.TASK_REJECTED: self._translate_task_rejected,
            MessageType.PROGRESS_UPDATE: self._translate_progress_update,
            MessageType.TASK_RESULT: self._translate_task_result,
            MessageType.TASK_ERROR: self._translate_task_error,
            MessageType.APPROVAL_REQUEST: self._translate_approval_request,
            MessageType.APPROVAL_RESPONSE: self._translate_approval_response,
            MessageType.CAPABILITY_QUERY: self._translate_capability_query,
            MessageType.CAPABILITY_RESPONSE: self._translate_capability_response,
            MessageType.HANDOFF_REQUEST: self._translate_handoff_request,
            MessageType.HEARTBEAT: self._translate_heartbeat,
            MessageType.STATUS_REQUEST: self._translate_status_request,
            MessageType.STATUS_RESPONSE: self._translate_status_response,
            MessageType.SHUTDOWN: self._translate_shutdown,
        }

        # Capability name translations
        self._capability_names = {
            CapabilityCategory.CODE_GENERATION: "writing code",
            CapabilityCategory.CODE_REVIEW: "reviewing code",
            CapabilityCategory.TESTING: "testing",
            CapabilityCategory.DOCUMENTATION: "documentation",
            CapabilityCategory.RESEARCH: "research",
            CapabilityCategory.DATA_ANALYSIS: "data analysis",
            CapabilityCategory.COMMUNICATION: "communication",
            CapabilityCategory.PLANNING: "planning",
            CapabilityCategory.DEBUGGING: "debugging",
            CapabilityCategory.DEPLOYMENT: "deployment",
            CapabilityCategory.SECURITY: "security analysis",
            CapabilityCategory.DESIGN: "design",
        }

        # Agent type friendly names
        self._agent_type_names = {
            "SENIOR_DEVELOPER": "Senior Developer",
            "JUNIOR_DEVELOPER": "Junior Developer",
            "TESTER": "QA Tester",
            "CODE_REVIEWER": "Code Reviewer",
            "DOCUMENTATION_WRITER": "Documentation Writer",
            "RESEARCHER": "Researcher",
            "DEBUGGER": "Debugger",
            "DESIGNER": "Designer",
            "DEVOPS": "DevOps Engineer",
            "EXECUTIVE_ASSISTANT": "Executive Assistant",
        }

    def translate(self, message: AgentMessage) -> HumanReadableMessage:
        """
        Translate an agent message to human-readable format.

        Args:
            message: The agent message to translate

        Returns:
            Human-readable version of the message
        """
        translator = self._translators.get(message.message_type)

        if translator:
            return translator(message)

        # Default translation
        return self._default_translate(message)

    def translate_task(self, task: Task) -> HumanReadableMessage:
        """Translate a task to human-readable format."""
        status_text = {
            TaskStatus.PENDING: "waiting to start",
            TaskStatus.QUEUED: "in queue",
            TaskStatus.ASSIGNED: "assigned to an agent",
            TaskStatus.IN_PROGRESS: "in progress",
            TaskStatus.WAITING_APPROVAL: "waiting for your approval",
            TaskStatus.COMPLETED: "completed",
            TaskStatus.FAILED: "failed",
            TaskStatus.CANCELLED: "cancelled",
        }

        capabilities = [
            self._capability_names.get(cap, cap.value)
            for cap in task.required_capabilities
        ]

        summary = f"Task '{task.title}' is {status_text.get(task.status, 'unknown')}"

        if task.assigned_to:
            summary += f" (assigned to {task.assigned_to.id})"

        details = f"Description: {task.description}\n"
        details += f"Required skills: {', '.join(capabilities)}\n"
        details += f"Priority: {task.priority.name}"

        if task.deadline:
            details += f"\nDeadline: {task.deadline.strftime('%Y-%m-%d %H:%M')}"

        action_required = task.status == TaskStatus.WAITING_APPROVAL

        return HumanReadableMessage(
            title=f"Task: {task.title}",
            summary=summary,
            details=details,
            action_required=action_required,
            suggested_actions=["Approve", "Reject", "View Details"] if action_required else [],
            severity="warning" if action_required else "info"
        )

    def translate_result(self, result: TaskResult) -> HumanReadableMessage:
        """Translate a task result to human-readable format."""
        if result.success:
            return HumanReadableMessage(
                title="Task Completed Successfully",
                summary=f"Task {result.task_id} finished successfully",
                details=self._format_output(result.output),
                severity="success"
            )
        else:
            return HumanReadableMessage(
                title="Task Failed",
                summary=f"Task {result.task_id} encountered an error",
                details=self._format_output(result.output),
                severity="error",
                suggested_actions=["Retry", "View Details", "Assign to Different Agent"]
            )

    def translate_state(self, agent_name: str, state: AgentState) -> HumanReadableMessage:
        """Translate agent state to human-readable format."""
        state_text = {
            AgentStateType.IDLE: "is available and ready for tasks",
            AgentStateType.INITIALIZING: "is starting up",
            AgentStateType.RUNNING: "is working on a task",
            AgentStateType.WAITING: "is waiting",
            AgentStateType.PAUSED: "is paused",
            AgentStateType.COMPLETED: "has finished its task",
            AgentStateType.FAILED: "encountered an error",
            AgentStateType.TERMINATED: "has been stopped",
        }

        summary = f"{agent_name} {state_text.get(state.state_type, 'is in an unknown state')}"

        if state.task_id and state.state_type == AgentStateType.RUNNING:
            summary += f" (Task: {state.task_id}, {int(state.progress * 100)}% complete)"

        if state.reason:
            summary += f": {state.reason}"

        severity = "info"
        if state.state_type == AgentStateType.FAILED:
            severity = "error"
        elif state.state_type == AgentStateType.COMPLETED:
            severity = "success"
        elif state.state_type == AgentStateType.PAUSED:
            severity = "warning"

        return HumanReadableMessage(
            title=f"Agent Status: {agent_name}",
            summary=summary,
            severity=severity
        )

    def summarize_activity(
        self,
        messages: List[AgentMessage],
        time_window_minutes: int = 60
    ) -> HumanReadableMessage:
        """Create a summary of recent agent activity."""
        if not messages:
            return HumanReadableMessage(
                title="Activity Summary",
                summary="No recent agent activity",
                severity="info"
            )

        # Count by type
        type_counts = {}
        agents_involved = set()

        for msg in messages:
            type_counts[msg.message_type] = type_counts.get(msg.message_type, 0) + 1
            agents_involved.add(str(msg.from_agent))
            if msg.to_agent:
                agents_involved.add(str(msg.to_agent))

        # Build summary
        summary_parts = []

        task_msgs = type_counts.get(MessageType.TASK_ASSIGNMENT, 0)
        if task_msgs:
            summary_parts.append(f"{task_msgs} tasks assigned")

        completed = type_counts.get(MessageType.TASK_RESULT, 0)
        if completed:
            summary_parts.append(f"{completed} tasks completed")

        errors = type_counts.get(MessageType.TASK_ERROR, 0)
        if errors:
            summary_parts.append(f"{errors} errors")

        approvals = type_counts.get(MessageType.APPROVAL_REQUEST, 0)
        if approvals:
            summary_parts.append(f"{approvals} approval requests pending")

        summary = f"In the last {time_window_minutes} minutes: " + ", ".join(summary_parts)
        summary += f". {len(agents_involved)} agents involved."

        return HumanReadableMessage(
            title="Agent Activity Summary",
            summary=summary,
            details=f"Message types: {', '.join(t.value for t in type_counts.keys())}",
            action_required=approvals > 0,
            severity="warning" if errors > 0 else "info"
        )

    # Individual message type translators

    def _translate_task_assignment(self, msg: AgentMessage) -> HumanReadableMessage:
        task_data = msg.payload.get('task', {})
        task_title = task_data.get('title', 'Unknown task')

        return HumanReadableMessage(
            title="New Task Assigned",
            summary=f"Task '{task_title}' has been assigned to an agent",
            details=task_data.get('description', ''),
            original_message_id=msg.id
        )

    def _translate_task_accepted(self, msg: AgentMessage) -> HumanReadableMessage:
        return HumanReadableMessage(
            title="Task Accepted",
            summary=f"Agent accepted the task and will begin working",
            severity="success",
            original_message_id=msg.id
        )

    def _translate_task_rejected(self, msg: AgentMessage) -> HumanReadableMessage:
        reason = msg.payload.get('reason', 'No reason provided')
        return HumanReadableMessage(
            title="Task Rejected",
            summary=f"Agent could not accept the task",
            details=f"Reason: {reason}",
            severity="warning",
            suggested_actions=["Reassign to another agent", "Modify task requirements"],
            original_message_id=msg.id
        )

    def _translate_progress_update(self, msg: AgentMessage) -> HumanReadableMessage:
        progress = msg.payload.get('progress', 0)
        status = msg.payload.get('status', 'Working...')

        return HumanReadableMessage(
            title="Progress Update",
            summary=f"Task is {int(progress * 100)}% complete: {status}",
            original_message_id=msg.id
        )

    def _translate_task_result(self, msg: AgentMessage) -> HumanReadableMessage:
        success = msg.payload.get('success', False)
        output = msg.payload.get('output', '')

        if success:
            return HumanReadableMessage(
                title="Task Completed",
                summary="The task finished successfully",
                details=self._format_output(output),
                severity="success",
                original_message_id=msg.id
            )
        else:
            return HumanReadableMessage(
                title="Task Failed",
                summary="The task could not be completed",
                details=self._format_output(output),
                severity="error",
                suggested_actions=["Review output", "Retry task", "Assign to different agent"],
                original_message_id=msg.id
            )

    def _translate_task_error(self, msg: AgentMessage) -> HumanReadableMessage:
        error = msg.payload.get('error', 'Unknown error')
        return HumanReadableMessage(
            title="Error Occurred",
            summary=f"An error occurred during task execution",
            details=error,
            severity="error",
            action_required=True,
            suggested_actions=["View error details", "Retry", "Cancel task"],
            original_message_id=msg.id
        )

    def _translate_approval_request(self, msg: AgentMessage) -> HumanReadableMessage:
        action = msg.payload.get('action', 'Unknown action')
        reason = msg.payload.get('reason', '')

        return HumanReadableMessage(
            title="Approval Required",
            summary=f"An agent is requesting permission to: {action}",
            details=reason,
            action_required=True,
            suggested_actions=["Approve", "Deny", "Request more info"],
            severity="warning",
            original_message_id=msg.id
        )

    def _translate_approval_response(self, msg: AgentMessage) -> HumanReadableMessage:
        approved = msg.payload.get('approved', False)
        return HumanReadableMessage(
            title="Approval Response",
            summary=f"Request was {'approved' if approved else 'denied'}",
            severity="success" if approved else "info",
            original_message_id=msg.id
        )

    def _translate_capability_query(self, msg: AgentMessage) -> HumanReadableMessage:
        capability = msg.payload.get('capability', 'unknown')
        return HumanReadableMessage(
            title="Capability Query",
            summary=f"Agent is looking for help with {capability}",
            original_message_id=msg.id
        )

    def _translate_capability_response(self, msg: AgentMessage) -> HumanReadableMessage:
        can_help = msg.payload.get('can_help', False)
        return HumanReadableMessage(
            title="Capability Response",
            summary=f"Agent {'can' if can_help else 'cannot'} help with this capability",
            original_message_id=msg.id
        )

    def _translate_handoff_request(self, msg: AgentMessage) -> HumanReadableMessage:
        reason = msg.payload.get('reason', 'No reason provided')
        return HumanReadableMessage(
            title="Task Handoff Request",
            summary="An agent is requesting to hand off their task to another agent",
            details=f"Reason: {reason}",
            action_required=True,
            suggested_actions=["Approve handoff", "Keep current assignment", "Cancel task"],
            severity="warning",
            original_message_id=msg.id
        )

    def _translate_heartbeat(self, msg: AgentMessage) -> HumanReadableMessage:
        return HumanReadableMessage(
            title="Agent Heartbeat",
            summary=f"Agent {msg.from_agent.id} is alive and responsive",
            original_message_id=msg.id
        )

    def _translate_status_request(self, msg: AgentMessage) -> HumanReadableMessage:
        return HumanReadableMessage(
            title="Status Request",
            summary="Status information was requested",
            original_message_id=msg.id
        )

    def _translate_status_response(self, msg: AgentMessage) -> HumanReadableMessage:
        status = msg.payload.get('status', 'Unknown')
        return HumanReadableMessage(
            title="Status Update",
            summary=f"Agent status: {status}",
            details=str(msg.payload),
            original_message_id=msg.id
        )

    def _translate_shutdown(self, msg: AgentMessage) -> HumanReadableMessage:
        return HumanReadableMessage(
            title="Agent Shutdown",
            summary=f"Agent {msg.from_agent.id} is shutting down",
            severity="warning",
            original_message_id=msg.id
        )

    def _default_translate(self, msg: AgentMessage) -> HumanReadableMessage:
        """Default translation for unknown message types."""
        return HumanReadableMessage(
            title=f"Agent Message: {msg.message_type.value}",
            summary=f"Message from {msg.from_agent.id}",
            details=str(msg.payload),
            original_message_id=msg.id
        )

    def _format_output(self, output: Any) -> str:
        """Format output for human readability."""
        if output is None:
            return "No output"

        if isinstance(output, str):
            # Truncate long strings
            if len(output) > 1000:
                return output[:1000] + "... (truncated)"
            return output

        if isinstance(output, dict):
            # Pretty format dict
            lines = []
            for k, v in output.items():
                lines.append(f"• {k}: {v}")
            return "\n".join(lines)

        if isinstance(output, list):
            if len(output) > 10:
                return f"{len(output)} items (showing first 10):\n" + \
                       "\n".join(f"• {item}" for item in output[:10])
            return "\n".join(f"• {item}" for item in output)

        return str(output)

    def get_friendly_agent_name(self, agent_type: str) -> str:
        """Get a friendly name for an agent type."""
        return self._agent_type_names.get(agent_type, agent_type.replace("_", " ").title())

    def get_friendly_capability_name(self, capability: CapabilityCategory) -> str:
        """Get a friendly name for a capability."""
        return self._capability_names.get(capability, capability.value.replace("_", " "))


# Global instance
_translation_layer: Optional[TranslationLayer] = None


def get_translation_layer() -> TranslationLayer:
    """Get or create the global translation layer."""
    global _translation_layer

    if _translation_layer is None:
        _translation_layer = TranslationLayer()

    return _translation_layer


def translate(message: AgentMessage) -> HumanReadableMessage:
    """Convenience function to translate a message."""
    return get_translation_layer().translate(message)
