"""
Universal Session Protocol for ShadowAI Plugin Communication
Defines message formats and schemas for cross-platform session persistence
"""
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime
import json


class MessageType(Enum):
    """Standard message types for plugin communication"""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SESSION_MESSAGE = "session_message"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_RESPONSE = "approval_response"
    APPROVAL_DISMISS = "approval_dismiss"
    NOTIFICATION = "notification"
    HANDSHAKE = "handshake"
    HANDSHAKE_ACK = "handshake_ack"


class MessageRole(Enum):
    """Role in session messages"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


class ApprovalStatus(Enum):
    """Approval decision status"""
    APPROVED = True
    DENIED = False


class SessionMessage:
    """Universal session message structure"""
    
    def __init__(
        self,
        message_type: str,
        message_id: str,
        session_id: str,
        timestamp: int,
        payload: Dict[str, Any],
        device_id: Optional[str] = None
    ):
        self.type = message_type
        self.id = message_id
        self.session_id = session_id
        self.timestamp = timestamp
        self.payload = payload
        self.device_id = device_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "type": self.type.value if isinstance(self.type, Enum) else self.type,
            "id": self.id,
            "sessionId": self.session_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "deviceId": self.device_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionMessage':
        """Create from dictionary"""
        return cls(
            message_type=data.get("type", ""),
            message_id=data.get("id", ""),
            session_id=data.get("sessionId", data.get("session_id", "")),
            timestamp=data.get("timestamp", int(datetime.now().timestamp() * 1000)),
            payload=data.get("payload", {}),
            device_id=data.get("deviceId", data.get("device_id"))
        )


class SessionStartPayload:
    """Payload for session_start messages"""
    
    def __init__(
        self,
        hostname: str,
        cwd: str,
        transcript_path: Optional[str] = None,
        username: str = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        project_id: Optional[str] = None
    ):
        self.hostname = hostname
        self.cwd = cwd
        self.transcript_path = transcript_path
        self.username = username
        self.provider = provider
        self.model = model
        self.project_id = project_id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hostname": self.hostname,
            "cwd": self.cwd,
            "transcriptPath": self.transcript_path,
            "username": self.username,
            "provider": self.provider,
            "model": self.model,
            "projectId": self.project_id
        }


class SessionEndPayload:
    """Payload for session_end messages"""
    
    def __init__(
        self,
        hostname: str = None
    ):
        self.hostname = hostname
    
    def to_dict(self) -> Dict[str, Any]:
        data = {}
        if self.hostname:
            data["hostname"] = self.hostname
        return data


class SessionMessagePayload:
    """Payload for session_message (chat sync)"""
    
    def __init__(
        self,
        role: str,
        content: str,
        hostname: str = None,
        cwd: str = None,
        is_history: bool = False,
        tool_name: Optional[str] = None,
        tool_input: Optional[Dict[str, Any]] = None
    ):
        self.role = role
        self.content = content
        self.hostname = hostname
        self.cwd = cwd
        self.is_history = is_history
        self.tool_name = tool_name
        self.tool_input = tool_input
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "role": self.role,
            "content": self.content,
            "isHistory": self.is_history
        }
        if self.hostname:
            data["hostname"] = self.hostname
        if self.cwd:
            data["cwd"] = self.cwd
        if self.tool_name:
            data["toolName"] = self.tool_name
        if self.tool_input:
            data["toolInput"] = self.tool_input
        return data


class ApprovalRequestPayload:
    """Payload for approval_request messages"""
    
    def __init__(
        self,
        approval_id: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        prompt: str,
        prompt_type: str = "PERMISSION",
        options: List[str] = None,
        default_response: Optional[str] = None,
        context: Optional[str] = None,
        cwd: Optional[str] = None,
        tool_use_id: Optional[str] = None,
        allow_reply: bool = True
    ):
        self.approval_id = approval_id
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.prompt = prompt
        self.prompt_type = prompt_type
        self.options = options or ["Approve", "Deny", "Reply"]
        self.default_response = default_response
        self.context = context
        self.cwd = cwd
        self.tool_use_id = tool_use_id
        self.allow_reply = allow_reply
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "approvalId": self.approval_id,
            "toolName": self.tool_name,
            "toolInput": self.tool_input,
            "prompt": self.prompt,
            "promptType": self.prompt_type,
            "allowReply": self.allow_reply,
            "options": self.options
        }
        if self.default_response:
            data["defaultResponse"] = self.default_response
        if self.context:
            data["context"] = self.context
        if self.cwd:
            data["cwd"] = self.cwd
        if self.tool_use_id:
            data["toolUseId"] = self.tool_use_id
        return data


class NotificationPayload:
    """Payload for notification messages"""
    
    def __init__(
        self,
        message: str,
        level: str = "info",
        title: Optional[str] = None
    ):
        self.message = message
        self.level = level
        self.title = title
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "message": self.message,
            "level": self.level
        }
        if self.title:
            data["title"] = self.title
        return data


def generate_message_id() -> str:
    """Generate unique message ID"""
    import uuid
    return f"msg_{uuid.uuid4().hex[:8]}"


def get_friendly_tool_description(
    tool_name: str,
    tool_input: Optional[Dict[str, Any]] = None
) -> str:
    """Generate user-friendly description of tool usage"""
    
    if not tool_input:
        return f"{tool_name}: No input"
    
    # Extract key information based on tool type
    file_path = tool_input.get("file_path", tool_input.get("filePath", ""))
    command = tool_input.get("command", tool_input.get("cmd", ""))
    pattern = tool_input.get("pattern", tool_input.get("regex", ""))
    
    if tool_name.lower() in ["bash", "run_shell_command"]:
        # Show command summary
        if command and len(command) > 80:
            cmd_preview = command[:80] + "..."
            return f"Run: {cmd_preview}"
        return f"Run: {command}"
    
    elif tool_name.lower() in ["edit", "replace"]:
        # Show file being edited
        short_path = "/".join(file_path.split("/")[-2:])
        return f"Edit: {short_path}"
    
    elif tool_name.lower() in ["write", "write_file"]:
        # Show file being created
        short_path = "/".join(file_path.split("/")[-2:])
        return f"Create: {short_path}"
    
    elif tool_name.lower() in ["read", "read_f
