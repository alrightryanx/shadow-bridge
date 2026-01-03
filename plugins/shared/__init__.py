"""
Shared utilities for ShadowAI plugin development
"""
from .tcp_client import ShadowBridgeTCPClient, create_client
from .session_protocol import (
    MessageType, MessageRole, ApprovalStatus,
    SessionMessage, SessionStartPayload, SessionEndPayload,
    SessionMessagePayload, ApprovalRequestPayload, NotificationPayload,
    generate_message_id, get_friendly_tool_description
)

__all__ = [
    'ShadowBridgeTCPClient',
    'create_client',
    'MessageType',
    'MessageRole',
    'ApprovalStatus',
    'SessionMessage',
    'SessionStartPayload',
    'SessionEndPayload',
    'SessionMessagePayload',
    'ApprovalRequestPayload',
    'NotificationPayload',
    'generate_message_id',
    'get_friendly_tool_description'
]
