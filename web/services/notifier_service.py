import socket
import json
import logging
import subprocess
import time
from typing import Dict, Any, Optional

# Constants matching bridge config
BRIDGE_PORT = 19286
# Path to shadow-android/scripts/sentinel_quality.py (assumed next to shadow-bridge)
import os
AEGIS_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "sentinel_quality.py")

logger = logging.getLogger(__name__)

class NotifierService:
    """
    Handles notifications and quality sentinel alerts for ShadowBridge.
    Connects to the companion relay port to send messages to the Android app.
    """
    
    def __init__(self):
        self.last_notify_time = 0
        self.notification_interval = 2.0 # Rate limit notifications

    def notify_quality_issue(self, text: str, content_type: str, session_id: str, provider: str = "shadow") -> bool:
        """
        Runs the quality sentinel script and notifies if issues are found.
        content_type: 'prompt' or 'response'
        """
        if not os.path.exists(AEGIS_SCRIPT):
            return True

        try:
            # Run sentinel script
            cmd = ["python", AEGIS_SCRIPT, "--type", content_type, "--text", text]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                score = data.get("score", 100)
                reason = data.get("reason", "Unknown")
                
                if content_type == "prompt" and score < 60:
                    # Low quality prompt - notify user on phone
                    self.send_bridge_notification(
                        session_id=session_id,
                        message=f"🚩 LOW QUALITY PROMPT: {reason}. Suggesting improvement...",
                        summary="ShadowRefiner: Alert",
                        notif_type="warning",
                        provider=provider
                    )
                    
                    # Get improved prompt
                    improve_cmd = ["python", AEGIS_SCRIPT, "--type", content_type, "--mode", "improve", "--text", text]
                    imp_res = subprocess.run(improve_cmd, capture_output=True, text=True, encoding='utf-8')
                    if imp_res.returncode == 0:
                        improved_text = imp_res.stdout.strip()
                        self.send_bridge_notification(
                            session_id=session_id,
                            message=f"✨ AUTO-IMPROVED: I refined your prompt for better results.",
                            summary="ShadowRefiner: Improved",
                            notif_type="info",
                            provider=provider
                        )
                
                elif content_type == "response" and score < 50:
                    # Poor response - offer auto-retry
                    self.send_bridge_notification(
                        session_id=session_id,
                        message=f"⚠️ POOR RESPONSE: {reason}. Score: {score}/100.",
                        summary="ShadowRefiner: Quality Audit",
                        notif_type="error",
                        provider=provider
                    )
                    
                    # Auto-Regenerate for very poor responses (< 30)
                    if score < 30:
                        logger.info(f"ShadowRefiner: Triggering auto-regenerate for session {session_id}")
                        self._trigger_regenerate(session_id)
            
            return True
        except Exception as e:
            logger.error(f"Error in notify_quality_issue: {e}")
            return True

    def _trigger_regenerate(self, session_id: str):
        """Sends a regenerate command to the active session."""
        regen_msg = {
            "type": "user_input",
            "sessionId": session_id,
            "payload": {
                "text": "/regenerate", 
                "action": "terminal_inject",
                "autoSubmit": True
            }
        }
        self._send_to_bridge(regen_msg)

    def send_bridge_notification(self, session_id: str, message: str, summary: str, notif_type: str = "info", provider: str = "shadow"):
        """
        Sends a notification message to the ShadowAI Bridge for mobile relay.
        """
        import time
        import uuid

        notif_msg = {
            "type": "notification",
            "id": f"msg_{uuid.uuid4().hex[:8]}",
            "sessionId": session_id,
            "timestamp": int(time.time() * 1000),
            "payload": {
                "notificationId": f"notif_{int(time.time())}",
                "message": message,
                "summary": summary,
                "notificationType": notif_type,
                "hostname": socket.gethostname(),
                "provider": provider,
                "options": ["Dismiss"],
                "promptType": "NOTIFICATION"
            }
        }
        
        # Add "Regenerate" for errors
        if notif_type == "error":
            notif_msg["payload"]["options"] = ["Regenerate", "Dismiss"]

        self._send_to_bridge(notif_msg)

    def _send_to_bridge(self, message: Dict[str, Any]):
        """Internal helper to send raw message to companion port."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2.0)
                s.connect(("127.0.0.1", BRIDGE_PORT))
                
                # Handshake
                handshake = json.dumps({"type": "handshake"})
                s.sendall(len(handshake).to_bytes(4, byteorder='big'))
                s.sendall(handshake.encode('utf-8'))
                
                # Read ack
                ack_len_buf = s.recv(4)
                if not ack_len_buf: return
                
                # Send actual message
                msg_data = json.dumps(message).encode('utf-8')
                s.sendall(len(msg_data).to_bytes(4, byteorder='big'))
                s.sendall(msg_data)
        except Exception as e:
            logger.debug(f"Failed to send to bridge: {e}")

_instance = None
def get_notifier():
    global _instance
    if _instance is None:
        _instance = NotifierService()
    return _instance