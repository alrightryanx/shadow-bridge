"""
Universal TCP Client for ShadowBridge Plugin Communication
Supports length-prefixed JSON protocol for all platform plugins
"""
import socket
import json
import struct
import logging
from typing import Optional, Dict, Any
from threading import Lock

logger = logging.getLogger(__name__)

# Protocol constants
MAGIC_HEADER = b"SHADOW"
VERSION = 1
MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10MB
DEFAULT_PORT = 19286
DEFAULT_TIMEOUT = 30  # seconds


class ShadowBridgeTCPClient:
    """TCP client for communicating with ShadowBridge"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self._send_lock = Lock()
        
    def connect(self, timeout: int = DEFAULT_TIMEOUT) -> bool:
        """Connect to ShadowBridge with handshake"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(timeout)
            self.sock.connect((self.host, self.port))
            
            # Send handshake
            handshake = {"type": "handshake", "deviceId": None}
            self._send_message(handshake)
            
            # Wait for handshake ack
            ack = self._receive_message(timeout=5)
            if ack and ack.get("type") == "handshake_ack":
                logger.info(f"Connected to ShadowBridge at {self.host}:{self.port}")
                return True
            else:
                logger.warning(f"Unexpected handshake response: {ack}")
                return False
                
        except socket.timeout:
            logger.error(f"Connection timeout to {self.host}:{self.port}")
            return False
        except Exception as e:
            logger.error(f"Connection failed to {self.host}:{self.port}: {e}")
            return False
    
    def send(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send message and return response"""
        if not self.sock:
            logger.error("Not connected to ShadowBridge")
            return None
            
        try:
            self._send_message(message)
            response = self._receive_message(timeout=60)
            return response
        except Exception as e:
            logger.error(f"Send/Receive failed: {e}")
            return None
    
    def send_fire_and_forget(self, message: Dict[str, Any]) -> None:
        """Send message without waiting for response (for fire-and-forget operations)"""
        if not self.sock:
            logger.warning("Not connected, cannot send fire-and-forget message")
            return
            
        try:
            self._send_message(message)
            logger.debug(f"Sent fire-and-forget message: {message.get('type')}")
        except Exception as e:
            logger.error(f"Fire-and-forget failed: {e}")
    
    def _send_message(self, message: Dict[str, Any]) -> None:
        """Send length-prefixed JSON message"""
        with self._send_lock:
            try:
                json_bytes = json.dumps(message, ensure_ascii=False).encode('utf-8')
                length_prefix = struct.pack('>I', len(json_bytes))
                
                self.sock.sendall(MAGIC_HEADER + length_prefix + json_bytes)
                logger.debug(f"Sent message type={message.get('type')}, length={len(json_bytes)}")
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
                raise
    
    def _receive_message(self, timeout: int = 60) -> Optional[Dict[str, Any]]:
        """Receive length-prefixed JSON message"""
        if not self.sock:
            return None
            
        try:
            # Read magic header
            magic = self._recv_exact(len(MAGIC_HEADER), timeout)
            if magic != MAGIC_HEADER:
                logger.warning(f"Invalid magic header: {magic}")
                return None
            
            # Read length prefix (4 bytes, big-endian)
            length_bytes = self._recv_exact(4, timeout)
            if not length_bytes:
                return None
                
            message_length = struct.unpack('>I', length_bytes)[0]
            
            if message_length <= 0 or message_length > MAX_MESSAGE_SIZE:
                logger.warning(f"Invalid message length: {message_length}")
                return None
            
            # Read message body
            message_bytes = self._recv_exact(message_length, timeout)
            if not message_bytes:
                return None
                
            message_str = message_bytes.decode('utf-8')
            message = json.loads(message_str)
            
            logger.debug(f"Received message type={message.get('type')}, length={message_length}")
            return message
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return None
        except socket.timeout:
            logger.warning(f"Receive timeout after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"Receive failed: {e}")
            return None
    
    def _recv_exact(self, size: int, timeout: int) -> Optional[bytes]:
        """Receive exactly size bytes or timeout"""
        if not self.sock:
            return None
            
        self.sock.settimeout(timeout)
        data = b''
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def disconnect(self) -> None:
        """Close connection to ShadowBridge"""
        if self.sock:
            try:
                self.sock.close()
                logger.info(f"Disconnected from ShadowBridge at {self.host}:{self.port}")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
            finally:
                self.sock = None


def create_client(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> ShadowBridgeTCPClient:
    """Factory function to create TCP client with default config"""
    return ShadowBridgeTCPClient(host, port)


if __name__ == "__main__":
    # Test connection
    import sys
    logging.basicConfig(level=logging.DEBUG)
    
    client = create_client()
    if client.connect(timeout=5):
        # Send test message
        client.send_fire_and_forget({"type": "test", "message": "Hello from plugin"})
        client.disconnect()
    else:
        print("Failed to connect")
        sys.exit(1)
