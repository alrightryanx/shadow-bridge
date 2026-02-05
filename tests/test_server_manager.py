"""
Tests for WebServerManager.
"""

import pytest
import socket
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shadow_bridge.web.server_manager import WebServerManager


class TestWebServerManager:
    """Tests for WebServerManager class."""
    
    @pytest.fixture
    def free_port(self):
        """Find a free port for testing."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            return s.getsockname()[1]
    
    def test_port_availability_check(self, free_port):
        """Test port availability detection."""
        manager = WebServerManager(host="127.0.0.1", port=free_port)
        assert manager.is_port_available() is True
        
        # Occupy the port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('127.0.0.1', free_port))
            s.listen(1)
            
            assert manager.is_port_available() is False
    
    def test_find_available_port(self, free_port):
        """Test finding alternative port."""
        manager = WebServerManager(host="127.0.0.1", port=free_port)
        
        # Occupy the preferred port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('127.0.0.1', free_port))
            s.listen(1)
            
            # Should find alternative
            alt = manager.find_available_port(free_port)
            assert alt is not None
            assert alt != free_port
    
    def test_start_stop_minimal_server(self, free_port):
        """Test starting and stopping minimal server."""
        manager = WebServerManager(host="127.0.0.1", port=free_port)
        
        try:
            # Start without Flask app (uses minimal server)
            result = manager.start()
            assert result is True
            assert manager.is_running is True
            
            # Give server time to start
            time.sleep(0.5)
            
            # Health check
            health = manager.health_check()
            assert health["status"] == "ok"
            assert health["port"] == free_port
            
            # Stop
            result = manager.stop()
            assert result is True
            assert manager.is_running is False
            
        finally:
            manager.stop()
    
    def test_double_start_is_idempotent(self, free_port):
        """Starting twice should be safe."""
        manager = WebServerManager(host="127.0.0.1", port=free_port)
        
        try:
            manager.start()
            result = manager.start()  # Second start
            assert result is True  # Should return True (already running)
        finally:
            manager.stop()
    
    def test_health_check_when_stopped(self, free_port):
        """Health check should work when server is stopped."""
        manager = WebServerManager(host="127.0.0.1", port=free_port)
        
        health = manager.health_check()
        assert health["status"] == "stopped"
        assert health["uptime_seconds"] == 0
    
    def test_uptime_tracking(self, free_port):
        """Uptime should be tracked correctly."""
        manager = WebServerManager(host="127.0.0.1", port=free_port)
        
        try:
            manager.start()
            time.sleep(1)
            
            assert manager.uptime_seconds >= 1.0
            
            manager.stop()
            assert manager.uptime_seconds == 0
            
        finally:
            manager.stop()
    
    def test_port_conflict_with_alt_port(self, free_port):
        """Should find alternative port when preferred is busy."""
        # Occupy the preferred port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('127.0.0.1', free_port))
            s.listen(1)
            
            manager = WebServerManager(host="127.0.0.1", port=free_port)
            
            try:
                # Should fail without alt port
                result = manager.start(use_alt_port=False)
                assert result is False
                
                # Should succeed with alt port
                result = manager.start(use_alt_port=True)
                assert result is True
                assert manager.port != free_port
                
            finally:
                manager.stop()
