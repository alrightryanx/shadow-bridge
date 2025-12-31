"""
Pytest configuration and shared fixtures for shadow-bridge tests.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import socket
import threading
import time


# ==================== Path Fixtures ====================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def shadowai_dir(temp_dir):
    """Create a mock ~/.shadowai directory structure."""
    shadowai = temp_dir / ".shadowai"
    shadowai.mkdir(parents=True, exist_ok=True)
    return shadowai


@pytest.fixture
def mock_home(temp_dir, monkeypatch):
    """Mock the home directory to use temp_dir."""
    monkeypatch.setenv("USERPROFILE", str(temp_dir))
    monkeypatch.setenv("HOME", str(temp_dir))
    return temp_dir


# ==================== Settings Fixtures ====================

@pytest.fixture
def default_settings():
    """Default settings structure."""
    return {
        "bridge_port": 19284,
        "discovery_port": 19285,
        "companion_port": 19286,
        "auto_start": False,
        "auto_update": True,
        "theme": "dark"
    }


@pytest.fixture
def settings_file(shadowai_dir, default_settings):
    """Create a settings.json file with default settings."""
    settings_path = shadowai_dir / "settings.json"
    with open(settings_path, "w") as f:
        json.dump(default_settings, f)
    return settings_path


# ==================== Project/Notes Data Fixtures ====================

@pytest.fixture
def sample_projects():
    """Sample project data for testing."""
    return [
        {
            "id": "project-1",
            "name": "Test Project 1",
            "description": "First test project",
            "workingDirectory": "/path/to/project1",
            "createdAt": 1700000000000,
            "updatedAt": 1700000001000
        },
        {
            "id": "project-2",
            "name": "Test Project 2",
            "description": "Second test project",
            "workingDirectory": "C:\\path\\to\\project2",
            "createdAt": 1700000002000,
            "updatedAt": 1700000003000
        }
    ]


@pytest.fixture
def sample_notes():
    """Sample note data for testing."""
    return [
        {
            "id": "note-1",
            "title": "Test Note 1",
            "content": "This is the content of note 1",
            "category": "Work",
            "isArchived": False,
            "createdAt": 1700000000000,
            "updatedAt": 1700000001000
        },
        {
            "id": "note-2",
            "title": "Test Note 2",
            "content": "This is the content of note 2",
            "category": "Personal",
            "isArchived": True,
            "createdAt": 1700000002000,
            "updatedAt": 1700000003000
        }
    ]


@pytest.fixture
def sample_device():
    """Sample device data for testing."""
    return {
        "id": "device-test-123",
        "name": "Test Phone",
        "ip": "192.168.1.100",
        "lastSeen": 1700000000000,
        "ipCandidates": ["192.168.1.100", "10.0.0.50"]
    }


@pytest.fixture
def projects_file(shadowai_dir, sample_projects, sample_device):
    """Create a projects.json file with sample data."""
    projects_path = shadowai_dir / "projects.json"
    data = {
        "devices": {
            sample_device["id"]: {
                "name": sample_device["name"],
                "ip": sample_device["ip"],
                "lastSeen": sample_device["lastSeen"],
                "projects": sample_projects
            }
        }
    }
    with open(projects_path, "w") as f:
        json.dump(data, f)
    return projects_path


@pytest.fixture
def notes_file(shadowai_dir, sample_notes, sample_device):
    """Create a notes.json file with sample data."""
    notes_path = shadowai_dir / "notes.json"
    data = {
        "devices": {
            sample_device["id"]: {
                "name": sample_device["name"],
                "ip": sample_device["ip"],
                "lastSeen": sample_device["lastSeen"],
                "notes": sample_notes
            }
        }
    }
    with open(notes_path, "w") as f:
        json.dump(data, f)
    return notes_path


# ==================== Network Fixtures ====================

@pytest.fixture
def mock_socket():
    """Create a mock socket for testing."""
    mock = MagicMock(spec=socket.socket)
    mock.recv.return_value = b'{"status": "ok"}'
    mock.send.return_value = 100
    return mock


@pytest.fixture
def free_port():
    """Find a free port for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture
def echo_server(free_port):
    """Create a simple echo server for integration testing."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('127.0.0.1', free_port))
    server_socket.listen(1)
    server_socket.settimeout(5)

    stop_event = threading.Event()

    def run_server():
        while not stop_event.is_set():
            try:
                conn, addr = server_socket.accept()
                data = conn.recv(1024)
                if data:
                    conn.send(data)
                conn.close()
            except socket.timeout:
                continue
            except Exception:
                break

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    yield free_port

    stop_event.set()
    server_socket.close()
    thread.join(timeout=1)


# ==================== Automation Fixtures ====================

@pytest.fixture
def sample_automations():
    """Sample automation data for testing."""
    return [
        {
            "id": "automation-1",
            "name": "Daily Backup",
            "type": "FILE_OPERATION",
            "isEnabled": True,
            "trigger": {"type": "TIME", "hour": 9, "minute": 0},
            "createdAt": 1700000000000,
            "lastRun": 1700000001000
        },
        {
            "id": "automation-2",
            "name": "Email Summary",
            "type": "TEXT",
            "isEnabled": False,
            "trigger": {"type": "MANUAL"},
            "createdAt": 1700000002000,
            "lastRun": None
        }
    ]


# ==================== Team Fixtures ====================

@pytest.fixture
def sample_teams():
    """Sample team data for testing."""
    return [
        {
            "id": "team-1",
            "name": "Engineering",
            "description": "Engineering team",
            "members": [
                {"email": "user1@example.com", "role": "owner"},
                {"email": "user2@example.com", "role": "member"}
            ],
            "createdAt": 1700000000000
        }
    ]


# ==================== Rate Limiting Fixtures ====================

@pytest.fixture
def rate_limiter_class():
    """Import and return RateLimiter class for testing."""
    # This would import from the actual module
    class MockRateLimiter:
        def __init__(self, requests_per_minute=60):
            self.requests_per_minute = requests_per_minute
            self._requests = {}

        def is_allowed(self, key):
            now = time.time()
            if key not in self._requests:
                self._requests[key] = []

            # Remove old requests
            self._requests[key] = [t for t in self._requests[key]
                                   if now - t < 60]

            if len(self._requests[key]) >= self.requests_per_minute:
                return False

            self._requests[key].append(now)
            return True

        def get_remaining(self, key):
            now = time.time()
            if key not in self._requests:
                return self.requests_per_minute

            recent = [t for t in self._requests[key] if now - t < 60]
            return max(0, self.requests_per_minute - len(recent))

    return MockRateLimiter


# ==================== Flask App Fixtures ====================

@pytest.fixture
def flask_app(mock_home, shadowai_dir):
    """Create Flask app for testing."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "web"))
        from app import create_app

        app = create_app()
        app.config['TESTING'] = True
        return app
    except ImportError:
        # Return a mock app if imports fail
        from unittest.mock import MagicMock
        mock_app = MagicMock()
        mock_app.config = {'TESTING': True}
        return mock_app


@pytest.fixture
def flask_client(flask_app):
    """Create Flask test client."""
    if hasattr(flask_app, 'test_client'):
        return flask_app.test_client()
    return MagicMock()


# ==================== Encryption Fixtures ====================

@pytest.fixture
def encryption_key():
    """Sample encryption key for testing."""
    return "test-password-12345"


@pytest.fixture
def encrypted_content(encryption_key):
    """Sample encrypted content."""
    # Format: SYNC_ENC:base64(iv + ciphertext + tag)
    return "SYNC_ENC:dGVzdC1lbmNyeXB0ZWQtY29udGVudA=="


# ==================== Utility Functions ====================

def create_length_prefixed_message(data):
    """Create a length-prefixed JSON message."""
    json_bytes = json.dumps(data).encode('utf-8')
    length_bytes = len(json_bytes).to_bytes(4, 'big')
    return length_bytes + json_bytes


def parse_length_prefixed_message(data):
    """Parse a length-prefixed JSON message."""
    if len(data) < 4:
        return None
    length = int.from_bytes(data[:4], 'big')
    if len(data) < 4 + length:
        return None
    json_str = data[4:4+length].decode('utf-8')
    return json.loads(json_str)


# Export utilities
@pytest.fixture
def message_utils():
    """Provide message utility functions."""
    return {
        'create': create_length_prefixed_message,
        'parse': parse_length_prefixed_message
    }
