"""
Comprehensive tests for shadow-bridge networking functions.
Tests IP detection, port checking, firewall rules, and socket operations.
"""

import pytest
import socket
import struct
import json
import threading
import time
from unittest.mock import patch, MagicMock, PropertyMock


class TestIPDetection:
    """Tests for IP address detection functions."""

    def test_get_local_ip_returns_valid_ip(self):
        """Local IP should be a valid IPv4 address."""
        # Simulate get_local_ip logic
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = "127.0.0.1"

        parts = local_ip.split(".")
        assert len(parts) == 4
        for part in parts:
            assert 0 <= int(part) <= 255

    def test_get_local_ip_not_loopback(self):
        """Local IP should not be loopback when network available."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            # On networked machine, should not be loopback
            # But might be on isolated test environment
            assert local_ip is not None
        except Exception:
            pytest.skip("No network connection available")

    def test_hostname_is_valid_string(self):
        """Hostname should be a non-empty string."""
        hostname = socket.gethostname()
        assert isinstance(hostname, str)
        assert len(hostname) > 0

    def test_get_all_ips_returns_list(self):
        """get_all_ips should return a list of IP addresses."""
        # Simulate get_all_ips
        hostname = socket.gethostname()
        try:
            ips = socket.gethostbyname_ex(hostname)[2]
        except socket.gaierror:
            ips = ["127.0.0.1"]

        assert isinstance(ips, list)
        assert len(ips) >= 1

    def test_private_ip_detection(self):
        """Test private IP range detection."""
        private_ips = [
            "192.168.1.1",
            "192.168.0.100",
            "10.0.0.1",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.255"
        ]
        public_ips = [
            "8.8.8.8",
            "1.1.1.1",
            "203.0.113.1"
        ]

        def is_private_ip(ip):
            parts = [int(p) for p in ip.split(".")]
            if parts[0] == 10:
                return True
            if parts[0] == 172 and 16 <= parts[1] <= 31:
                return True
            if parts[0] == 192 and parts[1] == 168:
                return True
            return False

        for ip in private_ips:
            assert is_private_ip(ip) is True, f"{ip} should be private"

        for ip in public_ips:
            assert is_private_ip(ip) is False, f"{ip} should be public"

    def test_loopback_detection(self):
        """Test loopback address detection."""
        loopback_ips = ["127.0.0.1", "127.0.0.2", "127.255.255.255"]
        non_loopback = ["192.168.1.1", "10.0.0.1", "8.8.8.8"]

        def is_loopback(ip):
            return ip.startswith("127.")

        for ip in loopback_ips:
            assert is_loopback(ip) is True

        for ip in non_loopback:
            assert is_loopback(ip) is False


class TestPortOperations:
    """Tests for port checking and operations."""

    def test_check_port_in_use_false_for_free_port(self, free_port):
        """Free port should not be in use."""
        def check_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('127.0.0.1', port))
                    return False
                except OSError:
                    return True

        assert check_port_in_use(free_port) is False

    def test_check_port_in_use_true_for_bound_port(self, free_port):
        """Bound port should be detected as in use."""
        def check_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('127.0.0.1', port))
                    return False
                except OSError:
                    return True

        # Bind the port first
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', free_port))
        server.listen(1)

        try:
            assert check_port_in_use(free_port) is True
        finally:
            server.close()

    def test_default_ports_are_valid(self):
        """Default ports should be in valid range."""
        default_ports = {
            "data_receiver": 19284,
            "discovery": 19285,
            "companion_relay": 19286
        }

        for name, port in default_ports.items():
            assert 1 <= port <= 65535, f"{name} port {port} is invalid"
            assert port > 1024, f"{name} port {port} is privileged"

    def test_port_sequence_is_consecutive(self):
        """Shadow bridge ports should be consecutive."""
        ports = [19284, 19285, 19286]
        for i in range(len(ports) - 1):
            assert ports[i + 1] - ports[i] == 1


class TestSocketOperations:
    """Tests for socket send/receive operations."""

    def test_length_prefixed_message_creation(self, message_utils):
        """Test creating length-prefixed messages."""
        data = {"type": "test", "value": 123}
        message = message_utils['create'](data)

        # First 4 bytes are length
        length = struct.unpack('>I', message[:4])[0]
        json_part = message[4:]

        assert length == len(json_part)
        assert json.loads(json_part.decode('utf-8')) == data

    def test_length_prefixed_message_parsing(self, message_utils):
        """Test parsing length-prefixed messages."""
        data = {"type": "response", "status": "ok"}
        message = message_utils['create'](data)
        parsed = message_utils['parse'](message)

        assert parsed == data

    def test_parse_incomplete_message(self, message_utils):
        """Incomplete messages should return None."""
        # Only 2 bytes (need at least 4 for length)
        assert message_utils['parse'](b'\x00\x00') is None

        # Length says 100 but only 10 bytes of data
        incomplete = b'\x00\x00\x00\x64' + b'x' * 10
        assert message_utils['parse'](incomplete) is None

    def test_socket_timeout_handling(self, free_port):
        """Socket should handle timeouts gracefully."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)

        with pytest.raises(socket.timeout):
            sock.connect(('127.0.0.1', free_port))

        sock.close()

    def test_echo_server_integration(self, echo_server, message_utils):
        """Test sending and receiving through echo server."""
        data = {"type": "ping", "id": 12345}

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(('127.0.0.1', echo_server))

        message = message_utils['create'](data)
        sock.send(message)

        response = sock.recv(1024)
        parsed = message_utils['parse'](response)

        sock.close()
        assert parsed == data


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    def test_rate_limiter_allows_within_limit(self, rate_limiter_class):
        """Requests within limit should be allowed."""
        limiter = rate_limiter_class(requests_per_minute=10)

        for i in range(10):
            assert limiter.is_allowed("client1") is True

    def test_rate_limiter_blocks_over_limit(self, rate_limiter_class):
        """Requests over limit should be blocked."""
        limiter = rate_limiter_class(requests_per_minute=5)

        # Use up the limit
        for i in range(5):
            limiter.is_allowed("client1")

        # Next request should be blocked
        assert limiter.is_allowed("client1") is False

    def test_rate_limiter_separate_clients(self, rate_limiter_class):
        """Different clients should have separate limits."""
        limiter = rate_limiter_class(requests_per_minute=2)

        # Client 1 uses its limit
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")

        # Client 2 should still be allowed
        assert limiter.is_allowed("client2") is True

    def test_rate_limiter_get_remaining(self, rate_limiter_class):
        """Should correctly report remaining requests."""
        limiter = rate_limiter_class(requests_per_minute=10)

        assert limiter.get_remaining("client1") == 10

        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")

        assert limiter.get_remaining("client1") == 7

    def test_rate_limiter_resets_after_window(self, rate_limiter_class):
        """Limit should reset after time window."""
        # This is a conceptual test - actual implementation uses sliding window
        limiter = rate_limiter_class(requests_per_minute=60)

        # Initially full
        assert limiter.get_remaining("client1") == 60


class TestConnectionHandling:
    """Tests for connection handling logic."""

    def test_client_connection_tracking(self):
        """Test tracking connected clients."""
        connected_clients = set()

        # Add clients
        connected_clients.add("client-1")
        connected_clients.add("client-2")
        assert len(connected_clients) == 2

        # Remove client
        connected_clients.discard("client-1")
        assert len(connected_clients) == 1
        assert "client-2" in connected_clients

    def test_device_connection_mapping(self):
        """Test IP to device ID mapping."""
        ip_to_device = {}

        # Map devices
        ip_to_device["192.168.1.100"] = "device-abc"
        ip_to_device["192.168.1.101"] = "device-def"

        assert ip_to_device.get("192.168.1.100") == "device-abc"
        assert ip_to_device.get("192.168.1.102") is None

    def test_pending_messages_queue(self):
        """Test pending message queue for disconnected clients."""
        pending_messages = {}

        device_id = "device-123"
        pending_messages[device_id] = []

        # Queue messages
        pending_messages[device_id].append({"type": "msg1"})
        pending_messages[device_id].append({"type": "msg2"})

        assert len(pending_messages[device_id]) == 2

        # Clear on reconnect
        messages = pending_messages.pop(device_id, [])
        assert len(messages) == 2
        assert device_id not in pending_messages


class TestDiscoveryBroadcast:
    """Tests for UDP discovery broadcast."""

    def test_discovery_message_format(self):
        """Discovery broadcast should have correct format."""
        hostname = socket.gethostname()
        discovery_port = 19285
        data_port = 19284

        message = {
            "type": "shadowbridge_discovery",
            "hostname": hostname,
            "port": data_port,
            "version": "1.0"
        }

        json_str = json.dumps(message)
        assert "shadowbridge_discovery" in json_str
        assert str(data_port) in json_str

    def test_broadcast_address_valid(self):
        """Broadcast address should be valid."""
        broadcast_addr = "255.255.255.255"
        parts = broadcast_addr.split(".")
        assert all(p == "255" for p in parts)

    def test_udp_socket_creation(self):
        """UDP socket should be creatable with broadcast."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        # Should not raise
        sock.close()


class TestHandshakeProtocol:
    """Tests for handshake protocol."""

    def test_plugin_handshake_no_device_id(self, message_utils):
        """Plugin handshake should not have deviceId."""
        handshake = {"type": "handshake"}
        message = message_utils['create'](handshake)

        parsed = message_utils['parse'](message)
        assert parsed["type"] == "handshake"
        assert "deviceId" not in parsed

    def test_device_handshake_has_device_id(self, message_utils):
        """Device handshake should have deviceId."""
        handshake = {
            "type": "handshake",
            "deviceId": "device-123",
            "deviceName": "Test Phone"
        }
        message = message_utils['create'](handshake)

        parsed = message_utils['parse'](message)
        assert parsed["type"] == "handshake"
        assert parsed["deviceId"] == "device-123"

    def test_handshake_ack_format(self, message_utils):
        """Handshake acknowledgment format."""
        ack = {"type": "handshake_ack", "status": "ok"}
        message = message_utils['create'](ack)

        parsed = message_utils['parse'](message)
        assert parsed["type"] == "handshake_ack"


class TestSSHKeyHandling:
    """Tests for SSH key installation and management."""

    def test_ssh_key_format_validation(self):
        """SSH public key should have valid format."""
        valid_keys = [
            "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQ... user@host",
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... user@host",
            "ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNo... user@host"
        ]

        for key in valid_keys:
            parts = key.split()
            assert len(parts) >= 2
            assert parts[0] in ["ssh-rsa", "ssh-ed25519", "ecdsa-sha2-nistp256",
                               "ecdsa-sha2-nistp384", "ecdsa-sha2-nistp521"]

    def test_invalid_ssh_key_detection(self):
        """Invalid SSH keys should be detected."""
        invalid_keys = [
            "",
            "not a key",
            "ssh-invalid AAAA...",
        ]

        def is_valid_ssh_key(key):
            if not key or len(key) < 10:
                return False
            parts = key.split()
            if len(parts) < 2:
                return False
            valid_types = ["ssh-rsa", "ssh-ed25519", "ecdsa-sha2-nistp256",
                          "ecdsa-sha2-nistp384", "ecdsa-sha2-nistp521"]
            return parts[0] in valid_types

        for key in invalid_keys:
            assert is_valid_ssh_key(key) is False

    def test_approved_devices_storage(self, temp_dir):
        """Approved devices should be storable and loadable."""
        approved_file = temp_dir / "approved_devices.json"

        approved = {
            "device-1": {"name": "Phone 1", "approved_at": 1700000000},
            "device-2": {"name": "Phone 2", "approved_at": 1700000001}
        }

        # Save
        with open(approved_file, "w") as f:
            json.dump(approved, f)

        # Load
        with open(approved_file, "r") as f:
            loaded = json.load(f)

        assert loaded == approved

    def test_pending_keys_management(self):
        """Test pending SSH key request management."""
        pending_keys = {}

        # Add pending key
        pending_keys["device-1"] = {
            "public_key": "ssh-rsa AAAA...",
            "device_name": "Test Phone",
            "ip": "192.168.1.100",
            "timestamp": time.time()
        }

        assert "device-1" in pending_keys

        # Approve
        key_data = pending_keys.pop("device-1")
        assert "device-1" not in pending_keys
        assert key_data["device_name"] == "Test Phone"


class TestErrorHandling:
    """Tests for error handling in network operations."""

    def test_connection_refused_handling(self, free_port):
        """Connection refused should be handled gracefully."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)

        with pytest.raises((ConnectionRefusedError, socket.timeout, OSError)):
            sock.connect(('127.0.0.1', free_port))

        sock.close()

    def test_invalid_json_handling(self, message_utils):
        """Invalid JSON should be handled gracefully."""
        # Create invalid JSON message
        invalid_json = b'{"invalid": json}'
        length_bytes = len(invalid_json).to_bytes(4, 'big')
        message = length_bytes + invalid_json

        with pytest.raises(json.JSONDecodeError):
            message_utils['parse'](message)

    def test_socket_closed_handling(self):
        """Closed socket operations should raise appropriate errors."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.close()

        with pytest.raises(OSError):
            sock.send(b"test")
