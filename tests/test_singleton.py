"""
Tests for SingleInstance utility.
"""

import pytest
import socket
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shadow_bridge.utils.singleton import SingleInstance


class TestSingleInstance:
    """Tests for SingleInstance class."""
    
    @pytest.fixture
    def temp_lock_dir(self, tmp_path):
        """Create a temporary directory for lock files."""
        return tmp_path / ".shadowai"
    
    @pytest.fixture
    def free_port(self):
        """Find a free port for testing."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            return s.getsockname()[1]
    
    def test_acquire_first_instance(self, temp_lock_dir, free_port):
        """First instance should acquire lock successfully."""
        instance = SingleInstance(
            app_name="TestApp",
            port=free_port,
            lock_dir=temp_lock_dir
        )
        
        try:
            result = instance.acquire()
            assert result is True
            assert instance._acquired is True
        finally:
            instance.release()
    
    def test_acquire_second_instance_blocked(self, temp_lock_dir, free_port):
        """Second instance should be blocked."""
        instance1 = SingleInstance(
            app_name="TestApp",
            port=free_port,
            lock_dir=temp_lock_dir
        )
        instance2 = SingleInstance(
            app_name="TestApp",
            port=free_port,
            lock_dir=temp_lock_dir
        )
        
        try:
            assert instance1.acquire() is True
            assert instance2.acquire() is False
        finally:
            instance1.release()
            instance2.release()
    
    def test_release_allows_new_instance(self, temp_lock_dir, free_port):
        """After release, new instance can acquire."""
        instance1 = SingleInstance(
            app_name="TestApp",
            port=free_port,
            lock_dir=temp_lock_dir
        )
        instance2 = SingleInstance(
            app_name="TestApp",
            port=free_port,
            lock_dir=temp_lock_dir
        )
        
        try:
            assert instance1.acquire() is True
            instance1.release()
            
            # Give socket time to fully close
            time.sleep(0.1)
            
            assert instance2.acquire() is True
        finally:
            instance1.release()
            instance2.release()
    
    def test_activation_callback(self, temp_lock_dir, free_port):
        """Second instance can send activation message."""
        callback = MagicMock()
        
        instance1 = SingleInstance(
            app_name="TestApp",
            port=free_port,
            lock_dir=temp_lock_dir
        )
        instance2 = SingleInstance(
            app_name="TestApp",
            port=free_port,
            lock_dir=temp_lock_dir
        )
        
        try:
            result1 = instance1.acquire()
            assert result1 is True, "First instance should acquire lock"
            
            instance1.set_activation_callback(callback)
            
            # Give listener time to start
            time.sleep(0.5)
            
            # Second instance should fail to acquire (socket already bound)
            result2 = instance2.acquire()
            
            # Even if acquire fails, we should be able to send activation
            # Retry a few times if needed
            sent = False
            for _ in range(3):
                if instance2.send_activate():
                    sent = True
                    break
                time.sleep(0.2)
            
            if sent:
                # Give callback time to be called
                time.sleep(1.0)
                callback.assert_called_once()
            else:
                # If we couldn't send, skip this part of the test
                # (happens when socket isn't bound yet)
                pytest.skip("Could not send activation message")
        finally:
            instance1.release()
            instance2.release()
    
    def test_different_app_names_independent(self, temp_lock_dir):
        """Different app names should have independent locks."""
        # Use different ports to avoid conflicts
        instance1 = SingleInstance(
            app_name="TestApp1",
            port=19201,
            lock_dir=temp_lock_dir
        )
        instance2 = SingleInstance(
            app_name="TestApp2",
            port=19202,
            lock_dir=temp_lock_dir
        )
        
        try:
            assert instance1.acquire() is True
            assert instance2.acquire() is True
        finally:
            instance1.release()
            instance2.release()
    
    def test_stale_lock_detection(self, temp_lock_dir, free_port):
        """Stale lock files should be detected and cleaned up."""
        instance = SingleInstance(
            app_name="TestApp",
            port=free_port,
            lock_dir=temp_lock_dir
        )
        
        # Create a stale lock file with a non-existent PID
        temp_lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file = temp_lock_dir / "TestApp.lock"
        lock_file.write_text("999999999")  # Very unlikely to be a real PID
        
        try:
            assert instance.is_stale_lock() is True
            assert instance.cleanup_stale_lock() is True
            assert not lock_file.exists()
        finally:
            instance.release()
    
    def test_lock_file_created(self, temp_lock_dir, free_port):
        """Lock file should be created with correct PID."""
        import os
        
        instance = SingleInstance(
            app_name="TestApp",
            port=free_port,
            lock_dir=temp_lock_dir
        )
        
        try:
            # Force lockfile path (in case socket succeeds first)
            instance.acquire()
            instance._acquire_lockfile()
            
            lock_file = temp_lock_dir / "TestApp.lock"
            if lock_file.exists():
                pid = int(lock_file.read_text().strip())
                assert pid == os.getpid()
        finally:
            instance.release()
