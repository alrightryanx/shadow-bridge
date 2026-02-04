"""
Single Instance Enforcement for ShadowBridge
---------------------------------------------
Multi-method single-instance detection with IPC for window activation.

Methods used:
1. Windows Named Mutex (primary, most reliable)
2. Socket binding (fallback, cross-platform)
3. Lock file with PID (tertiary fallback)
"""

import atexit
import ctypes
import logging
import os
import platform
import socket
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"


class SingleInstance:
    """
    Cross-platform single instance enforcement with IPC.
    
    Usage:
        instance = SingleInstance("ShadowBridge", port=19287)
        if not instance.acquire():
            instance.send_activate()  # Tell existing instance to show window
            sys.exit(0)
        
        instance.set_activation_callback(my_window.show)
        
        # ... app runs ...
        
        # Cleanup is automatic via atexit
    """
    
    def __init__(
        self,
        app_name: str = "ShadowBridge",
        port: int = 19287,
        lock_dir: Optional[Path] = None,
    ):
        self.app_name = app_name
        self.port = port
        self.lock_dir = lock_dir or Path.home() / ".shadowai"
        
        self._mutex_handle = None
        self._socket = None
        self._lock_file = None
        self._listener_thread = None
        self._activation_callback: Optional[Callable[[], None]] = None
        self._ping_callback: Optional[Callable[[], None]] = None
        self._stop_event = threading.Event()
        self._acquired = False
        
        # Register cleanup on exit
        atexit.register(self.release)
    
    def acquire(self) -> bool:
        """
        Try to acquire single-instance lock.
        
        Returns:
            True if lock acquired (this is the primary instance)
            False if another instance is running
        """
        if self._acquired:
            return True
        
        # Try to acquire socket first (needed for IPC/Activation)
        socket_acquired = self._acquire_socket()

        # On Windows, use Mutex as the authoritative source of truth for "Single Instance"
        if IS_WINDOWS:
            mutex_acquired = self._acquire_mutex()
            
            if mutex_acquired:
                # We are the one true instance
                self._acquired = True
                
                # We should have the socket for IPC. If not, something else is blocking port
                # but we still own the "app instance" lock.
                if socket_acquired:
                    self._start_listener()
                else:
                    logger.warning("Acquired Mutex but Socket was busy. Activation features may fail.")
                    
                return True
            else:
                # We failed to get mutex, so we are a second instance.
                # If we grabbed the socket by accident (race?), release it so primary can have it
                # (though primary likely already failed to get it if we got it)
                if socket_acquired:
                    self._release_socket()
                return False

        # Non-Windows logic
        if socket_acquired:
            self._acquired = True
            self._start_listener()
            return True
        
        if self._acquire_lockfile():
            self._acquired = True
            self._start_listener()
            return True
        
        return False
    
    def release(self):
        """Release all locks and clean up."""
        self._stop_event.set()
        
        if self._listener_thread and self._listener_thread.is_alive():
            try:
                self._listener_thread.join(timeout=1)
            except Exception:
                pass
        
        self._release_mutex()
        self._release_socket()
        self._release_lockfile()
        
        self._acquired = False
    
    def send_activate(self, retries: int = 3, backoff: float = 0.5) -> bool:
        """
        Send activation message to existing instance with retry logic.

        Args:
            retries: Number of retry attempts
            backoff: Initial backoff time in seconds (doubles each retry)

        Returns:
            True if message was sent successfully
        """
        for attempt in range(retries):
            sock = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                sock.connect(("127.0.0.1", self.port))
                sock.sendall(b"ACTIVATE")
                logger.info("Sent activation message to existing instance")
                return True
            except socket.error as e:
                wait_time = backoff * (2 ** attempt)
                logger.debug(f"Activation attempt {attempt + 1}/{retries} failed: {e}, retrying in {wait_time}s")
                time.sleep(wait_time)
            finally:
                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass

        logger.warning(f"Could not send activation message after {retries} attempts")
        return False
    
    

    def send_ping(self, retries: int = 3, backoff: float = 0.5) -> bool:
        for attempt in range(retries):
            sock = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                sock.connect(("127.0.0.1", self.port))
                sock.sendall(b"PING")
                return True
            except socket.error:
                time.sleep(backoff * (2 ** attempt))
            finally:
                if sock: sock.close()
        return False

    def set_ping_callback(self, callback: Callable[[], None]):
        self._ping_callback = callback
    def set_activation_callback(self, callback: Callable[[], None]):
        """
        Set callback to be called when another instance requests activation.
        
        Args:
            callback: Function to call (e.g., window.show, window.deiconify)
        """
        self._activation_callback = callback
    
    def is_stale_lock(self) -> bool:
        """
        Check if an existing lock appears to be stale (from crashed process).
        
        Returns:
            True if lock exists but process is dead
        """
        lock_file = self.lock_dir / f"{self.app_name}.lock"
        if not lock_file.exists():
            return False
        
        try:
            with open(lock_file, "r") as f:
                pid = int(f.read().strip())
            
            return not self._is_process_running(pid)
        except Exception:
            return True  # Can't read file, assume stale
    
    def cleanup_stale_lock(self) -> bool:
        """
        Remove stale lock if detected.
        
        Returns:
            True if stale lock was cleaned up
        """
        if not self.is_stale_lock():
            return False
        
        lock_file = self.lock_dir / f"{self.app_name}.lock"
        try:
            lock_file.unlink()
            logger.info("Cleaned up stale lock file")
            return True
        except Exception as e:
            logger.warning(f"Failed to clean up stale lock: {e}")
            return False
    
    # ============ Windows Mutex ============
    
    def _acquire_mutex(self) -> bool:
        """Acquire Windows named mutex."""
        if not IS_WINDOWS:
            return False
        
        try:
            mutex_name = f"Global\\{self.app_name}_SingleInstance"
            
            # CreateMutexW returns handle or NULL
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.CreateMutexW(None, True, mutex_name)
            
            if handle == 0:
                return False
            
            # Check if we created it or it already existed
            ERROR_ALREADY_EXISTS = 183
            if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
                kernel32.CloseHandle(handle)
                return False
            
            self._mutex_handle = handle
            logger.debug("Acquired Windows mutex lock")
            return True
            
        except Exception as e:
            logger.debug(f"Failed to acquire mutex: {e}")
            return False
    
    def _release_mutex(self):
        """Release Windows mutex."""
        if self._mutex_handle and IS_WINDOWS:
            try:
                ctypes.windll.kernel32.ReleaseMutex(self._mutex_handle)
                ctypes.windll.kernel32.CloseHandle(self._mutex_handle)
                self._mutex_handle = None
                logger.debug("Released Windows mutex lock")
            except Exception as e:
                logger.debug(f"Failed to release mutex: {e}")
    
    # ============ Socket Lock ============
    
    def _acquire_socket(self) -> bool:
        """Acquire socket-based lock."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            sock.bind(("127.0.0.1", self.port))
            sock.listen(5)
            sock.settimeout(0.5)  # Non-blocking for listener loop
            
            self._socket = sock
            logger.debug(f"Acquired socket lock on port {self.port}")
            return True
            
        except socket.error as e:
            logger.debug(f"Failed to acquire socket lock: {e}")
            return False
    
    def _release_socket(self):
        """Release socket lock."""
        if self._socket:
            try:
                self._socket.close()
                self._socket = None
                logger.debug("Released socket lock")
            except Exception as e:
                logger.debug(f"Failed to release socket: {e}")
    
    # ============ Lock File ============
    
    def _acquire_lockfile(self) -> bool:
        """Acquire lock file."""
        try:
            self.lock_dir.mkdir(parents=True, exist_ok=True)
            lock_file = self.lock_dir / f"{self.app_name}.lock"
            
            # Check for stale lock
            if lock_file.exists():
                if self.is_stale_lock():
                    self.cleanup_stale_lock()
                else:
                    return False
            
            # Write our PID
            with open(lock_file, "w") as f:
                f.write(str(os.getpid()))
            
            self._lock_file = lock_file
            logger.debug(f"Acquired lock file: {lock_file}")
            return True
            
        except Exception as e:
            logger.debug(f"Failed to acquire lock file: {e}")
            return False
    
    def _release_lockfile(self):
        """Release lock file."""
        if self._lock_file and self._lock_file.exists():
            try:
                self._lock_file.unlink()
                self._lock_file = None
                logger.debug("Released lock file")
            except Exception as e:
                logger.debug(f"Failed to release lock file: {e}")
    
    # ============ IPC Listener ============
    
    def _start_listener(self):
        """Start background thread to listen for activation messages."""
        if not self._socket:
            return
        
        def listener():
            while not self._stop_event.is_set():
                try:
                    conn, addr = self._socket.accept()
                    data = conn.recv(32).decode("utf-8", errors="ignore").strip()
                    conn.close()
                    
                    if data == "ACTIVATE" and self._activation_callback:
                        logger.info("Received activation request from another instance")
                        try:
                            self._activation_callback()
                        except Exception as e:
                            logger.error(f"Activation callback failed: {e}")
                    elif data == "PING" and self._ping_callback:
                        logger.info("Received ping request from another instance")
                        try:
                            self._ping_callback()
                        except Exception as e:
                            logger.error(f"Ping callback failed: {e}")
                            
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self._stop_event.is_set():
                        logger.debug(f"Listener error: {e}")
                    break
        
        self._listener_thread = threading.Thread(target=listener, daemon=True)
        self._listener_thread.start()
        logger.debug("Started activation listener thread")
    
    # ============ Utilities ============
    
    @staticmethod
    def _is_process_running(pid: int) -> bool:
        """Check if a process with given PID is running."""
        if IS_WINDOWS:
            try:
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                handle = ctypes.windll.kernel32.OpenProcess(
                    PROCESS_QUERY_LIMITED_INFORMATION, False, pid
                )
                if handle:
                    ctypes.windll.kernel32.CloseHandle(handle)
                    return True
                return False
            except Exception:
                return False
        else:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False
