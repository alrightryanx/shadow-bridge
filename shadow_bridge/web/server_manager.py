"""
Web Server Lifecycle Manager for ShadowBridge
----------------------------------------------
Manages Flask/SocketIO server start, stop, restart, and health checks.
"""

import atexit
import logging
import socket
import threading
import time
from typing import Callable, Optional, Dict, Any

logger = logging.getLogger(__name__)


class WebServerManager:
    """
    Manages the Flask/SocketIO web server lifecycle.
    
    Features:
    - Graceful start with port conflict detection
    - Clean shutdown with resource cleanup
    - Health check endpoint support
    - Automatic cleanup on app exit
    
    Usage:
        manager = WebServerManager(host="0.0.0.0", port=6767)
        manager.start()
        # ... app runs ...
        manager.stop()  # or automatic via atexit
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 6767,
        debug: bool = False,
    ):
        self.host = host
        self.port = port
        self.debug = debug
        
        self._server_thread: Optional[threading.Thread] = None
        self._app = None
        self._socketio = None
        self._is_running = False
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        
        # Register cleanup on exit
        atexit.register(self.stop)
    
    @property
    def is_running(self) -> bool:
        """Check if server is currently running."""
        return self._is_running and self._server_thread is not None and self._server_thread.is_alive()
    
    @property
    def uptime_seconds(self) -> float:
        """Get server uptime in seconds."""
        if self._start_time and self._is_running:
            return time.time() - self._start_time
        return 0
    
    def is_port_available(self) -> bool:
        """Check if the configured port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.host if self.host != "0.0.0.0" else "127.0.0.1", self.port))
                return True
        except socket.error:
            return False
    
    def find_available_port(self, start_port: int = None, max_attempts: int = 10) -> Optional[int]:
        """Find an available port starting from start_port."""
        start = start_port or self.port
        
        for offset in range(max_attempts):
            port = start + offset
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("127.0.0.1", port))
                    return port
            except socket.error:
                continue
        
        return None
    
    def start(self, app_factory: Callable = None, use_alt_port: bool = False) -> bool:
        """
        Start the web server.
        
        Args:
            app_factory: Optional callable that returns (app, socketio) tuple
            use_alt_port: If True, find alternative port if preferred is busy
            
        Returns:
            True if started successfully, False otherwise
        """
        with self._lock:
            if self._is_running:
                logger.warning("Web server is already running")
                return True
            
            # Check port availability
            if not self.is_port_available():
                if use_alt_port:
                    alt_port = self.find_available_port(self.port + 1)
                    if alt_port:
                        logger.warning(
                            f"Port {self.port} in use, using {alt_port}"
                        )
                        self.port = alt_port
                    else:
                        logger.error(
                            f"Port {self.port} in use and no alternatives available"
                        )
                        return False
                else:
                    logger.error(f"Port {self.port} is already in use")
                    return False
            
            # Create app if factory provided
            if app_factory:
                try:
                    result = app_factory()
                    if isinstance(result, tuple):
                        self._app, self._socketio = result
                    else:
                        self._app = result
                        self._socketio = None
                except Exception as e:
                    logger.error(f"Failed to create app: {e}")
                    return False
            
            # Start server in background thread
            self._shutdown_event.clear()
            self._server_thread = threading.Thread(
                target=self._run_server,
                daemon=True,
                name="ShadowBridge-WebServer"
            )
            self._server_thread.start()
            self._is_running = True
            self._start_time = time.time()
            
            # Wait a moment to verify startup
            time.sleep(0.5)
            if not self._server_thread.is_alive():
                self._is_running = False
                logger.error("Web server thread died during startup")
                return False
            
            logger.info(f"Web server started on {self.host}:{self.port}")
            return True
    
    def stop(self, timeout: float = 5.0) -> bool:
        """
        Stop the web server gracefully.
        
        Args:
            timeout: Maximum seconds to wait for shutdown
            
        Returns:
            True if stopped cleanly, False if forced
        """
        with self._lock:
            if not self._is_running:
                return True
            
            logger.info("Stopping web server...")
            self._shutdown_event.set()
            
            # Signal Flask/SocketIO to shutdown
            if self._socketio:
                try:
                    self._socketio.stop()
                except Exception as e:
                    logger.debug(f"SocketIO stop error: {e}")
            
            # Wait for thread to finish
            if self._server_thread and self._server_thread.is_alive():
                self._server_thread.join(timeout=timeout)
                
                if self._server_thread.is_alive():
                    logger.warning("Web server thread did not stop gracefully")
                    self._is_running = False
                    return False
            
            self._is_running = False
            self._start_time = None
            logger.info("Web server stopped")
            return True
    
    def restart(self, app_factory: Callable = None) -> bool:
        """
        Restart the web server.
        
        Args:
            app_factory: Optional callable that returns (app, socketio) tuple
            
        Returns:
            True if restarted successfully
        """
        logger.info("Restarting web server...")
        
        if not self.stop():
            logger.warning("Forced shutdown during restart")
        
        # Brief delay to ensure port is released
        time.sleep(0.5)
        
        return self.start(app_factory=app_factory)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Get server health status.
        
        Returns:
            Dictionary with health information
        """
        return {
            "status": "ok" if self.is_running else "stopped",
            "host": self.host,
            "port": self.port,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "thread_alive": self._server_thread.is_alive() if self._server_thread else False,
        }
    
    def _run_server(self):
        """Internal method to run the server."""
        try:
            if self._socketio and self._app:
                self._socketio.run(
                    self._app,
                    host=self.host,
                    port=self.port,
                    debug=self.debug,
                    use_reloader=False,
                    log_output=False,
                )
            elif self._app:
                self._app.run(
                    host=self.host,
                    port=self.port,
                    debug=self.debug,
                    use_reloader=False,
                )
            else:
                # No app provided - run minimal server for testing
                logger.warning("No Flask app provided, running minimal server")
                self._run_minimal_server()
                
        except Exception as e:
            if not self._shutdown_event.is_set():
                logger.error(f"Web server error: {e}")
        finally:
            self._is_running = False
    
    def _run_minimal_server(self):
        """Run a minimal HTTP server for testing."""
        import http.server
        import socketserver
        
        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress logging
            
            def do_GET(self):
                if self.path == "/api/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    import json
                    self.wfile.write(json.dumps({"status": "ok"}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
        
        with socketserver.TCPServer((self.host, self.port), QuietHandler) as httpd:
            httpd.timeout = 0.5
            while not self._shutdown_event.is_set():
                httpd.handle_request()


# Global instance for convenience
_server_manager: Optional[WebServerManager] = None


def get_server_manager(host: str = "0.0.0.0", port: int = 6767) -> WebServerManager:
    """Get or create the global server manager instance."""
    global _server_manager
    if _server_manager is None:
        _server_manager = WebServerManager(host=host, port=port)
    return _server_manager
