"""
Thread watchdog system for monitoring and auto-restarting critical server threads.
"""

import threading
import time
import logging
from typing import Dict, Callable, Optional

log = logging.getLogger(__name__)


class ThreadWatchdog:
    """Monitor critical threads and auto-restart them if they crash.

    Example:
        watchdog = ThreadWatchdog(check_interval=5)
        watchdog.register("DataReceiver", data_receiver_thread, restart_data_receiver_fn)
        watchdog.start()
    """

    def __init__(self, check_interval: int = 5):
        """Initialize thread watchdog.

        Args:
            check_interval: Seconds between health checks (default: 5)
        """
        self.check_interval = check_interval
        self.monitored: Dict[str, Dict] = {}
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        thread: threading.Thread,
        restart_callback: Callable[[], threading.Thread],
        max_restarts: int = 5,
        restart_window: int = 300,
    ):
        """Register a thread for monitoring.

        Args:
            name: Descriptive name for the thread
            thread: Thread object to monitor
            restart_callback: Function that creates and starts a new thread
            max_restarts: Maximum restarts allowed within restart_window (default: 5)
            restart_window: Time window in seconds for counting restarts (default: 300)
        """
        with self._lock:
            self.monitored[name] = {
                "thread": thread,
                "restart_fn": restart_callback,
                "restart_count": 0,
                "last_restart": 0,
                "max_restarts": max_restarts,
                "restart_window": restart_window,
                "given_up": False,
            }
        log.debug(f"Watchdog: Registered {name} for monitoring")

    def unregister(self, name: str):
        """Stop monitoring a thread.

        Args:
            name: Name of the thread to unregister
        """
        with self._lock:
            if name in self.monitored:
                del self.monitored[name]
                log.debug(f"Watchdog: Unregistered {name}")

    def start(self):
        """Start watchdog monitoring thread."""
        if self.running:
            log.warning("Watchdog already running")
            return

        self.running = True
        self.thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="Watchdog"
        )
        self.thread.start()
        log.info(f"Watchdog started (check interval: {self.check_interval}s)")

    def stop(self):
        """Stop watchdog monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=self.check_interval + 1)
        log.debug("Watchdog stopped")

    def get_status(self) -> Dict[str, Dict]:
        """Get status of all monitored threads.

        Returns:
            Dictionary mapping thread names to their status information
        """
        with self._lock:
            status = {}
            for name, info in self.monitored.items():
                status[name] = {
                    "alive": info["thread"].is_alive() if info["thread"] else False,
                    "restart_count": info["restart_count"],
                    "last_restart": info["last_restart"],
                    "given_up": info["given_up"],
                }
            return status

    def _monitor_loop(self):
        """Main monitoring loop - checks thread health periodically."""
        while self.running:
            try:
                self._check_all_threads()
            except Exception as e:
                log.error(f"Watchdog error during health check: {e}")

            # Sleep in small increments to allow quick shutdown
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)

    def _check_all_threads(self):
        """Check health of all registered threads."""
        now = time.time()

        with self._lock:
            for name, info in self.monitored.items():
                thread = info["thread"]

                # Skip if already given up
                if info["given_up"]:
                    continue

                # Check if thread is alive
                if thread and thread.is_alive():
                    # Thread is healthy
                    # Reset restart count if outside restart window
                    if now - info["last_restart"] > info["restart_window"]:
                        if info["restart_count"] > 0:
                            log.debug(
                                f"Watchdog: {name} stable, resetting restart counter"
                            )
                        info["restart_count"] = 0
                    continue

                # Thread is dead - attempt restart
                self._attempt_restart(name, info, now)

    def _attempt_restart(self, name: str, info: Dict, now: float):
        """Attempt to restart a crashed thread.

        Args:
            name: Thread name
            info: Thread info dictionary
            now: Current timestamp
        """
        # Check if we're within restart window
        if now - info["last_restart"] < 60:
            # Don't restart more than once per minute
            log.debug(f"Watchdog: {name} too soon to restart (< 60s since last)")
            return

        # Check restart limit
        if info["restart_count"] >= info["max_restarts"]:
            if not info["given_up"]:
                log.error(
                    f"Watchdog: ❌ {name} crashed {info['max_restarts']} times, giving up"
                )
                info["given_up"] = True
            return

        # Attempt restart
        log.warning(
            f"Watchdog: {name} died, attempting restart #{info['restart_count'] + 1}"
        )

        try:
            new_thread = info["restart_fn"]()
            if new_thread and new_thread.is_alive():
                info["thread"] = new_thread
                info["restart_count"] += 1
                info["last_restart"] = now
                log.info(f"Watchdog: {name} restarted successfully")
            else:
                log.error(f"Watchdog: {name} restart failed - thread not alive")
                info["restart_count"] += 1
                info["last_restart"] = now
        except Exception as e:
            log.error(f"Watchdog: {name} restart exception: {e}")
            info["restart_count"] += 1
            info["last_restart"] = now
