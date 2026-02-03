"""
Shadow Sentinel - Health monitoring and auto-healing for AI services.

Monitors:
- ShadowBridge Web Server
- Ollama (Local LLM)
- Node.js Backend (if applicable)
- Telemetry trends (Ouroboros V2)
- Health scoring (Ouroboros V2)
"""

import threading
import time
import logging
import requests
import socket
import subprocess
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '..', 'backend', 'data', 'shadow_ai.db'
)

class Sentinel:
    """
    Background service that monitors the health of local AI services.
    Extended with Ouroboros V2: trend analysis and health scoring.
    """

    def __init__(self, check_interval: int = 30, db_path: str = None):
        self.check_interval = check_interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.db_path = db_path or DEFAULT_DB_PATH

        # Ouroboros V2 components
        self.trend_analyzer = None
        self.health_scorer = None
        self._trend_check_counter = 0
        self._trend_check_interval = 30  # Run trend check every 30 loops (~15 min at 30s interval)

        self.services = {
            "shadow_bridge": {
                "name": "ShadowBridge Web",
                "url": "http://localhost:6767/api/telemetry/ping",
                "type": "http",
                "critical": True
            },
            "ollama": {
                "name": "Ollama",
                "url": "http://localhost:11434/api/tags",
                "type": "http",
                "critical": False
            }
        }
        
        self.health_status = {name: True for name in self.services}

    def start(self):
        """Starts the sentinel monitoring thread."""
        if self.running:
            return

        # Initialize Ouroboros V2 components
        try:
            from .trend_analyzer import TrendAnalyzer
            from .health_scorer import HealthScorer
            self.trend_analyzer = TrendAnalyzer(self.db_path)
            self.health_scorer = HealthScorer(self.db_path)
            logger.info("Ouroboros V2 components initialized")
        except Exception as e:
            logger.warning(f"Ouroboros V2 init failed (non-critical): {e}")

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("Shadow Sentinel started")

    def stop(self):
        """Stops the sentinel monitoring thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info("Shadow Sentinel stopped")

    def get_status(self) -> Dict[str, bool]:
        """Returns current health status of all monitored services."""
        return self.health_status

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                for key, config in self.services.items():
                    is_healthy = self._check_service(config)

                    if not is_healthy and self.health_status.get(key, True):
                        logger.warning(f"Service {config['name']} is DOWN")
                        self._attempt_healing(key, config)
                    elif is_healthy and not self.health_status.get(key, True):
                        logger.info(f"Service {config['name']} recovered")

                    self.health_status[key] = is_healthy

                # Ouroboros V2: Run trend analysis periodically
                self._trend_check_counter += 1
                if self._trend_check_counter >= self._trend_check_interval:
                    self._trend_check_counter = 0
                    self._run_trend_analysis()

            except Exception as e:
                logger.error(f"Sentinel monitor error: {e}")

            time.sleep(self.check_interval)

    def _run_trend_analysis(self):
        """Run Ouroboros V2 trend analysis and health scoring."""
        try:
            if self.trend_analyzer:
                alerts = self.trend_analyzer.run()
                if alerts:
                    logger.info(f"[Sentinel] Trend analysis: {len(alerts)} alerts generated")

            if self.health_scorer:
                score = self.health_scorer.compute_score()
                logger.info(f"[Sentinel] Health score: {score.overall}/100 ({score.status})")
                self.health_scorer.push_notification_if_degraded(score)

        except Exception as e:
            logger.error(f"[Sentinel] Trend analysis failed: {e}")

    def _check_service(self, config: Dict) -> bool:
        """Checks if a specific service is healthy."""
        if config["type"] == "http":
            return self._check_http(config["url"])
        return False

    def _check_http(self, url: str) -> bool:
        """Checks HTTP endpoint."""
        try:
            response = requests.get(url, timeout=2)
            return 200 <= response.status_code < 300
        except (requests.RequestException, OSError) as e:
            logger.debug(f"HTTP check failed for {url}: {e}")
            return False

    def _attempt_healing(self, key: str, config: Dict):
        """Attempts to heal/restart a failed service."""
        logger.info(f"Attempting to heal {config['name']}...")
        
        if key == "ollama":
            # Try to start Ollama if it's down
            try:
                if self._is_process_running("ollama"):
                    logger.info("Ollama process exists but unresponsive")
                    # Could kill and restart, but risky
                else:
                    logger.info("Ollama not running, attempting start...")
                    # subprocess.Popen(["ollama", "serve"], ...)
                    # For now, just log, as starting servers might be complex
                    pass
            except Exception as e:
                logger.error(f"Healing failed: {e}")

    def _is_process_running(self, process_name: str) -> bool:
        """Checks if a process is running."""
        try:
            # Simple check using tasklist on Windows
            output = subprocess.check_output(
                ["tasklist", "/FI", f"IMAGENAME eq {process_name}.exe"],
                text=True
            )
            return process_name in output
        except (subprocess.SubprocessError, OSError) as e:
            logger.debug(f"Process check failed for {process_name}: {e}")
            return False
