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
import sqlite3
from datetime import datetime, timezone, timedelta
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

                # Ouroboros V2: Run trend analysis and fix verification periodically
                self._trend_check_counter += 1
                if self._trend_check_counter >= self._trend_check_interval:
                    self._trend_check_counter = 0
                    self._run_trend_analysis()
                    self._verify_deployed_fixes()

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

    def _verify_deployed_fixes(self):
        """Verify whether deployed fixes actually resolved the crash patterns.

        For each pending fix older than 1 hour:
        - If the crash pattern's last_seen is BEFORE deployed_at -> verified
        - If last_seen is AFTER deployed_at AND occurrence_count increased -> regressed
        - Otherwise -> keep pending (not enough data yet)
        """
        try:
            conn = sqlite3.connect(self.db_path)

            # Check if deployed_fixes table exists
            table_check = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='deployed_fixes'"
            ).fetchone()
            if not table_check:
                conn.close()
                return

            cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

            cursor = conn.execute("""
                SELECT df.id, df.issue_number, df.pattern_signature, df.deployed_at,
                       cp.last_seen, cp.occurrence_count
                FROM deployed_fixes df
                LEFT JOIN crash_patterns cp ON df.pattern_signature = cp.id
                WHERE df.verification_status = 'pending'
                  AND df.deployed_at < ?
            """, (cutoff,))

            rows = cursor.fetchall()
            if not rows:
                conn.close()
                return

            verified_count = 0
            regressed_count = 0
            now_iso = datetime.now(timezone.utc).isoformat()

            for fix_id, issue_number, pattern_sig, deployed_at, last_seen, occurrence_count in rows:
                if not pattern_sig or not last_seen:
                    # No linked pattern or no crash data - skip for now
                    continue

                if last_seen < deployed_at:
                    # Pattern hasn't recurred since fix -> verified
                    conn.execute(
                        "UPDATE deployed_fixes SET verification_status = 'verified', verified_at = ? WHERE id = ?",
                        (now_iso, fix_id)
                    )
                    verified_count += 1
                    logger.info(f"[Sentinel] Fix {fix_id} for issue #{issue_number} VERIFIED - pattern not seen since deployment")
                elif last_seen > deployed_at:
                    # Pattern recurred after fix -> regressed
                    conn.execute(
                        "UPDATE deployed_fixes SET verification_status = 'regressed', verified_at = ?, "
                        "notes = 'Pattern recurred after fix deployment' WHERE id = ?",
                        (now_iso, fix_id)
                    )
                    regressed_count += 1
                    logger.warning(f"[Sentinel] Fix {fix_id} for issue #{issue_number} REGRESSED - pattern recurred after deployment")

                    # Create a critical trend alert for regressions
                    import uuid
                    alert_id = str(uuid.uuid4())[:12]
                    conn.execute("""
                        INSERT OR IGNORE INTO trend_alerts
                            (id, metric_name, severity, current_value, baseline_value,
                             deviation_pct, message, recommendation, resolved, created_at)
                        VALUES (?, ?, 'critical', ?, 0, 100.0, ?, ?, 0, ?)
                    """, (
                        alert_id,
                        'fix_regression',
                        occurrence_count or 0,
                        f"Fix for issue #{issue_number} regressed - crash pattern recurred after deployment",
                        f"Review issue #{issue_number} and apply a different fix strategy. Previous approach failed.",
                        now_iso,
                    ))

            conn.commit()
            conn.close()

            if verified_count or regressed_count:
                logger.info(f"[Sentinel] Fix verification: {verified_count} verified, {regressed_count} regressed")

        except Exception as e:
            logger.error(f"[Sentinel] Fix verification failed: {e}")

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
