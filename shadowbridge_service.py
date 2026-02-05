"""
ShadowBridge Service Wrapper

Runs ShadowBridge as a persistent background process with auto-restart.
Can be used standalone or registered as a Windows service via NSSM.

Usage:
    # Run as watchdog (auto-restarts on crash):
    python shadowbridge_service.py --watchdog

    # Run as watchdog with specific args:
    python shadowbridge_service.py --watchdog --args "--aidev"

    # Install as Windows service (requires NSSM):
    python shadowbridge_service.py --install

    # Uninstall Windows service:
    python shadowbridge_service.py --uninstall

    # Check service status:
    python shadowbridge_service.py --status
"""

import os
import sys
import time
import signal
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Setup logging
LOG_DIR = Path(os.path.expanduser("~")) / ".shadowai" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "shadowbridge_service.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("shadowbridge_service")

# Config
BRIDGE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shadow_bridge_gui.py")
SERVICE_NAME = "ShadowBridge"
SERVICE_DISPLAY = "ShadowBridge AI Companion"
SERVICE_DESCRIPTION = "ShadowAI PC companion - clipboard sync, SSH keys, Claude Code relay, autonomous agents"

# Watchdog config
MAX_RESTARTS = 10  # Max restarts before giving up
RESTART_COOLDOWN = [2, 5, 10, 30, 60, 120, 300]  # Backoff in seconds
HEALTH_CHECK_INTERVAL = 30  # Seconds between health checks
CRASH_RESET_AFTER = 3600  # Reset restart counter after 1 hour of stability


class ShadowBridgeWatchdog:
    """Watchdog that keeps ShadowBridge running with auto-restart."""

    def __init__(self, extra_args: str = ""):
        self.extra_args = extra_args
        self.process = None
        self.restart_count = 0
        self.last_crash_time = 0.0
        self.last_stable_time = time.time()
        self.running = True
        self._setup_signals()

    def _setup_signals(self):
        """Handle graceful shutdown signals."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        self._stop_process()

    def _get_backoff(self) -> int:
        """Get restart delay based on restart count."""
        idx = min(self.restart_count, len(RESTART_COOLDOWN) - 1)
        return RESTART_COOLDOWN[idx]

    def _start_process(self):
        """Start the ShadowBridge process."""
        cmd = [sys.executable, BRIDGE_SCRIPT]
        if self.extra_args:
            cmd.extend(self.extra_args.split())

        logger.info(f"Starting ShadowBridge: {' '.join(cmd)}")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=os.path.dirname(BRIDGE_SCRIPT),
            )
            logger.info(f"ShadowBridge started (PID: {self.process.pid})")
            return True
        except Exception as e:
            logger.error(f"Failed to start ShadowBridge: {e}")
            return False

    def _stop_process(self):
        """Gracefully stop the ShadowBridge process."""
        if not self.process:
            return

        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
                logger.info("ShadowBridge stopped gracefully")
            except subprocess.TimeoutExpired:
                self.process.kill()
                logger.warning("ShadowBridge force-killed after timeout")
        except Exception as e:
            logger.error(f"Error stopping ShadowBridge: {e}")

        self.process = None

    def _log_output(self):
        """Read and log process output (non-blocking)."""
        if not self.process or not self.process.stdout:
            return

        try:
            import select
            # On Windows, use a simple readline with timeout approach
            line = self.process.stdout.readline()
            if line:
                line = line.strip()
                if line:
                    logger.debug(f"[bridge] {line}")
        except Exception:
            pass

    def run(self):
        """Main watchdog loop."""
        logger.info("=" * 60)
        logger.info(f"ShadowBridge Watchdog starting")
        logger.info(f"Script: {BRIDGE_SCRIPT}")
        logger.info(f"Args: {self.extra_args or '(none)'}")
        logger.info(f"Log: {LOG_FILE}")
        logger.info("=" * 60)

        while self.running:
            # Start process if not running
            if self.process is None or self.process.poll() is not None:
                if self.process is not None:
                    exit_code = self.process.returncode
                    self.last_crash_time = time.time()
                    self.restart_count += 1

                    logger.warning(
                        f"ShadowBridge exited with code {exit_code} "
                        f"(restart {self.restart_count}/{MAX_RESTARTS})"
                    )

                    if self.restart_count >= MAX_RESTARTS:
                        logger.critical(
                            f"Max restarts ({MAX_RESTARTS}) reached. "
                            f"Giving up. Check logs at {LOG_FILE}"
                        )
                        break

                    backoff = self._get_backoff()
                    logger.info(f"Waiting {backoff}s before restart...")
                    time.sleep(backoff)

                if not self._start_process():
                    time.sleep(5)
                    continue

            # Reset restart counter after stability period
            if self.process and self.process.poll() is None:
                uptime = time.time() - self.last_crash_time if self.last_crash_time else time.time() - self.last_stable_time
                if uptime > CRASH_RESET_AFTER and self.restart_count > 0:
                    logger.info(
                        f"Stable for {uptime:.0f}s, resetting restart counter "
                        f"(was {self.restart_count})"
                    )
                    self.restart_count = 0
                    self.last_stable_time = time.time()

            time.sleep(HEALTH_CHECK_INTERVAL)

        self._stop_process()
        logger.info("Watchdog exiting")


def install_service(extra_args: str = ""):
    """Install ShadowBridge as a Windows service using NSSM."""
    nssm = _find_nssm()
    if not nssm:
        logger.error(
            "NSSM not found. Install it from https://nssm.cc/ or via:\n"
            "  choco install nssm\n"
            "  scoop install nssm"
        )
        return False

    python_exe = sys.executable
    script = os.path.abspath(__file__)
    args = f'"{script}" --watchdog'
    if extra_args:
        args += f' --args "{extra_args}"'

    logger.info(f"Installing service '{SERVICE_NAME}'...")

    cmds = [
        [nssm, "install", SERVICE_NAME, python_exe, args],
        [nssm, "set", SERVICE_NAME, "DisplayName", SERVICE_DISPLAY],
        [nssm, "set", SERVICE_NAME, "Description", SERVICE_DESCRIPTION],
        [nssm, "set", SERVICE_NAME, "Start", "SERVICE_AUTO_START"],
        [nssm, "set", SERVICE_NAME, "AppStdout", str(LOG_DIR / "service_stdout.log")],
        [nssm, "set", SERVICE_NAME, "AppStderr", str(LOG_DIR / "service_stderr.log")],
        [nssm, "set", SERVICE_NAME, "AppRotateFiles", "1"],
        [nssm, "set", SERVICE_NAME, "AppRotateBytes", "10485760"],  # 10MB
    ]

    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
            return False

    logger.info(f"Service '{SERVICE_NAME}' installed successfully!")
    logger.info(f"Start with: nssm start {SERVICE_NAME}")
    logger.info(f"Or via: net start {SERVICE_NAME}")
    return True


def uninstall_service():
    """Uninstall the ShadowBridge Windows service."""
    nssm = _find_nssm()
    if not nssm:
        logger.error("NSSM not found")
        return False

    # Stop first
    subprocess.run([nssm, "stop", SERVICE_NAME], capture_output=True)
    time.sleep(1)

    result = subprocess.run(
        [nssm, "remove", SERVICE_NAME, "confirm"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        logger.info(f"Service '{SERVICE_NAME}' removed")
        return True
    else:
        logger.error(f"Failed to remove service: {result.stderr}")
        return False


def check_status():
    """Check if the ShadowBridge service is running."""
    nssm = _find_nssm()

    # Check NSSM service
    if nssm:
        result = subprocess.run(
            [nssm, "status", SERVICE_NAME],
            capture_output=True, text=True,
        )
        status = result.stdout.strip() if result.returncode == 0 else "NOT_INSTALLED"
        print(f"Windows Service: {status}")
    else:
        print("Windows Service: NSSM not found")

    # Check if ShadowBridge process is running
    try:
        import psutil
        bridge_procs = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline') or []
                cmdline_str = ' '.join(cmdline).lower()
                if 'shadow_bridge_gui' in cmdline_str:
                    bridge_procs.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if bridge_procs:
            print(f"ShadowBridge processes: {len(bridge_procs)}")
            for p in bridge_procs:
                print(f"  PID {p['pid']}: {' '.join(p.get('cmdline', []))[:100]}")
        else:
            print("ShadowBridge processes: none running")
    except ImportError:
        print("(psutil not available for process check)")

    # Check log file
    if LOG_FILE.exists():
        size_kb = LOG_FILE.stat().st_size / 1024
        mtime = datetime.fromtimestamp(LOG_FILE.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"Log file: {LOG_FILE} ({size_kb:.1f} KB, last modified: {mtime})")
    else:
        print(f"Log file: {LOG_FILE} (not found)")


def _find_nssm() -> str:
    """Find NSSM executable."""
    import shutil
    nssm = shutil.which("nssm")
    if nssm:
        return nssm

    # Check common install locations
    for path in [
        r"C:\Program Files\nssm\win64\nssm.exe",
        r"C:\Program Files (x86)\nssm\win64\nssm.exe",
        r"C:\tools\nssm\win64\nssm.exe",
        os.path.expanduser(r"~\scoop\apps\nssm\current\nssm.exe"),
    ]:
        if os.path.isfile(path):
            return path

    return ""


def main():
    parser = argparse.ArgumentParser(description="ShadowBridge Service Manager")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--watchdog", action="store_true", help="Run as watchdog (auto-restart on crash)")
    group.add_argument("--install", action="store_true", help="Install as Windows service via NSSM")
    group.add_argument("--uninstall", action="store_true", help="Uninstall Windows service")
    group.add_argument("--status", action="store_true", help="Check service status")

    parser.add_argument("--args", default="", help='Extra args for shadow_bridge_gui.py (e.g., "--aidev")')

    args = parser.parse_args()

    if args.watchdog:
        watchdog = ShadowBridgeWatchdog(extra_args=args.args)
        watchdog.run()
    elif args.install:
        install_service(extra_args=args.args)
    elif args.uninstall:
        uninstall_service()
    elif args.status:
        check_status()


if __name__ == "__main__":
    main()
