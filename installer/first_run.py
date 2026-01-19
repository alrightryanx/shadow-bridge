"""
First-Run Setup Wizard for ShadowBridge
---------------------------------------
Runs on first launch to configure essential settings.
"""

import json
import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"
SHADOWAI_DIR = Path.home() / ".shadowai"
FIRST_RUN_MARKER = SHADOWAI_DIR / ".first_run_complete"
SETTINGS_FILE = SHADOWAI_DIR / "settings.json"


def is_first_run() -> bool:
    """Check if this is the first run after installation."""
    return not FIRST_RUN_MARKER.exists()


def mark_first_run_complete():
    """Mark first run as complete."""
    SHADOWAI_DIR.mkdir(parents=True, exist_ok=True)
    FIRST_RUN_MARKER.touch()
    logger.info("First run setup completed")


def check_ssh_status() -> Tuple[bool, str]:
    """
    Check if SSH server is running and accessible.

    Returns:
        Tuple of (is_running, status_message)
    """
    if IS_WINDOWS:
        try:
            result = subprocess.run(
                ["sc", "query", "sshd"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if "RUNNING" in result.stdout:
                return True, "OpenSSH Server is running"
            elif "STOPPED" in result.stdout:
                return False, "OpenSSH Server is installed but stopped"
            else:
                return False, "OpenSSH Server may not be installed"
        except Exception as e:
            return False, f"Could not check SSH status: {e}"
    else:
        try:
            result = subprocess.run(
                ["systemctl", "is-active", "sshd"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return True, "SSH daemon is running"
            return False, "SSH daemon is not running"
        except Exception:
            return False, "Could not check SSH status"


def check_firewall_rules() -> Tuple[bool, list]:
    """
    Check if ShadowBridge firewall rules exist.

    Returns:
        Tuple of (all_rules_exist, list of missing rules)
    """
    if not IS_WINDOWS:
        return True, []  # Non-Windows doesn't need this check

    required_rules = [
        "ShadowBridge Discovery",
        "ShadowBridge Data",
        "ShadowBridge Companion",
        "ShadowBridge Dashboard",
    ]

    missing_rules = []

    try:
        result = subprocess.run(
            ["netsh", "advfirewall", "firewall", "show", "rule", "name=all"],
            capture_output=True,
            text=True,
            timeout=10
        )

        for rule in required_rules:
            if rule not in result.stdout:
                missing_rules.append(rule)

        return len(missing_rules) == 0, missing_rules

    except Exception as e:
        logger.error(f"Failed to check firewall rules: {e}")
        return False, required_rules


def configure_firewall_rules() -> bool:
    """
    Configure Windows Firewall rules for ShadowBridge.
    Requires admin privileges.

    Returns:
        True if successful
    """
    if not IS_WINDOWS:
        return True

    rules = [
        ("ShadowBridge Discovery", "UDP", "19283"),
        ("ShadowBridge Data", "TCP", "19284"),
        ("ShadowBridge Companion", "TCP", "19286"),
        ("ShadowBridge Dashboard", "TCP", "6767"),
    ]

    success = True
    for name, protocol, port in rules:
        try:
            # Check if rule exists first
            check = subprocess.run(
                ["netsh", "advfirewall", "firewall", "show", "rule", f"name={name}"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if "No rules match" in check.stdout or check.returncode != 0:
                # Create rule
                result = subprocess.run(
                    [
                        "netsh", "advfirewall", "firewall", "add", "rule",
                        f"name={name}",
                        f"protocol={protocol}",
                        f"localport={port}",
                        "dir=in",
                        "action=allow"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0:
                    logger.info(f"Created firewall rule: {name}")
                else:
                    logger.error(f"Failed to create firewall rule {name}: {result.stderr}")
                    success = False
        except Exception as e:
            logger.error(f"Error configuring firewall rule {name}: {e}")
            success = False

    return success


def configure_startup(enable: bool = True) -> bool:
    """
    Configure ShadowBridge to start with Windows.

    Args:
        enable: True to enable startup, False to disable

    Returns:
        True if successful
    """
    if not IS_WINDOWS:
        return True

    try:
        import winreg

        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"

        # Find the executable path
        exe_path = None

        # Check Program Files first
        pf_path = Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "ShadowBridge" / "ShadowBridge.exe"
        if pf_path.exists():
            exe_path = str(pf_path)
        else:
            # Check LocalAppData
            la_path = Path.home() / "AppData" / "Local" / "Programs" / "ShadowBridge" / "ShadowBridge.exe"
            if la_path.exists():
                exe_path = str(la_path)

        if not exe_path:
            logger.warning("Could not find ShadowBridge executable for startup configuration")
            return False

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            key_path,
            0,
            winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE
        )

        try:
            if enable:
                winreg.SetValueEx(key, "ShadowBridge", 0, winreg.REG_SZ, f'"{exe_path}" --minimized')
                logger.info("Enabled ShadowBridge startup")
            else:
                winreg.DeleteValue(key, "ShadowBridge")
                logger.info("Disabled ShadowBridge startup")
        except FileNotFoundError:
            if not enable:
                pass  # Key doesn't exist, which is fine for disable
        finally:
            winreg.CloseKey(key)

        return True

    except Exception as e:
        logger.error(f"Failed to configure startup: {e}")
        return False


def load_settings() -> dict:
    """Load settings from settings file."""
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load settings: {e}")
    return {}


def save_settings(settings: dict):
    """Save settings to settings file."""
    try:
        SHADOWAI_DIR.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")


def run_first_run_setup(
    configure_firewall: bool = True,
    enable_startup: bool = True,
    auto_update: bool = True
) -> dict:
    """
    Run first-run setup with specified options.

    Args:
        configure_firewall: Whether to configure firewall rules
        enable_startup: Whether to enable Windows startup
        auto_update: Whether to enable auto-updates

    Returns:
        Dict with setup results
    """
    results = {
        "firewall": None,
        "startup": None,
        "auto_update": None,
        "ssh_status": None,
    }

    # Check SSH status
    ssh_running, ssh_message = check_ssh_status()
    results["ssh_status"] = {"running": ssh_running, "message": ssh_message}

    # Configure firewall
    if configure_firewall:
        results["firewall"] = configure_firewall_rules()

    # Configure startup
    if enable_startup:
        results["startup"] = configure_startup(enable=True)

    # Save settings
    settings = load_settings()
    settings["auto_update"] = auto_update
    settings["first_run_complete"] = True
    save_settings(settings)
    results["auto_update"] = True

    # Mark complete
    mark_first_run_complete()

    return results
