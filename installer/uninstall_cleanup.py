"""
Uninstall Cleanup for ShadowBridge
----------------------------------
Removes firewall rules, startup registry entries, and optionally user data.
Run this script during uninstallation or manually to clean up.
"""

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"
SHADOWAI_DIR = Path.home() / ".shadowai"


def remove_firewall_rules() -> bool:
    """
    Remove ShadowBridge firewall rules from Windows Firewall.

    Returns:
        True if successful or not applicable
    """
    if not IS_WINDOWS:
        return True

    rules = [
        "ShadowBridge Discovery",
        "ShadowBridge Data",
        "ShadowBridge Companion",
        "ShadowBridge Dashboard",
    ]

    success = True
    for rule_name in rules:
        try:
            result = subprocess.run(
                ["netsh", "advfirewall", "firewall", "delete", "rule", f"name={rule_name}"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                logger.info(f"Removed firewall rule: {rule_name}")
            elif "No rules match" in result.stdout:
                logger.debug(f"Firewall rule not found (already removed): {rule_name}")
            else:
                logger.warning(f"Could not remove firewall rule {rule_name}: {result.stderr}")
                # Don't mark as failure if rule just doesn't exist

        except Exception as e:
            logger.error(f"Error removing firewall rule {rule_name}: {e}")
            success = False

    return success


def remove_startup_entry() -> bool:
    """
    Remove ShadowBridge from Windows startup.

    Returns:
        True if successful or not applicable
    """
    if not IS_WINDOWS:
        return True

    try:
        import winreg

        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            key_path,
            0,
            winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE
        )

        try:
            winreg.DeleteValue(key, "ShadowBridge")
            logger.info("Removed ShadowBridge from startup")
        except FileNotFoundError:
            logger.debug("ShadowBridge startup entry not found (already removed)")
        finally:
            winreg.CloseKey(key)

        return True

    except Exception as e:
        logger.error(f"Failed to remove startup entry: {e}")
        return False


def remove_install_path_registry() -> bool:
    """
    Remove ShadowBridge install path from registry.

    Returns:
        True if successful or not applicable
    """
    if not IS_WINDOWS:
        return True

    try:
        import winreg

        key_path = r"Software\ShadowBridge"

        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key_path)
            logger.info("Removed ShadowBridge registry key")
        except FileNotFoundError:
            logger.debug("ShadowBridge registry key not found (already removed)")

        return True

    except Exception as e:
        logger.error(f"Failed to remove registry key: {e}")
        return False


def remove_user_data(confirm: bool = False) -> bool:
    """
    Remove ShadowBridge user data directory.

    Args:
        confirm: Must be True to actually delete (safety check)

    Returns:
        True if successful or skipped
    """
    if not confirm:
        logger.info("User data removal skipped (confirmation required)")
        return True

    if not SHADOWAI_DIR.exists():
        logger.debug("User data directory doesn't exist")
        return True

    try:
        # Only remove specific ShadowBridge files, preserve other shadowai data
        sb_files = [
            "settings.json",
            "window_state.json",
            "shadowbridge.log",
            ".first_run_complete",
            "ShadowBridge.lock",
        ]

        for filename in sb_files:
            filepath = SHADOWAI_DIR / filename
            if filepath.exists():
                filepath.unlink()
                logger.info(f"Removed: {filepath}")

        # Remove known_devices if it only has ShadowBridge data
        known_devices = SHADOWAI_DIR / "known_devices.json"
        if known_devices.exists():
            # Leave this as user may want to keep device pairing info
            logger.info(f"Preserved: {known_devices} (device pairing data)")

        logger.info("User data cleanup completed")
        return True

    except Exception as e:
        logger.error(f"Failed to remove user data: {e}")
        return False


def full_cleanup(remove_data: bool = False) -> dict:
    """
    Perform full uninstall cleanup.

    Args:
        remove_data: Whether to remove user data

    Returns:
        Dict with cleanup results
    """
    results = {
        "firewall_rules": remove_firewall_rules(),
        "startup_entry": remove_startup_entry(),
        "registry_key": remove_install_path_registry(),
        "user_data": remove_user_data(confirm=remove_data),
    }

    success_count = sum(1 for v in results.values() if v)
    logger.info(f"Cleanup completed: {success_count}/{len(results)} tasks successful")

    return results


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="ShadowBridge Uninstall Cleanup")
    parser.add_argument("--remove-data", action="store_true", help="Also remove user data")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    if args.quiet:
        logging.disable(logging.CRITICAL)

    results = full_cleanup(remove_data=args.remove_data)

    if all(results.values()):
        print("Cleanup completed successfully.")
        sys.exit(0)
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"Cleanup completed with issues: {', '.join(failed)}")
        sys.exit(1)
