"""
Deployment Automation

Installs Android APKs to connected devices with verification and rollback.
"""

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .adb_service import get_adb_service

logger = logging.getLogger(__name__)

LAST_GOOD_STATE_FILE = os.path.join(os.path.expanduser("~"), ".shadowai", "last_good_apk.json")


@dataclass
class DeviceDeployResult:
    device_id: str
    success: bool
    install_output: str = ""
    verify_output: str = ""
    activity_output: str = ""
    error: Optional[str] = None


@dataclass
class DeploymentResult:
    success: bool
    workspace: str
    apk_path: str
    package_name: str
    devices: List[DeviceDeployResult] = field(default_factory=list)
    rollback_attempted: bool = False
    rollback_results: List[DeviceDeployResult] = field(default_factory=list)
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class DeploymentAutomation:
    """Deploy APKs to devices with verification and rollback support."""

    def __init__(self, state_file: str = LAST_GOOD_STATE_FILE):
        self._adb = get_adb_service()
        self._state_file = state_file
        self._lock = threading.Lock()

    def deploy_android(
        self,
        workspace: str,
        apk_path: Optional[str] = None,
        package_name: Optional[str] = None,
        activity: Optional[str] = None,
        allow_downgrade: bool = True,
        rollback_on_failure: bool = True
    ) -> DeploymentResult:
        with self._lock:
            resolved_apk = apk_path or self._find_latest_apk(workspace)
            if not resolved_apk or not os.path.exists(resolved_apk):
                return DeploymentResult(
                    success=False,
                    workspace=workspace,
                    apk_path=resolved_apk or "",
                    package_name=package_name or "",
                    error="APK path not found"
                )

            resolved_package = package_name or self._derive_package_name(workspace, resolved_apk)
            if not resolved_package:
                return DeploymentResult(
                    success=False,
                    workspace=workspace,
                    apk_path=resolved_apk,
                    package_name="",
                    error="Package name could not be resolved"
                )

            devices = self._adb.get_devices()
            if not devices:
                return DeploymentResult(
                    success=False,
                    workspace=workspace,
                    apk_path=resolved_apk,
                    package_name=resolved_package,
                    error="No ADB devices connected"
                )

            results = []
            for device_id in devices:
                results.append(
                    self._install_and_verify(
                        resolved_apk,
                        resolved_package,
                        device_id,
                        activity,
                        allow_downgrade
                    )
                )

            success = all(r.success for r in results)
            deploy_result = DeploymentResult(
                success=success,
                workspace=workspace,
                apk_path=resolved_apk,
                package_name=resolved_package,
                devices=results,
            )

            if success:
                self._save_last_good(resolved_apk, resolved_package, activity)
                return deploy_result

            if rollback_on_failure:
                rollback_result = self.rollback(devices)
                deploy_result.rollback_attempted = True
                deploy_result.rollback_results = rollback_result
                deploy_result.error = "Deployment failed; rollback attempted"
            else:
                deploy_result.error = "Deployment failed"

            return deploy_result

    def rollback(self, devices: Optional[List[str]] = None) -> List[DeviceDeployResult]:
        last_good = self._load_last_good()
        if not last_good:
            return [DeviceDeployResult(
                device_id="none",
                success=False,
                error="No last-known-good APK recorded"
            )]

        apk_path = last_good.get("apk_path")
        package_name = last_good.get("package_name")
        activity = last_good.get("activity")

        if not apk_path or not os.path.exists(apk_path):
            return [DeviceDeployResult(
                device_id="none",
                success=False,
                error="Last-known-good APK not found on disk"
            )]

        target_devices = devices or self._adb.get_devices()
        if not target_devices:
            return [DeviceDeployResult(
                device_id="none",
                success=False,
                error="No ADB devices connected"
            )]

        results = []
        for device_id in target_devices:
            results.append(
                self._install_and_verify(
                    apk_path,
                    package_name,
                    device_id,
                    activity,
                    allow_downgrade=True
                )
            )
        return results

    # ---- Internal helpers ----

    def _install_and_verify(
        self,
        apk_path: str,
        package_name: str,
        device_id: str,
        activity: Optional[str],
        allow_downgrade: bool
    ) -> DeviceDeployResult:
        output, success = self._adb.install_apk(
            apk_path,
            device_id=device_id,
            reinstall=True,
            downgrade=allow_downgrade
        )
        if not success:
            return DeviceDeployResult(
                device_id=device_id,
                success=False,
                install_output=output,
                error="adb install failed"
            )

        verify_output, verify_ok = self._adb.shell(f"pm path {package_name}", device_id)
        if not verify_ok or "package:" not in verify_output:
            return DeviceDeployResult(
                device_id=device_id,
                success=False,
                install_output=output,
                verify_output=verify_output,
                error="Package verification failed"
            )

        activity_output = ""
        if activity:
            activity_output, activity_ok = self._adb.shell(
                f"am start -n {package_name}/{activity}",
                device_id
            )
            if not activity_ok:
                return DeviceDeployResult(
                    device_id=device_id,
                    success=False,
                    install_output=output,
                    verify_output=verify_output,
                    activity_output=activity_output,
                    error="Activity launch failed"
                )

        return DeviceDeployResult(
            device_id=device_id,
            success=True,
            install_output=output,
            verify_output=verify_output,
            activity_output=activity_output
        )

    def _find_latest_apk(self, workspace: str) -> Optional[str]:
        apk_root = os.path.join(workspace, "app", "build", "outputs", "apk")
        if not os.path.isdir(apk_root):
            return None

        candidates = []
        for root, _, files in os.walk(apk_root):
            for filename in files:
                if not filename.endswith(".apk"):
                    continue
                full_path = os.path.join(root, filename)
                candidates.append(full_path)

        if not candidates:
            return None

        def rank(path: str):
            lower = path.lower()
            score = 0
            if "debug" in lower:
                score += 5
            if "/dev/" in lower or "\\dev\\" in lower:
                score += 2
            if "/aidev/" in lower or "\\aidev\\" in lower:
                score += 1
            return (score, os.path.getmtime(path))

        return max(candidates, key=rank)

    def _derive_package_name(self, workspace: str, apk_path: str) -> Optional[str]:
        gradle_path = os.path.join(workspace, "app", "build.gradle.kts")
        if not os.path.exists(gradle_path):
            return None

        with open(gradle_path, "r", encoding="utf-8") as f:
            content = f.read()

        base_match = re.search(r'applicationId\s*=\s*"([^"]+)"', content)
        if not base_match:
            return None
        base_id = base_match.group(1)

        debug_suffix = ""
        debug_match = re.search(
            r'debug\s*\{[^}]*?applicationIdSuffix\s*=\s*"([^"]+)"',
            content,
            re.S
        )
        if debug_match:
            debug_suffix = debug_match.group(1)

        flavor_suffix = ""
        if "aidev" in apk_path.lower():
            flavor_match = re.search(
                r'create\("aidev"\)\s*\{[^}]*?applicationIdSuffix\s*=\s*"([^"]+)"',
                content,
                re.S
            )
            if flavor_match:
                flavor_suffix = flavor_match.group(1)

        return f"{base_id}{flavor_suffix}{debug_suffix}"

    def _load_last_good(self) -> Optional[dict]:
        if not os.path.exists(self._state_file):
            return None
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _save_last_good(self, apk_path: str, package_name: str, activity: Optional[str]):
        os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
        payload = {
            "apk_path": apk_path,
            "package_name": package_name,
            "activity": activity,
            "timestamp": datetime.utcnow().isoformat()
        }
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
