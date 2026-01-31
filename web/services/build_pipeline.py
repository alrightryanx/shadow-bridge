"""
Build Pipeline Service

Handles version bumping, safety checks, building, and deployment
for the autonomous agent system.

Respects project rules:
- Unix paths in bash commands
- No adb uninstall
- No -r flag on adb install
- No fallbackToDestructiveMigration
- Gradle lock for single build at a time
- Version always read from file, increment exactly +1
"""

import os
import re
import subprocess
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths
ANDROID_ROOT = "C:/shadow/shadow-android"
BRIDGE_ROOT = "C:/shadow/shadow-bridge"
ADB_PATH = r"C:\android\platform-tools\adb.exe"
GRADLE_KTS = os.path.join(ANDROID_ROOT, "app", "build.gradle.kts")
BRIDGE_GUI = os.path.join(BRIDGE_ROOT, "shadow_bridge_gui.py")
BRIDGE_DATA_SERVICE = os.path.join(BRIDGE_ROOT, "web", "services", "data_service.py")

# Build limits
MAX_BUILDS_PER_SESSION = 6
MIN_BUILD_INTERVAL_SECONDS = 900  # 15 minutes

# Gradle lock file
GRADLE_LOCK_FILE = os.path.join(ANDROID_ROOT, ".aidev", "gradle.lock")
GRADLE_LOCK_TIMEOUT = 1800  # 30 minutes


@dataclass
class BuildResult:
    success: bool
    build_type: str  # "debug_apk", "release_aab", "bridge_exe"
    output_path: Optional[str] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    log: str = ""
    version_code: Optional[int] = None
    version_name: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class DeployResult:
    success: bool
    device_id: str
    apk_path: str
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SafetyResult:
    passed: bool
    checks: List[dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    success: bool
    safety: Optional[SafetyResult] = None
    build: Optional[BuildResult] = None
    deploys: List[DeployResult] = field(default_factory=list)
    error: Optional[str] = None


class BuildPipeline:
    """Manages the build pipeline for autonomous agents."""

    def __init__(self):
        self._build_count = 0
        self._last_build_time = 0.0
        self._build_history: List[dict] = []

    def increment_android_version(self) -> Tuple[int, str]:
        """
        Read build.gradle.kts, increment versionCode by 1 and versionName by 0.001.
        Always reads file first - never guesses.

        Returns:
            (new_version_code, new_version_name)
        """
        if not os.path.exists(GRADLE_KTS):
            raise FileNotFoundError(f"build.gradle.kts not found at {GRADLE_KTS}")

        with open(GRADLE_KTS, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract current versionCode
        vc_match = re.search(r"versionCode\s*=\s*(\d+)", content)
        if not vc_match:
            raise ValueError("Could not find versionCode in build.gradle.kts")
        current_vc = int(vc_match.group(1))

        # Extract current versionName
        vn_match = re.search(r'versionName\s*=\s*"([^"]+)"', content)
        if not vn_match:
            raise ValueError("Could not find versionName in build.gradle.kts")
        current_vn = vn_match.group(1)

        # Increment exactly +1
        new_vc = current_vc + 1
        new_vn_float = float(current_vn) + 0.001
        new_vn = f"{new_vn_float:.3f}"

        # Write back
        content = re.sub(
            r"versionCode\s*=\s*\d+",
            f"versionCode = {new_vc}",
            content
        )
        content = re.sub(
            r'versionName\s*=\s*"[^"]+"',
            f'versionName = "{new_vn}"',
            content
        )

        with open(GRADLE_KTS, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Android version bumped: {current_vc} -> {new_vc}, {current_vn} -> {new_vn}")
        return new_vc, new_vn

    def increment_bridge_version(self) -> str:
        """
        Increment APP_VERSION in shadow_bridge_gui.py AND data_service.py.
        Both must match.

        Returns:
            new_version string
        """
        # Read current version from gui file
        with open(BRIDGE_GUI, "r", encoding="utf-8") as f:
            gui_content = f.read()

        match = re.search(r'APP_VERSION\s*=\s*"([^"]+)"', gui_content)
        if not match:
            raise ValueError("Could not find APP_VERSION in shadow_bridge_gui.py")

        current = match.group(1)
        new_ver_float = float(current) + 0.001
        new_ver = f"{new_ver_float:.3f}"

        # Update gui file
        gui_content = re.sub(
            r'APP_VERSION\s*=\s*"[^"]+"',
            f'APP_VERSION = "{new_ver}"',
            gui_content
        )
        with open(BRIDGE_GUI, "w", encoding="utf-8") as f:
            f.write(gui_content)

        # Update data_service.py
        if os.path.exists(BRIDGE_DATA_SERVICE):
            with open(BRIDGE_DATA_SERVICE, "r", encoding="utf-8") as f:
                ds_content = f.read()
            # Match version in get_status() - pattern: "version": "X.YYY"
            ds_content = re.sub(
                r'"version"\s*:\s*"[^"]+"',
                f'"version": "{new_ver}"',
                ds_content
            )
            with open(BRIDGE_DATA_SERVICE, "w", encoding="utf-8") as f:
                f.write(ds_content)

        logger.info(f"Bridge version bumped: {current} -> {new_ver}")
        return new_ver

    def run_safety_checks(self, repo: str) -> SafetyResult:
        """
        Run safety checks before building.

        Checks:
        - Kotlin compilation (Android)
        - Python syntax (Bridge)
        - No fallbackToDestructiveMigration
        - No API keys in git diff
        """
        result = SafetyResult(passed=True)

        if "shadow-android" in repo:
            # Kotlin compile check
            check = self._run_compile_check_android()
            result.checks.append(check)
            if not check["passed"]:
                result.passed = False
                result.errors.append(check.get("error", "Kotlin compilation failed"))

            # Check for forbidden patterns
            forbidden = self._check_forbidden_patterns(repo)
            result.checks.append(forbidden)
            if not forbidden["passed"]:
                result.passed = False
                result.errors.extend(forbidden.get("errors", []))

        elif "shadow-bridge" in repo:
            # Python syntax check
            check = self._run_syntax_check_python()
            result.checks.append(check)
            if not check["passed"]:
                result.passed = False
                result.errors.append(check.get("error", "Python syntax check failed"))

        # Check for secrets in git diff
        secrets = self._check_secrets_in_diff(repo)
        result.checks.append(secrets)
        if not secrets["passed"]:
            result.passed = False
            result.errors.extend(secrets.get("errors", []))

        return result

    def build_android_debug(self) -> BuildResult:
        """Build debug APK using gradle."""
        if not self._can_build():
            return BuildResult(
                success=False,
                build_type="debug_apk",
                error="Build rate limit exceeded"
            )

        start = time.time()

        try:
            self._acquire_gradle_lock("build_pipeline")

            returncode, output = self._run_command(
                "cd /c/shadow/shadow-android && ./gradlew assembleDebug 2>&1",
                timeout=600
            )

            duration = time.time() - start
            apk_path = os.path.join(
                ANDROID_ROOT, "app", "build", "outputs", "apk", "debug", "app-debug.apk"
            )

            result = BuildResult(
                success=returncode == 0,
                build_type="debug_apk",
                output_path=apk_path if returncode == 0 else None,
                duration_seconds=duration,
                error=None if returncode == 0 else "Build failed",
                log=output[-2000:] if output else "",  # Last 2000 chars
            )

            if result.success:
                self._record_build(result)

            return result

        except Exception as e:
            return BuildResult(
                success=False,
                build_type="debug_apk",
                error=str(e),
                duration_seconds=time.time() - start,
            )
        finally:
            self._release_gradle_lock("build_pipeline")

    def build_android_release(self) -> BuildResult:
        """Build release AAB for Play Store."""
        if not self._can_build():
            return BuildResult(
                success=False,
                build_type="release_aab",
                error="Build rate limit exceeded"
            )

        start = time.time()

        try:
            self._acquire_gradle_lock("build_pipeline")

            returncode, output = self._run_command(
                "cd /c/shadow/shadow-android && ./gradlew bundleRelease 2>&1",
                timeout=600
            )

            duration = time.time() - start
            aab_path = os.path.join(
                ANDROID_ROOT, "app", "build", "outputs", "bundle", "release", "app-release.aab"
            )

            result = BuildResult(
                success=returncode == 0,
                build_type="release_aab",
                output_path=aab_path if returncode == 0 else None,
                duration_seconds=duration,
                error=None if returncode == 0 else "Build failed",
                log=output[-2000:] if output else "",
            )

            if result.success:
                self._record_build(result)

            return result

        except Exception as e:
            return BuildResult(
                success=False,
                build_type="release_aab",
                error=str(e),
                duration_seconds=time.time() - start,
            )
        finally:
            self._release_gradle_lock("build_pipeline")

    def build_bridge_exe(self) -> BuildResult:
        """Build ShadowBridge EXE with PyInstaller."""
        start = time.time()

        try:
            returncode, output = self._run_command(
                'cd /c/shadow/shadow-bridge && "/c/Windows/py.exe" -m PyInstaller ShadowBridge.spec --noconfirm 2>&1',
                timeout=300
            )

            duration = time.time() - start
            exe_path = os.path.join(BRIDGE_ROOT, "dist", "ShadowBridge.exe")

            return BuildResult(
                success=returncode == 0,
                build_type="bridge_exe",
                output_path=exe_path if returncode == 0 else None,
                duration_seconds=duration,
                error=None if returncode == 0 else "PyInstaller build failed",
                log=output[-2000:] if output else "",
            )
        except Exception as e:
            return BuildResult(
                success=False,
                build_type="bridge_exe",
                error=str(e),
                duration_seconds=time.time() - start,
            )

    def deploy_to_devices(self, apk_path: str) -> List[DeployResult]:
        """
        Install debug APK to all connected ADB devices.
        No -r flag, no uninstall. If install fails, report and move on.
        """
        results = []

        if not os.path.exists(apk_path):
            return [DeployResult(
                success=False, device_id="none",
                apk_path=apk_path, error="APK file not found"
            )]

        # Get connected devices
        devices = self._get_adb_devices()
        if not devices:
            return [DeployResult(
                success=False, device_id="none",
                apk_path=apk_path, error="No ADB devices connected"
            )]

        for device_id in devices:
            try:
                returncode, output = self._run_command(
                    f'cmd.exe /c "{ADB_PATH} -s {device_id} install {apk_path}"',
                    timeout=120
                )
                results.append(DeployResult(
                    success=returncode == 0,
                    device_id=device_id,
                    apk_path=apk_path,
                    error=None if returncode == 0 else output[:200],
                ))
            except Exception as e:
                results.append(DeployResult(
                    success=False,
                    device_id=device_id,
                    apk_path=apk_path,
                    error=str(e),
                ))

        return results

    def run_pipeline(self, repo: str, deploy: bool = True) -> PipelineResult:
        """
        Full pipeline: version bump -> safety check -> build -> deploy.

        Args:
            repo: "shadow-android" or "shadow-bridge"
            deploy: Whether to deploy after build (debug APK only)
        """
        try:
            # Version bump
            if "shadow-android" in repo:
                vc, vn = self.increment_android_version()
            elif "shadow-bridge" in repo:
                self.increment_bridge_version()
            else:
                return PipelineResult(success=False, error=f"Unknown repo: {repo}")

            # Safety checks
            repo_path = ANDROID_ROOT if "android" in repo else BRIDGE_ROOT
            safety = self.run_safety_checks(repo_path)
            if not safety.passed:
                return PipelineResult(
                    success=False,
                    safety=safety,
                    error=f"Safety checks failed: {'; '.join(safety.errors)}"
                )

            # Build
            if "shadow-android" in repo:
                build = self.build_android_debug()
            else:
                build = self.build_bridge_exe()

            if not build.success:
                return PipelineResult(
                    success=False,
                    safety=safety,
                    build=build,
                    error=f"Build failed: {build.error}"
                )

            # Deploy (debug APK only)
            deploys = []
            if deploy and build.build_type == "debug_apk" and build.output_path:
                deploys = self.deploy_to_devices(build.output_path)

            return PipelineResult(
                success=True,
                safety=safety,
                build=build,
                deploys=deploys,
            )

        except Exception as e:
            logger.exception("Pipeline failed")
            return PipelineResult(success=False, error=str(e))

    def get_build_history(self) -> List[dict]:
        """Return build history for dashboard."""
        return list(self._build_history)

    # ---- Internal helpers ----

    def _can_build(self) -> bool:
        """Check rate limits."""
        if self._build_count >= MAX_BUILDS_PER_SESSION:
            logger.warning("Max builds per session reached")
            return False

        now = time.time()
        if now - self._last_build_time < MIN_BUILD_INTERVAL_SECONDS:
            remaining = int(MIN_BUILD_INTERVAL_SECONDS - (now - self._last_build_time))
            logger.warning(f"Build rate limited, {remaining}s remaining")
            return False

        return True

    def _record_build(self, result: BuildResult):
        """Record a build in history."""
        self._build_count += 1
        self._last_build_time = time.time()
        self._build_history.append(asdict(result))

    def _run_command(self, command: str, timeout: int = 120) -> Tuple[int, str]:
        """Run a shell command and return (returncode, output)."""
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.path.expanduser("~"),
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            return proc.returncode, output
        except subprocess.TimeoutExpired:
            return 1, f"Command timed out after {timeout}s"
        except Exception as e:
            return 1, str(e)

    def _run_compile_check_android(self) -> dict:
        """Quick kotlin compile check."""
        returncode, output = self._run_command(
            "cd /c/shadow/shadow-android && ./gradlew compileDebugKotlin 2>&1",
            timeout=300
        )
        return {
            "name": "kotlin_compile",
            "passed": returncode == 0,
            "error": output[-500:] if returncode != 0 else None,
        }

    def _run_syntax_check_python(self) -> dict:
        """Python syntax check on key files."""
        key_files = [BRIDGE_GUI]
        for fpath in key_files:
            if os.path.exists(fpath):
                returncode, output = self._run_command(
                    f'"/c/Windows/py.exe" -m py_compile "{fpath}" 2>&1',
                    timeout=30
                )
                if returncode != 0:
                    return {
                        "name": "python_syntax",
                        "passed": False,
                        "error": output[:500],
                    }
        return {"name": "python_syntax", "passed": True}

    def _check_forbidden_patterns(self, repo: str) -> dict:
        """Check for forbidden code patterns."""
        errors = []
        repo_path = Path(repo)

        # Check for fallbackToDestructiveMigration
        for kt_file in repo_path.rglob("*.kt"):
            try:
                content = kt_file.read_text(encoding="utf-8", errors="ignore")
                if "fallbackToDestructiveMigration" in content:
                    errors.append(
                        f"FORBIDDEN: fallbackToDestructiveMigration in {kt_file.relative_to(repo_path)}"
                    )
            except OSError:
                continue

        return {
            "name": "forbidden_patterns",
            "passed": len(errors) == 0,
            "errors": errors,
        }

    def _check_secrets_in_diff(self, repo: str) -> dict:
        """Check for secrets in uncommitted changes."""
        errors = []
        returncode, output = self._run_command(
            f'cd "{repo}" && git diff --cached --diff-filter=d 2>&1',
            timeout=30
        )

        if output:
            secret_patterns = [
                r"(api[_-]?key|apikey)\s*[=:]\s*['\"][A-Za-z0-9]",
                r"(secret|password|token)\s*[=:]\s*['\"][A-Za-z0-9]",
                r"sk-[A-Za-z0-9]{20,}",
                r"AIza[A-Za-z0-9_-]{35}",
            ]
            for pattern in secret_patterns:
                if re.search(pattern, output, re.IGNORECASE):
                    errors.append(f"Potential secret found in git diff matching: {pattern}")

        return {
            "name": "secrets_check",
            "passed": len(errors) == 0,
            "errors": errors,
        }

    def _get_adb_devices(self) -> List[str]:
        """Get list of connected ADB device IDs."""
        try:
            returncode, output = self._run_command(
                f'cmd.exe /c "{ADB_PATH} devices"',
                timeout=10
            )
            if returncode != 0:
                return []

            devices = []
            for line in output.strip().split("\n")[1:]:
                parts = line.strip().split("\t")
                if len(parts) >= 2 and parts[1] == "device":
                    devices.append(parts[0])
            return devices
        except Exception:
            return []

    def _acquire_gradle_lock(self, agent_id: str):
        """Acquire gradle build lock."""
        lock_dir = os.path.dirname(GRADLE_LOCK_FILE)
        os.makedirs(lock_dir, exist_ok=True)

        # Check if lock exists and is stale
        if os.path.exists(GRADLE_LOCK_FILE):
            try:
                with open(GRADLE_LOCK_FILE, "r") as f:
                    lock_data = json.load(f)
                age = time.time() - lock_data.get("timestamp", 0)
                if age > GRADLE_LOCK_TIMEOUT:
                    os.remove(GRADLE_LOCK_FILE)
                    logger.info("Removed stale gradle lock")
                else:
                    owner = lock_data.get("agent_id", "unknown")
                    raise RuntimeError(
                        f"Gradle lock held by {owner} for {int(age)}s"
                    )
            except (json.JSONDecodeError, OSError):
                os.remove(GRADLE_LOCK_FILE)

        lock_data = {
            "agent_id": agent_id,
            "timestamp": time.time(),
            "pid": os.getpid(),
        }
        with open(GRADLE_LOCK_FILE, "w") as f:
            json.dump(lock_data, f)

    def _release_gradle_lock(self, agent_id: str):
        """Release gradle build lock."""
        try:
            if os.path.exists(GRADLE_LOCK_FILE):
                os.remove(GRADLE_LOCK_FILE)
        except OSError:
            pass
