"""
Build Automation

Runs build/test gates in the workspace that contains the changes.
Returns structured results for downstream routing and UI.
"""

import json
import os
import re
import subprocess
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

DEFAULT_ANDROID_ROOT = "C:/shadow/shadow-android"
DEFAULT_BRIDGE_ROOT = "C:/shadow/shadow-bridge"
DEFAULT_ANDROID_SMOKE_TESTS = [
    "com.shadow.projects.ProjectManagerTest",
    "com.shadow.util.VoiceCommandResolverTest",
]
DEFAULT_BRIDGE_SMOKE_COMMANDS = [
    'cmd.exe /c "py -c \"import shadow_bridge; print(\'ok\')\""',
    'cmd.exe /c "py -c \"from shadow_bridge.utils import singleton; print(\'ok\')\""',
]


def _get_config_value(key: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(key)
    if val:
        return val

    env_path = "C:/shadow/backend/.env"
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(f"{key}="):
                        return line.split("=", 1)[1].strip()
        except OSError:
            pass
    return default


@dataclass
class GateStepResult:
    name: str
    success: bool
    command: Optional[str] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    log: str = ""
    artifacts: List[str] = field(default_factory=list)


@dataclass
class GateResult:
    success: bool
    repo: str
    workspace: str
    steps: List[GateStepResult] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class BuildAutomation:
    """Build/test gate runner that executes within a given workspace."""

    def __init__(self):
        self._history: List[dict] = []

    def run_gate(self, repo: str, workspace: Optional[str] = None) -> GateResult:
        start = time.time()
        normalized_repo = repo or ""
        workspace_root = workspace or self._default_workspace(normalized_repo)
        steps: List[GateStepResult] = []

        if not workspace_root or not os.path.isdir(workspace_root):
            return GateResult(
                success=False,
                repo=normalized_repo,
                workspace=workspace_root or "",
                error=f"Workspace not found: {workspace_root}"
            )

        try:
            # Pre-build safety: check for uncommitted changes
            preflight = self._preflight_check(workspace_root, normalized_repo)
            steps.append(preflight)
            if not preflight.success:
                return self._finalize(False, normalized_repo, workspace_root, steps, start, preflight.error)

            if "android" in normalized_repo.lower():
                bump = self._bump_android_version(workspace_root)
                steps.append(bump)
                if not bump.success:
                    return self._finalize(False, normalized_repo, workspace_root, steps, start, bump.error)

                build = self._run_gradle(workspace_root, "assembleDebug", "android_build", timeout=900)
                steps.append(build)
                if not build.success:
                    return self._finalize(False, normalized_repo, workspace_root, steps, start, build.error)

                smoke = self._run_android_smoke_tests(workspace_root)
                steps.append(smoke)
                if not smoke.success:
                    return self._finalize(False, normalized_repo, workspace_root, steps, start, smoke.error)

                return self._finalize(True, normalized_repo, workspace_root, steps, start, None)

            if "bridge" in normalized_repo.lower():
                bump = self._bump_bridge_version(workspace_root)
                steps.append(bump)
                if not bump.success:
                    return self._finalize(False, normalized_repo, workspace_root, steps, start, bump.error)

                build = self._run_pyinstaller(workspace_root)
                steps.append(build)
                if not build.success:
                    return self._finalize(False, normalized_repo, workspace_root, steps, start, build.error)

                smoke_steps = self._run_bridge_smoke(workspace_root)
                steps.extend(smoke_steps)
                if any(not step.success for step in smoke_steps):
                    error = next((step.error for step in smoke_steps if not step.success), "Bridge smoke tests failed")
                    return self._finalize(False, normalized_repo, workspace_root, steps, start, error)

                return self._finalize(True, normalized_repo, workspace_root, steps, start, None)

            return self._finalize(False, normalized_repo, workspace_root, steps, start, "Unknown repo")

        except Exception as exc:
            logger.exception("Build gate failed")
            return self._finalize(False, normalized_repo, workspace_root, steps, start, str(exc))

    def get_build_history(self) -> List[dict]:
        return list(self._history)

    # ---- Internal helpers ----

    def _preflight_check(self, workspace: str, repo: str) -> GateStepResult:
        """Pre-build safety checks: verify workspace is clean and buildable."""
        start = time.time()
        issues = []

        # Check git status for uncommitted changes
        returncode, output, _ = self._run_command("git status --porcelain", workspace, 30)
        if returncode == 0 and output.strip():
            dirty_files = len(output.strip().split("\n"))
            # Auto-commit changes from agents before building
            self._run_command("git add -A", workspace, 30)
            self._run_command(
                'git commit -m "chore: auto-commit agent changes before build"',
                workspace, 30
            )
            issues.append(f"Auto-committed {dirty_files} dirty files")

        # Verify no merge conflicts
        returncode, output, _ = self._run_command(
            "git diff --check HEAD", workspace, 30
        )
        if returncode != 0 and "conflict" in output.lower():
            return GateStepResult(
                name="preflight",
                success=False,
                duration_seconds=time.time() - start,
                error=f"Merge conflicts detected: {output[:500]}"
            )

        # Check workspace directory exists and has expected files
        if "android" in repo.lower():
            if not os.path.exists(os.path.join(workspace, "gradlew.bat")) and \
               not os.path.exists(os.path.join(workspace, "gradlew")):
                return GateStepResult(
                    name="preflight",
                    success=False,
                    duration_seconds=time.time() - start,
                    error="Missing gradlew - workspace may be corrupted"
                )
        elif "bridge" in repo.lower():
            if not os.path.exists(os.path.join(workspace, "shadow_bridge_gui.py")):
                return GateStepResult(
                    name="preflight",
                    success=False,
                    duration_seconds=time.time() - start,
                    error="Missing shadow_bridge_gui.py - workspace may be corrupted"
                )

        log = "; ".join(issues) if issues else "Workspace clean"
        return GateStepResult(
            name="preflight",
            success=True,
            duration_seconds=time.time() - start,
            log=log
        )

    def _default_workspace(self, repo: str) -> str:
        if "android" in repo.lower():
            return DEFAULT_ANDROID_ROOT
        if "bridge" in repo.lower():
            return DEFAULT_BRIDGE_ROOT
        return ""

    def _finalize(
        self,
        success: bool,
        repo: str,
        workspace: str,
        steps: List[GateStepResult],
        start_time: float,
        error: Optional[str]
    ) -> GateResult:
        duration = time.time() - start_time
        result = GateResult(
            success=success,
            repo=repo,
            workspace=workspace,
            steps=steps,
            duration_seconds=duration,
            error=error
        )
        self._history.append(asdict(result))
        return result

    def _run_command(self, command: str, cwd: str, timeout: int) -> Tuple[int, str, float]:
        start = time.time()
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            duration = time.time() - start
            return proc.returncode, output, duration
        except subprocess.TimeoutExpired:
            return 1, f"Command timed out after {timeout}s", time.time() - start
        except Exception as exc:
            return 1, str(exc), time.time() - start

    def _run_gradle(self, workspace: str, target: str, name: str, timeout: int) -> GateStepResult:
        command = f'cmd.exe /c ".\\gradlew.bat {target}"'
        returncode, output, duration = self._run_command(command, workspace, timeout)
        return GateStepResult(
            name=name,
            success=returncode == 0,
            command=command,
            duration_seconds=duration,
            error=None if returncode == 0 else "Gradle command failed",
            log=output[-2000:] if output else ""
        )

    def _run_pyinstaller(self, workspace: str) -> GateStepResult:
        command = 'cmd.exe /c "py -m PyInstaller ShadowBridge.spec --noconfirm"'
        returncode, output, duration = self._run_command(command, workspace, 600)
        artifacts = []
        exe_path = os.path.join(workspace, "dist", "ShadowBridge.exe")
        if os.path.exists(exe_path):
            artifacts.append(exe_path)
        return GateStepResult(
            name="bridge_build",
            success=returncode == 0,
            command=command,
            duration_seconds=duration,
            error=None if returncode == 0 else "PyInstaller build failed",
            log=output[-2000:] if output else "",
            artifacts=artifacts
        )

    def _run_bridge_smoke(self, workspace: str) -> List[GateStepResult]:
        commands = self._get_bridge_smoke_commands()
        if not commands:
            return [
                GateStepResult(
                    name="bridge_smoke",
                    success=False,
                    error="No bridge smoke commands configured"
                )
            ]

        steps: List[GateStepResult] = []
        for idx, command in enumerate(commands, start=1):
            returncode, output, duration = self._run_command(command, workspace, 60)
            steps.append(GateStepResult(
                name=f"bridge_smoke_{idx}",
                success=returncode == 0,
                command=command,
                duration_seconds=duration,
                error=None if returncode == 0 else "Bridge smoke test failed",
                log=output[-1000:] if output else ""
            ))
        return steps

    def _run_android_smoke_tests(self, workspace: str) -> GateStepResult:
        tests = self._get_android_smoke_tests()
        if not tests:
            return GateStepResult(
                name="android_smoke",
                success=False,
                error="No Android smoke tests configured"
            )

        includes = " ".join([f'--tests "{test}"' for test in tests])
        command = f'cmd.exe /c ".\\gradlew.bat :app:testDebugUnitTest {includes}"'
        returncode, output, duration = self._run_command(command, workspace, 600)
        return GateStepResult(
            name="android_smoke",
            success=returncode == 0,
            command=command,
            duration_seconds=duration,
            error=None if returncode == 0 else "Android smoke tests failed",
            log=output[-2000:] if output else ""
        )

    def _get_android_smoke_tests(self) -> List[str]:
        raw = _get_config_value("ANDROID_SMOKE_TESTS")
        if raw:
            return [item.strip() for item in raw.split(",") if item.strip()]
        return list(DEFAULT_ANDROID_SMOKE_TESTS)

    def _get_bridge_smoke_commands(self) -> List[str]:
        raw = _get_config_value("BRIDGE_SMOKE_COMMANDS")
        if raw:
            stripped = raw.strip()
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    return [item.strip() for item in parsed if str(item).strip()]
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in raw.split("||") if item.strip()]
        return list(DEFAULT_BRIDGE_SMOKE_COMMANDS)

    def _bump_android_version(self, workspace: str) -> GateStepResult:
        gradle_path = os.path.join(workspace, "app", "build.gradle.kts")
        start = time.time()
        if not os.path.exists(gradle_path):
            return GateStepResult(
                name="android_version_bump",
                success=False,
                error=f"Missing build.gradle.kts at {gradle_path}"
            )

        with open(gradle_path, "r", encoding="utf-8") as f:
            content = f.read()

        vc_match = re.search(r"versionCode\s*=\s*(\d+)", content)
        vn_match = re.search(r'versionName\s*=\s*"([^"]+)"', content)
        if not vc_match or not vn_match:
            return GateStepResult(
                name="android_version_bump",
                success=False,
                error="Could not parse versionCode/versionName"
            )

        current_vc = int(vc_match.group(1))
        current_vn = vn_match.group(1)
        new_vc = current_vc + 1
        try:
            new_vn_float = float(current_vn) + 0.001
            new_vn = f"{new_vn_float:.3f}"
        except ValueError:
            return GateStepResult(
                name="android_version_bump",
                success=False,
                error=f"Invalid versionName format: {current_vn}"
            )

        content = re.sub(r"versionCode\s*=\s*\d+", f"versionCode = {new_vc}", content)
        content = re.sub(r'versionName\s*=\s*"[^"]+"', f'versionName = "{new_vn}"', content)

        with open(gradle_path, "w", encoding="utf-8") as f:
            f.write(content)

        duration = time.time() - start
        return GateStepResult(
            name="android_version_bump",
            success=True,
            duration_seconds=duration,
            artifacts=[f"versionCode={new_vc}", f"versionName={new_vn}"]
        )

    def _bump_bridge_version(self, workspace: str) -> GateStepResult:
        gui_path = os.path.join(workspace, "shadow_bridge_gui.py")
        data_service_path = os.path.join(workspace, "web", "services", "data_service.py")
        start = time.time()

        if not os.path.exists(gui_path):
            return GateStepResult(
                name="bridge_version_bump",
                success=False,
                error=f"Missing shadow_bridge_gui.py at {gui_path}"
            )

        with open(gui_path, "r", encoding="utf-8") as f:
            gui_content = f.read()

        match = re.search(r'APP_VERSION\s*=\s*"([^"]+)"', gui_content)
        if not match:
            return GateStepResult(
                name="bridge_version_bump",
                success=False,
                error="Could not find APP_VERSION"
            )

        current = match.group(1)
        try:
            new_ver_float = float(current) + 0.001
            new_ver = f"{new_ver_float:.3f}"
        except ValueError:
            return GateStepResult(
                name="bridge_version_bump",
                success=False,
                error=f"Invalid APP_VERSION format: {current}"
            )

        gui_content = re.sub(
            r'APP_VERSION\s*=\s*"[^"]+"',
            f'APP_VERSION = "{new_ver}"',
            gui_content
        )
        with open(gui_path, "w", encoding="utf-8") as f:
            f.write(gui_content)

        if os.path.exists(data_service_path):
            with open(data_service_path, "r", encoding="utf-8") as f:
                ds_content = f.read()
            ds_content = re.sub(
                r'"version"\s*:\s*"[^"]+"',
                f'"version": "{new_ver}"',
                ds_content
            )
            with open(data_service_path, "w", encoding="utf-8") as f:
                f.write(ds_content)

        duration = time.time() - start
        return GateStepResult(
            name="bridge_version_bump",
            success=True,
            duration_seconds=duration,
            artifacts=[f"version={new_ver}"]
        )
