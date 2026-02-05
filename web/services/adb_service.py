"""
ADB Service for interacting with Android devices.
Enables pushing files and installing APKs.
"""
import subprocess
import os
import logging
import shutil

logger = logging.getLogger(__name__)

# Preferred ADB path (from CLAUDE.md)
DEFAULT_ADB_PATH = r"C:\android\platform-tools\adb.exe"

class AdbService:
    def __init__(self, adb_path=None):
        self.adb_path = adb_path or DEFAULT_ADB_PATH
        if not os.path.exists(self.adb_path):
            # Fallback to system path
            self.adb_path = shutil.which("adb") or "adb"
            
    def _run_command(self, cmd_args):
        """Run an ADB command and return stdout."""
        full_cmd = [self.adb_path] + cmd_args
        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            return result.stdout.strip(), True
        except subprocess.CalledProcessError as e:
            logger.error(f"ADB Command failed: {' '.join(full_cmd)}")
            logger.error(f"Error: {e.stderr}")
            return e.stderr.strip(), False
        except Exception as e:
            logger.error(f"Failed to run ADB command: {e}")
            return str(e), False

    def get_devices(self):
        """List connected devices."""
        output, success = self._run_command(["devices"])
        if not success:
            return []
        
        devices = []
        lines = output.splitlines()
        for line in lines[1:]: # Skip header
            if line.strip() and "device" in line:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "device":
                    devices.append(parts[0])
        return devices

    def push_file(self, local_path, remote_path, device_id=None):
        """Push a file to the device. Returns (output, success)."""
        cmd = []
        if device_id:
            cmd += ["-s", device_id]
        cmd += ["push", local_path, remote_path]

        output, success = self._run_command(cmd)
        return output, success

    def install_apk(self, apk_path, device_id=None, reinstall=True, downgrade=False):
        """Install an APK on the device. Returns (output, success)."""
        cmd = []
        if device_id:
            cmd += ["-s", device_id]
        cmd += ["install"]
        if reinstall:
            cmd.append("-r")
        if downgrade:
            cmd.append("-d")
        cmd.append(apk_path)

        output, success = self._run_command(cmd)
        return output, success

    def shell(self, command, device_id=None):
        """Run a shell command on the device. Returns (output, success)."""
        cmd = []
        if device_id:
            cmd += ["-s", device_id]
        cmd += ["shell", command]

        output, success = self._run_command(cmd)
        return output, success

def get_adb_service():
    return AdbService()
