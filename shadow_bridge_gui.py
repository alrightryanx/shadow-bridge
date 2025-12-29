#!/usr/bin/env python3
"""
ShadowBridge GUI - Desktop Quick Connect Tool
----------------------------------------------
A polished Windows GUI app that helps connect Shadow Android app to this PC.

Features:
- Compact popup-style UI near system tray
- QR code for instant app configuration
- Network discovery for automatic PC finding
- One-click CLI tools installation (Claude Code, Codex, Gemini)

Usage:
    python shadow_bridge_gui.py

Requirements:
    pip install qrcode pillow pystray
"""

import os
import sys
import json
import time
import socket
import platform
import subprocess
import threading
import base64
import webbrowser
import logging
import tempfile
import ctypes

from pathlib import Path
from io import BytesIO

# Import data service for bi-directional sync (web -> Android)
try:
    from web.services.data_service import get_pending_sync_items, mark_items_synced
    SYNC_SERVICE_AVAILABLE = True
except ImportError:
    SYNC_SERVICE_AVAILABLE = False

WEB_SERVER_MODE = "--web-server" in sys.argv


def is_admin():
    """Check if running with administrator privileges."""
    if platform.system() != 'Windows':
        return True  # Non-Windows doesn't need this check
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def run_as_admin():
    """Re-launch the script with administrator privileges."""
    if platform.system() != 'Windows':
        return False

    try:
        # Get the path to Python executable and this script
        if getattr(sys, 'frozen', False):
            # Running as compiled exe
            script = sys.executable
            params = ' '.join(sys.argv[1:])
        else:
            # Running as Python script
            script = sys.executable
            params = ' '.join([f'"{sys.argv[0]}"'] + sys.argv[1:])

        # Use ShellExecuteW to request elevation
        ret = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",  # Request elevation
            script,
            params,
            None,
            1  # SW_SHOWNORMAL
        )

        # If ShellExecuteW returns > 32, it succeeded
        return ret > 32
    except Exception as e:
        print(f"Failed to elevate: {e}")
        return False


# Auto-elevate on Windows if not already admin
if platform.system() == 'Windows' and not is_admin() and not WEB_SERVER_MODE:
    print("ShadowBridge requires administrator privileges for SSH key installation.")
    print("Requesting elevation...")
    if run_as_admin():
        sys.exit(0)  # Exit this instance, elevated one will take over
    else:
        # Failed to elevate (User clicked No or error)
        try:
            ctypes.windll.user32.MessageBoxW(
                0,
                "ShadowBridge requires Administrator privileges to configure SSH keys and network settings.\n\nThe app will now close. Please run as Administrator.",
                "Administrator Required",
                0x10 | 0x40000  # MB_ICONERROR | MB_TOPMOST
            )
        except:
            print("WARNING: Could not elevate. Please restart as Administrator.")
        sys.exit(1)


# Setup logging to file
LOG_DIR = os.path.join(os.environ.get('USERPROFILE', os.path.expanduser('~')), '.shadowai')
LOG_FILE = os.path.join(LOG_DIR, 'shadowbridge.log')
WEB_LOG_FILE = os.path.join(LOG_DIR, 'shadowbridge_web.log')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
    ]
)
log = logging.getLogger('ShadowBridge')

# Platform detection
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    import winreg

# Configuration
DISCOVERY_PORT = 19283
DISCOVERY_MAGIC = b"SHADOWAI_DISCOVER"
DATA_PORT = 19284  # TCP port for receiving project data from Android app
NOTE_CONTENT_PORT = 19285  # TCP port for fetching note content from Android app
COMPANION_PORT = 19286  # TCP port for Claude Code Companion relay
APP_NAME = "ShadowBridge"
APP_VERSION = "1.000"
PROJECTS_FILE = os.path.join(os.environ.get('USERPROFILE', os.path.expanduser('~')), '.shadowai', 'projects.json')
NOTES_FILE = os.path.join(os.environ.get('USERPROFILE', os.path.expanduser('~')), '.shadowai', 'notes.json')
WINDOW_STATE_FILE = os.path.join(os.environ.get('USERPROFILE', os.path.expanduser('~')), '.shadowai', 'window_state.json')

# Theme colors - M3 Dark Theme with Shadow Red
COLORS = {
    # Backgrounds - deeper, richer darks
    'bg_dark': '#0d0d0d',
    'bg_surface': '#141414',
    'bg_card': '#1a1a1a',
    'bg_elevated': '#242424',
    'bg_input': '#1e1e1e',

    # Accent colors - vibrant red
    'accent': '#e53935',
    'accent_hover': '#ff5252',
    'accent_light': '#ff6f60',
    'accent_container': '#3d1a1a',

    # Status colors - softer, M3 style
    'success': '#81c784',
    'success_dim': '#2e4a2f',
    'warning': '#ffb74d',
    'warning_dim': '#4a3a1f',
    'error': '#e57373',
    'error_dim': '#4a2020',

    # Text colors
    'text': '#ece0df',
    'text_secondary': '#b8a8a6',
    'text_dim': '#8a7a78',
    'text_muted': '#5a4a48',

    # Borders and dividers
    'border': '#2a2a2a',
    'border_light': '#3a3a3a',
    'divider': '#1f1f1f'
}

# Enable DPI awareness on Windows BEFORE importing tkinter
if platform.system() == 'Windows':
    try:
        # Windows 10 1607+ Per-Monitor DPI awareness
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            # Fallback for older Windows
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

def set_app_user_model_id(app_id):
    """Set AppUserModelID for proper taskbar grouping and notifications."""
    if not IS_WINDOWS:
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass

def apply_windows_11_theme(root):
    """Enable dark titlebar and optional Mica backdrop on Windows 11."""
    if not IS_WINDOWS:
        return
    try:
        hwnd = root.winfo_id()
        # Dark title bar
        use_dark = ctypes.c_int(1)
        for attr in (20, 19):  # 20 is Win10 20H1+, 19 is older fallback
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(use_dark), ctypes.sizeof(use_dark)
            )
        # Mica backdrop (Windows 11)
        backdrop = ctypes.c_int(2)  # DWMSBT_MAINWINDOW
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 38, ctypes.byref(backdrop), ctypes.sizeof(backdrop)
        )
    except Exception:
        pass

# Try to import optional dependencies
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False
    print("Error: tkinter not available")
    sys.exit(1)

# Try to import sv_ttk for modern Windows theming
try:
    import sv_ttk
    HAS_SV_TTK = True
except ImportError:
    HAS_SV_TTK = False

try:
    import qrcode
    HAS_QRCODE = True
except ImportError:
    HAS_QRCODE = False

try:
    from PIL import Image, ImageTk, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pystray
    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False


def get_all_ips():
    """Get all available IP addresses from all network interfaces."""
    ips = {
        'local': [],      # Private LAN IPs (192.168.x.x, 10.x.x.x, 172.16-31.x.x)
        'tailscale': [],  # Tailscale IPs (100.x.x.x)
        'vpn': [],        # Common VPN ranges
        'other': []       # Other IPs
    }

    try:
        # Get all IPs from hostname
        hostname = socket.gethostname()
        all_ips = socket.getaddrinfo(hostname, None, socket.AF_INET)

        for item in all_ips:
            ip = item[4][0]
            if ip.startswith('127.'):
                continue  # Skip loopback

            # Categorize IP
            if ip.startswith('100.64.') or ip.startswith('100.'):
                ips['tailscale'].append(ip)
            elif ip.startswith('10.') and (ip.startswith('10.8.') or ip.startswith('10.9.')):
                ips['vpn'].append(ip)  # Common OpenVPN range
            elif ip.startswith('192.168.') or ip.startswith('10.') or ip.startswith('172.'):
                ips['local'].append(ip)
            else:
                ips['other'].append(ip)
    except Exception:
        pass

    # Also try the socket method for primary IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        primary_ip = s.getsockname()[0]
        s.close()

        # Add to appropriate category if not already there
        found = False
        for category in ips.values():
            if primary_ip in category:
                found = True
                break
        if not found and not primary_ip.startswith('127.'):
            if primary_ip.startswith('192.168.') or primary_ip.startswith('10.') or primary_ip.startswith('172.'):
                ips['local'].insert(0, primary_ip)
            else:
                ips['other'].insert(0, primary_ip)
    except Exception:
        pass

    return ips


def get_local_ip():
    """Get primary local IP address."""
    all_ips = get_all_ips()
    if all_ips['local']:
        return all_ips['local'][0]
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_tailscale_ip():
    """Get Tailscale IP if available (stable 100.x.x.x address)."""
    # First check our detected IPs
    all_ips = get_all_ips()
    if all_ips['tailscale']:
        return all_ips['tailscale'][0]

    # Try tailscale CLI
    try:
        result = subprocess.run(
            "tailscale ip -4",
            shell=True,
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
        )
        if result.returncode == 0:
            ip = result.stdout.strip().split('\n')[0]
            if ip.startswith('100.'):
                return ip
    except Exception:
        pass
    return None


def get_hostname_local():
    """Get hostname.local for mDNS resolution."""
    try:
        hostname = socket.gethostname()
        return f"{hostname}.local"
    except Exception:
        return None


def get_zerotier_ip():
    """Get ZeroTier IP if available."""
    all_ips = get_all_ips()
    # ZeroTier typically uses 10.x.x.x range configured by network
    # Check for zerotier-cli
    try:
        result = subprocess.run(
            "zerotier-cli listnetworks",
            shell=True,
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
        )
        if result.returncode == 0 and '200' in result.stdout:
            # ZeroTier is running, IPs would be in 'other' or 'local'
            return True
    except Exception:
        pass
    return None


def get_username():
    """Get current username."""
    try:
        return os.getlogin()
    except Exception:
        import getpass
        return getpass.getuser()


def find_ssh_port():
    """Find which port SSH is running on by checking common ports and netstat."""
    # First try netstat to find sshd
    try:
        result = subprocess.run(
            'netstat -an | findstr "LISTENING" | findstr ":22"',
            shell=True, capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
        )
        # Parse netstat output to find SSH ports
        for line in result.stdout.splitlines():
            parts = line.split()
            for part in parts:
                if ':' in part:
                    try:
                        port = int(part.split(':')[-1])
                        if port > 0:
                            # Verify it's actually listening
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            sock.settimeout(0.5)
                            if sock.connect_ex(('127.0.0.1', port)) == 0:
                                sock.close()
                                return port
                            sock.close()
                    except (ValueError, socket.error):
                        continue
    except Exception:
        pass

    # Check sshd_config for Port setting
    sshd_config_paths = [
        r"C:\ProgramData\ssh\sshd_config",
        r"C:\Windows\System32\OpenSSH\sshd_config",
        os.path.expanduser("~/.ssh/sshd_config"),
    ]
    for config_path in sshd_config_paths:
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('Port ') and not line.startswith('#'):
                        port = int(line.split()[1])
                        return port
        except Exception:
            continue

    # Try common SSH ports
    common_ports = [22, 2222, 2269, 22022, 8022]
    for port in common_ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        try:
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                sock.close()
                return port
        except Exception:
            pass
        finally:
            sock.close()

    return None  # No SSH found


def check_ssh_running(port=22):
    """Check if SSH server is running on specific port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        result = sock.connect_ex(('127.0.0.1', port))
        return result == 0
    except Exception:
        return False
    finally:
        sock.close()


def setup_firewall_rule():
    """Add Windows Firewall rule for UDP discovery port."""
    if not IS_WINDOWS:
        return True
    try:
        # Check if rule exists
        check = subprocess.run(
            ['netsh', 'advfirewall', 'firewall', 'show', 'rule', 'name=ShadowBridge Discovery'],
            capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        if 'ShadowBridge Discovery' in check.stdout:
            return True  # Rule already exists

        # Add inbound UDP rule
        result = subprocess.run(
            ['netsh', 'advfirewall', 'firewall', 'add', 'rule',
             'name=ShadowBridge Discovery', 'dir=in', 'action=allow',
             'protocol=UDP', 'localport=19283'],
            capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        return result.returncode == 0
    except Exception:
        return False


def check_npm_installed():
    """Check if npm is installed."""
    try:
        result = subprocess.run(
            "npm --version",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
        )
        return result.returncode == 0
    except Exception:
        return False


def ensure_windows_user_path_contains(entry):
    """Ensure the current user's PATH contains the given entry (Windows only)."""
    if not IS_WINDOWS:
        return False
    try:
        entry = os.path.normpath(entry)
        env_key_path = r"Environment"
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, env_key_path, 0, winreg.KEY_READ | winreg.KEY_SET_VALUE)
        try:
            try:
                current_value, current_type = winreg.QueryValueEx(key, "Path")
            except FileNotFoundError:
                current_value, current_type = "", winreg.REG_EXPAND_SZ

            current_value = current_value or ""
            parts = [p.strip() for p in current_value.split(";") if p.strip()]
            normalized_parts = {os.path.normpath(p) for p in parts}
            if entry in normalized_parts:
                # Still ensure current process env PATH is up to date.
                try:
                    proc_path = os.environ.get("PATH", "") or ""
                    proc_parts = {os.path.normpath(p.strip()) for p in proc_path.split(";") if p.strip()}
                    if entry not in proc_parts:
                        os.environ["PATH"] = (proc_path + ";" + entry) if proc_path else entry
                except Exception:
                    pass
                return True

            parts.append(entry)
            new_value = ";".join(parts)
            # Preserve existing registry type when possible.
            winreg.SetValueEx(key, "Path", 0, current_type, new_value)
            log.info(f"Added to user PATH: {entry}")

            # Update current process PATH so tools launched from ShadowBridge work immediately.
            try:
                proc_path = os.environ.get("PATH", "") or ""
                os.environ["PATH"] = (proc_path + ";" + entry) if proc_path else entry
            except Exception:
                pass

            # Broadcast environment change so new shells can pick up PATH without logoff.
            try:
                HWND_BROADCAST = 0xFFFF
                WM_SETTINGCHANGE = 0x001A
                SMTO_ABORTIFHUNG = 0x0002
                ctypes.windll.user32.SendMessageTimeoutW(
                    HWND_BROADCAST,
                    WM_SETTINGCHANGE,
                    0,
                    "Environment",
                    SMTO_ABORTIFHUNG,
                    2000,
                    None
                )
            except Exception as e:
                log.debug(f"Failed to broadcast env change: {e}")

            return True
        finally:
            winreg.CloseKey(key)
    except Exception as e:
        log.warning(f"Failed to update user PATH: {e}")
        return False


def check_winget_installed():
    """Check if winget is installed (Windows Package Manager)."""
    if not IS_WINDOWS:
        return False
    try:
        result = subprocess.run(
            "winget --version",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        return result.returncode == 0
    except Exception:
        return False


def check_python_pip_installed():
    """Check if pip is available via either the Windows 'py' launcher or 'python'."""
    candidates = [
        "py -m pip --version",
        "python -m pip --version",
    ]
    flags = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
    for cmd in candidates:
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=flags
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass
    return False


def check_tool_installed(tool_name):
    """Check if a CLI tool is installed."""
    # Map tool IDs to actual command names
    tool_commands = {
        'claude': ['claude', 'claude-code'],
        'codex': ['codex'],
        'gemini': ['gemini', 'gemini-cli'],
        'aider': ['aider'],
        'ollama': ['ollama'],
    }

    commands_to_try = tool_commands.get(tool_name, [tool_name])

    # Check npm global directories directly on Windows
    if IS_WINDOWS:
        npm_paths = [
            os.path.join(os.environ.get('APPDATA', ''), 'npm'),
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'npm'),
            os.path.join(os.environ.get('USERPROFILE', ''), 'AppData', 'Roaming', 'npm'),
        ]
        for cmd in commands_to_try:
            for npm_path in npm_paths:
                # Check for .cmd file (Windows npm installs)
                cmd_path = os.path.join(npm_path, f'{cmd}.cmd')
                if os.path.isfile(cmd_path):
                    return True
                # Check for .ps1 file
                ps1_path = os.path.join(npm_path, f'{cmd}.ps1')
                if os.path.isfile(ps1_path):
                    return True
                # Check for plain executable
                exe_path = os.path.join(npm_path, f'{cmd}.exe')
                if os.path.isfile(exe_path):
                    return True

    # Try 'where' command on Windows
    for cmd in commands_to_try:
        if IS_WINDOWS:
            try:
                result = subprocess.run(
                    f'where {cmd}',
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                if result.returncode == 0 and result.stdout.strip():
                    return True
            except Exception:
                pass

        # Try running with --version
        try:
            result = subprocess.run(
                f'{cmd} --version',
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass

    return False


def get_startup_registry_key():
    """Get the Windows startup registry key."""
    return r"Software\Microsoft\Windows\CurrentVersion\Run"


def is_startup_enabled():
    """Check if app is set to start with Windows."""
    if not IS_WINDOWS:
        return False
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, get_startup_registry_key(), 0, winreg.KEY_READ)
        try:
            winreg.QueryValueEx(key, APP_NAME)
            return True
        except FileNotFoundError:
            return False
        finally:
            winreg.CloseKey(key)
    except Exception:
        return False


def set_startup_enabled(enabled):
    """Enable or disable starting with Windows."""
    if not IS_WINDOWS:
        return False
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, get_startup_registry_key(), 0, winreg.KEY_SET_VALUE)
        try:
            if enabled:
                exe_path = sys.executable if getattr(sys, 'frozen', False) else None
                if exe_path:
                    winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, f'"{exe_path}" --minimized')
                    return True
                return False
            else:
                try:
                    winreg.DeleteValue(key, APP_NAME)
                except FileNotFoundError:
                    pass
                return True
        finally:
            winreg.CloseKey(key)
    except Exception:
        return False


def get_app_icon_path():
    """Get the path to the app icon."""
    # Check various locations for logo.png
    possible_paths = [
        Path(__file__).parent / "logo.png",
        Path(sys.executable).parent / "logo.png" if getattr(sys, 'frozen', False) else None,
        Path("logo.png"),
    ]
    for p in possible_paths:
        if p and p.exists():
            return str(p)
    return None


def create_app_icon(size=64):
    """Load app icon from logo.png or create fallback."""
    if not HAS_PIL:
        return None

    # Try to load logo.png
    icon_path = get_app_icon_path()
    if icon_path:
        try:
            img = Image.open(icon_path)
            img = img.resize((size, size), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            return img.convert('RGBA')
        except Exception:
            pass

    # Fallback: create programmatic icon
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([2, 2, size-2, size-2], fill='#1a0808', outline='#e53935', width=2)
    center = size // 2
    points = [
        (center + 8, center - 18), (center - 4, center - 2),
        (center + 4, center - 2), (center - 8, center + 18),
        (center + 4, center + 2), (center - 4, center + 2),
    ]
    draw.polygon(points, fill='#e53935')
    return img


class DataReceiver(threading.Thread):
    """TCP server that receives project and notes data from Android app.

    Security Features:
    - Rate limiting: Max 5 connection attempts per IP per minute
    - Known device tracking: Only known devices can sync without approval
    - SSH key approval: New device keys require user confirmation
    """

    # Rate limiting settings
    RATE_LIMIT_WINDOW = 60  # seconds
    RATE_LIMIT_MAX = 5  # max attempts per window

    def __init__(self, on_data_received, on_device_connected, on_notes_received=None, on_key_approval_needed=None):
        super().__init__(daemon=True)
        self.on_data_received = on_data_received
        self.on_device_connected = on_device_connected
        self.on_notes_received = on_notes_received
        self.on_key_approval_needed = on_key_approval_needed  # Callback for key approval UI
        self.running = True
        self.sock = None
        self.connected_devices = {}  # device_id -> {name, ip, last_seen}
        self.ip_to_device_id = {}  # ip -> device_id
        self._devices_lock = threading.Lock()
        self._storage_lock = threading.Lock()

        # Security: Rate limiting
        self._rate_limit_tracker = {}  # ip -> [timestamps]
        self._rate_limit_lock = threading.Lock()

        # Security: Pending key approvals
        self._pending_keys = {}  # device_id -> {public_key, device_name, ip, timestamp}
        self._approved_devices = set()  # Set of approved device_ids
        self._load_approved_devices()

    def _load_approved_devices(self):
        """Load list of approved device IDs from disk."""
        try:
            approved_file = os.path.join(os.environ.get('USERPROFILE', os.path.expanduser('~')),
                                         '.shadowai', 'approved_devices.json')
            if os.path.exists(approved_file):
                with open(approved_file, 'r') as f:
                    data = json.load(f)
                    self._approved_devices = set(data.get('approved', []))
                    log.info(f"Loaded {len(self._approved_devices)} approved devices")
        except Exception as e:
            log.warning(f"Could not load approved devices: {e}")
            self._approved_devices = set()

    def _save_approved_devices(self):
        """Save list of approved device IDs to disk."""
        try:
            approved_file = os.path.join(os.environ.get('USERPROFILE', os.path.expanduser('~')),
                                         '.shadowai', 'approved_devices.json')
            os.makedirs(os.path.dirname(approved_file), exist_ok=True)
            with open(approved_file, 'w') as f:
                json.dump({'approved': list(self._approved_devices)}, f, indent=2)
        except Exception as e:
            log.warning(f"Could not save approved devices: {e}")

    def approve_device(self, device_id):
        """Approve a device for SSH key installation."""
        self._approved_devices.add(device_id)
        self._save_approved_devices()
        log.info(f"Device approved: {device_id}")

        # If there's a pending key for this device, install it now
        if device_id in self._pending_keys:
            pending = self._pending_keys.pop(device_id)
            result = self._install_ssh_key(pending['public_key'], pending['device_name'])
            log.info(f"Installed pending key for {device_id}: {result}")
            return result
        return {'success': True, 'message': 'Device approved'}

    def reject_device(self, device_id):
        """Reject a pending device key."""
        if device_id in self._pending_keys:
            del self._pending_keys[device_id]
            log.info(f"Device key rejected: {device_id}")

    def get_pending_keys(self):
        """Get list of pending key approvals."""
        return dict(self._pending_keys)

    def _check_rate_limit(self, ip):
        """Check if IP is rate limited. Returns True if request is allowed."""
        # Skip rate limiting for localhost (internal web dashboard connections)
        if ip in ('127.0.0.1', 'localhost', '::1'):
            return True

        now = time.time()
        with self._rate_limit_lock:
            if ip not in self._rate_limit_tracker:
                self._rate_limit_tracker[ip] = []

            # Remove old timestamps outside the window
            self._rate_limit_tracker[ip] = [
                ts for ts in self._rate_limit_tracker[ip]
                if now - ts < self.RATE_LIMIT_WINDOW
            ]

            # Check if over limit
            if len(self._rate_limit_tracker[ip]) >= self.RATE_LIMIT_MAX:
                log.warning(f"Rate limit exceeded for {ip}")
                return False

            # Record this attempt
            self._rate_limit_tracker[ip].append(now)
            return True

    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind(('', DATA_PORT))
            self.sock.listen(5)
            self.sock.settimeout(1.0)
            while self.running:
                try:
                    conn, addr = self.sock.accept()
                    threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()
                except socket.timeout:
                    continue
                except Exception:
                    pass
        finally:
            if self.sock:
                self.sock.close()

    def _handle_client(self, conn, addr):
        """Handle incoming data from Android app."""
        try:
            ip = addr[0]
            log.info(f"Client connected from {addr}")

            # Security: Rate limiting
            if not self._check_rate_limit(ip):
                log.warning(f"Rate limited connection from {ip}")
                self._send_response(conn, {'success': False, 'message': 'Rate limited. Try again later.'})
                return

            conn.settimeout(10)

            # Read length-prefixed message (4 bytes big-endian length + data)
            # Must read exactly 4 bytes for length prefix
            length_bytes = b''
            while len(length_bytes) < 4:
                chunk = conn.recv(4 - len(length_bytes))
                if not chunk:
                    log.error(f"Connection closed while reading length prefix")
                    return
                length_bytes += chunk

            log.debug(f"Received length bytes: {len(length_bytes)}, value: {int.from_bytes(length_bytes, 'big')}")

            msg_length = int.from_bytes(length_bytes, 'big')
            if msg_length > 1024 * 1024:  # 1MB limit
                log.error(f"Message too large: {msg_length}")
                return

            data = b''
            while len(data) < msg_length:
                chunk = conn.recv(min(4096, msg_length - len(data)))
                if not chunk:
                    log.error(f"Connection closed while reading message body")
                    break
                data += chunk

            log.debug(f"Received {len(data)} bytes of message data")

            if data:
                try:
                    payload = json.loads(data.decode('utf-8'))
                    action = payload.get('action', '')
                    ip = addr[0]
                    raw_device_id = payload.get('device_id', None)
                    device_id = raw_device_id or self.ip_to_device_id.get(ip) or ip

                    with self._devices_lock:
                        if raw_device_id:
                            self.ip_to_device_id[ip] = raw_device_id

                        existing = self.connected_devices.get(device_id)
                        device_name = payload.get('device_name', None)
                        if device_name is None:
                            device_name = existing.get('name') if existing else ip

                        # Merge legacy entry keyed by IP into stable device_id if needed.
                        if raw_device_id and raw_device_id != ip and ip in self.connected_devices and device_id != ip:
                            legacy = self.connected_devices.pop(ip, {})
                            if not existing:
                                existing = legacy

                        self.connected_devices[device_id] = {
                            'name': device_name,
                            'ip': ip,
                            'last_seen': time.time()
                        }

                    # If we learned a stable device_id after having only an IP-based id,
                    # migrate any previously saved projects to keep everything under one device.
                    if raw_device_id and raw_device_id != ip and device_id != ip:
                        self._migrate_saved_device(old_device_id=ip, new_device_id=device_id, device_name=device_name, ip=ip)

                    if self.on_device_connected:
                        self.on_device_connected(device_id, device_name, ip)

                    # Handle different actions
                    log.info(f"Action: {action}, Device: {device_name}")
                    if action == 'install_key':
                        # SSH key installation request - SECURITY: Requires approval for new devices
                        public_key = payload.get('public_key', '')

                        # Check if device is already approved
                        if device_id in self._approved_devices:
                            log.info(f"Installing SSH key for approved device: {device_name}")
                            response = self._install_ssh_key(public_key, device_name)
                            log.info(f"Install response: {response}")
                            self._send_response(conn, response)
                        else:
                            # Queue for approval
                            log.info(f"Queueing SSH key for approval: {device_name} ({device_id})")
                            self._pending_keys[device_id] = {
                                'public_key': public_key,
                                'device_name': device_name,
                                'ip': ip,
                                'timestamp': time.time()
                            }
                            # Notify UI for approval
                            if self.on_key_approval_needed:
                                self.on_key_approval_needed(device_id, device_name, ip)
                            response = {
                                'success': False,
                                'pending': True,
                                'message': 'SSH key queued for approval. Please approve on the PC.'
                            }
                            self._send_response(conn, response)
                        log.info(f"Response sent to {addr}")
                    elif 'projects' in payload:
                        # Handle projects data
                        ip_candidates = payload.get('ip_candidates')
                        note_content_port = payload.get('note_content_port')
                        self._save_projects(device_id, device_name, ip, payload['projects'], ip_candidates, note_content_port)
                        if self.on_data_received:
                            self.on_data_received(device_id, payload['projects'])

                        # Bi-directional sync: Include pending items from web
                        response = {'success': True, 'message': 'Projects synced'}
                        if SYNC_SERVICE_AVAILABLE:
                            pending = get_pending_sync_items(device_id)
                            if pending.get('projects'):
                                response['sync_to_device'] = {'projects': pending['projects']}
                                log.info(f"Including {len(pending['projects'])} pending projects for sync to device")
                        self._send_response(conn, response)
                    elif 'notes' in payload:
                        # Handle notes data (titles only, content fetched on-demand)
                        ip_candidates = payload.get('ip_candidates')
                        note_content_port = payload.get('note_content_port')
                        self._save_notes(device_id, device_name, ip, payload['notes'], ip_candidates, note_content_port)
                        if self.on_notes_received:
                            self.on_notes_received(device_id, payload['notes'])

                        # Bi-directional sync: Include pending items from web
                        response = {'success': True, 'message': 'Notes synced'}
                        if SYNC_SERVICE_AVAILABLE:
                            pending = get_pending_sync_items(device_id)
                            if pending.get('notes'):
                                response['sync_to_device'] = {'notes': pending['notes']}
                                log.info(f"Including {len(pending['notes'])} pending notes for sync to device")
                        self._send_response(conn, response)
                    elif action == 'sync_confirm':
                        # Android confirming it received and saved web-created items
                        if SYNC_SERVICE_AVAILABLE:
                            synced_projects = payload.get('synced_projects', [])
                            synced_notes = payload.get('synced_notes', [])
                            synced_automations = payload.get('synced_automations', [])
                            if synced_projects:
                                mark_items_synced(device_id, 'projects', synced_projects)
                                log.info(f"Marked {len(synced_projects)} projects as synced")
                            if synced_notes:
                                mark_items_synced(device_id, 'notes', synced_notes)
                                log.info(f"Marked {len(synced_notes)} notes as synced")
                            if synced_automations:
                                mark_items_synced(device_id, 'automations', synced_automations)
                                log.info(f"Marked {len(synced_automations)} automations as synced")
                        self._send_response(conn, {'success': True, 'message': 'Sync confirmed'})
                    else:
                        self._send_response(conn, {'success': True, 'message': 'OK'})

                except json.JSONDecodeError:
                    self._send_response(conn, {'success': False, 'message': 'Invalid JSON'})
        except Exception as e:
            try:
                self._send_response(conn, {'success': False, 'message': str(e)})
            except:
                pass
        finally:
            conn.close()

    def _send_response(self, conn, response):
        """Send length-prefixed JSON response."""
        try:
            data = json.dumps(response).encode('utf-8')
            conn.send(len(data).to_bytes(4, 'big'))
            conn.send(data)
        except:
            pass

    def _install_ssh_key(self, public_key, device_name):
        """Install SSH public key to authorized_keys (both user and admin on Windows)."""
        try:
            if not public_key or not public_key.startswith('ssh-'):
                return {'success': False, 'message': 'Invalid public key format'}

            # List of authorized_keys files to update
            auth_keys_files = []

            # User's ~/.ssh/authorized_keys
            home_dir = os.path.expanduser('~')
            ssh_dir = os.path.join(home_dir, '.ssh')
            user_auth_keys = os.path.join(ssh_dir, 'authorized_keys')
            auth_keys_files.append(user_auth_keys)

            # On Windows, also update administrators_authorized_keys for admin users
            if platform.system() == 'Windows':
                admin_auth_keys = r'C:\ProgramData\ssh\administrators_authorized_keys'
                if os.path.exists(r'C:\ProgramData\ssh'):
                    auth_keys_files.append(admin_auth_keys)
                    log.info(f"Windows detected: will also update {admin_auth_keys}")

            # Create .ssh directory if it doesn't exist
            if not os.path.exists(ssh_dir):
                os.makedirs(ssh_dir, mode=0o700)

            installed_count = 0
            for auth_keys_file in auth_keys_files:
                try:
                    # Read existing authorized_keys
                    existing_keys = set()
                    if os.path.exists(auth_keys_file):
                        with open(auth_keys_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    # Extract just the key part (type + base64) for comparison
                                    parts = line.split()
                                    if len(parts) >= 2:
                                        existing_keys.add(f"{parts[0]} {parts[1]}")

                    # Check if key already exists
                    key_parts = public_key.split()
                    if len(key_parts) >= 2:
                        key_fingerprint = f"{key_parts[0]} {key_parts[1]}"
                        if key_fingerprint in existing_keys:
                            log.info(f"Key already exists in {auth_keys_file}")
                            installed_count += 1
                            continue

                    # Append new key with comment
                    with open(auth_keys_file, 'a') as f:
                        # Add newline if file doesn't end with one
                        if os.path.exists(auth_keys_file) and os.path.getsize(auth_keys_file) > 0:
                            with open(auth_keys_file, 'rb') as rf:
                                rf.seek(-1, 2)
                                if rf.read(1) != b'\n':
                                    f.write('\n')
                        f.write(f"# Shadow device: {device_name} - added {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"{public_key}\n")

                    log.info(f"Key installed to {auth_keys_file}")
                    installed_count += 1

                    # Set correct permissions (non-Windows)
                    if platform.system() != 'Windows':
                        try:
                            os.chmod(auth_keys_file, 0o600)
                        except:
                            pass

                except Exception as e:
                    log.warning(f"Failed to update {auth_keys_file}: {e}")

            if installed_count > 0:
                return {'success': True, 'message': f'SSH key installed for {device_name} ({installed_count} files)'}
            else:
                return {'success': False, 'message': 'Failed to install key to any authorized_keys file'}

        except Exception as e:
            return {'success': False, 'message': f'Failed to install key: {str(e)}'}

    def _save_projects(self, device_id, device_name, ip, projects, ip_candidates=None, note_content_port=None):
        """Save per-device projects to local file for persistence.

        Also cleans up stale device entries that share the same base fingerprint
        (e.g., old entries without package name suffix).
        """
        try:
            if not isinstance(projects, list):
                return

            cleaned = []
            for project in projects:
                if not isinstance(project, dict):
                    continue
                name = project.get('name') or project.get('title') or 'Unnamed'
                path = (
                    project.get('path')
                    or project.get('workingDirectory')
                    or project.get('working_directory')
                    or project.get('dir')
                    or ''
                )
                cleaned.append({
                    'id': project.get('id') or project.get('projectId') or project.get('project_id'),
                    'name': str(name),
                    'path': str(path) if path is not None else ''
                })

            with self._storage_lock:
                state = load_projects_state()
                devices = state.get('devices', {})

                # Extract base fingerprint for cleanup
                base_fingerprint = device_id.split(':com.')[0] if ':com.' in device_id else None

                # Clean up stale device entries with same base fingerprint
                if base_fingerprint:
                    stale_ids = [
                        did for did in devices.keys()
                        if did != device_id and (
                            did == base_fingerprint or  # Old format (no package suffix)
                            did.startswith(base_fingerprint + ':com.')  # Different package
                        )
                    ]
                    for stale_id in stale_ids:
                        log.info(f"Removing stale device entry from projects: {stale_id}")
                        del devices[stale_id]

                device = devices.get(device_id, {})
                device.update({
                    'id': device_id,
                    'name': device_name,
                    'ip': ip,
                    'last_seen': time.time(),
                    'projects': cleaned
                })
                if isinstance(note_content_port, int) and 1 <= note_content_port <= 65535:
                    device['note_content_port'] = note_content_port
                merged_candidates = []
                existing_candidates = device.get('ip_candidates', [])
                if isinstance(ip_candidates, list):
                    for candidate in ip_candidates:
                        if isinstance(candidate, str) and candidate:
                            merged_candidates.append(candidate)
                if isinstance(existing_candidates, list):
                    for candidate in existing_candidates:
                        if isinstance(candidate, str) and candidate:
                            merged_candidates.append(candidate)
                merged_candidates = [
                    candidate for candidate in merged_candidates
                    if candidate and candidate != ip
                ]
                deduped = []
                for candidate in merged_candidates:
                    if candidate not in deduped:
                        deduped.append(candidate)
                if deduped:
                    device['ip_candidates'] = deduped
                else:
                    device.pop('ip_candidates', None)
                devices[device_id] = device

                save_projects_state({'devices': devices})
                log.info(f"Saved {len(cleaned)} projects for {device_name} ({device_id})")
        except Exception:
            log.exception("Failed to save projects")

    def _save_notes(self, device_id, device_name, ip, notes, ip_candidates=None, note_content_port=None):
        """Save per-device notes (titles only) to local file for persistence.

        Also cleans up stale device entries that share the same base fingerprint
        (e.g., old entries without package name suffix).
        """
        try:
            if not isinstance(notes, list):
                return

            cleaned = []
            for note in notes:
                if not isinstance(note, dict):
                    continue
                note_data = {
                    'id': note.get('id', ''),
                    'title': note.get('title', 'Untitled'),
                    'updatedAt': note.get('updatedAt', 0)
                }
                # Include content if provided (for offline access)
                if note.get('content'):
                    note_data['content'] = note.get('content')
                cleaned.append(note_data)

            with self._storage_lock:
                state = load_notes_state()
                devices = state.get('devices', {})

                # Extract base fingerprint for cleanup
                base_fingerprint = device_id.split(':com.')[0] if ':com.' in device_id else None

                # Clean up stale device entries with same base fingerprint
                if base_fingerprint:
                    stale_ids = [
                        did for did in devices.keys()
                        if did != device_id and (
                            did == base_fingerprint or  # Old format (no package suffix)
                            did.startswith(base_fingerprint + ':com.')  # Different package
                        )
                    ]
                    for stale_id in stale_ids:
                        log.info(f"Removing stale device entry: {stale_id}")
                        del devices[stale_id]

                device = devices.get(device_id, {})
                device.update({
                    'id': device_id,
                    'name': device_name,
                    'ip': ip,
                    'last_seen': time.time(),
                    'notes': cleaned
                })
                if isinstance(note_content_port, int) and 1 <= note_content_port <= 65535:
                    device['note_content_port'] = note_content_port
                merged_candidates = []
                existing_candidates = device.get('ip_candidates', [])
                if isinstance(ip_candidates, list):
                    for candidate in ip_candidates:
                        if isinstance(candidate, str) and candidate:
                            merged_candidates.append(candidate)
                if isinstance(existing_candidates, list):
                    for candidate in existing_candidates:
                        if isinstance(candidate, str) and candidate:
                            merged_candidates.append(candidate)
                merged_candidates = [
                    candidate for candidate in merged_candidates
                    if candidate and candidate != ip
                ]
                deduped = []
                for candidate in merged_candidates:
                    if candidate not in deduped:
                        deduped.append(candidate)
                if deduped:
                    device['ip_candidates'] = deduped
                else:
                    device.pop('ip_candidates', None)
                devices[device_id] = device

                save_notes_state({'devices': devices})
                log.info(f"Saved {len(cleaned)} notes for {device_name} ({device_id})")
        except Exception:
            log.exception("Failed to save notes")

    def _migrate_saved_device(self, old_device_id, new_device_id, device_name, ip):
        """Migrate persisted projects from an IP-based id to a stable device id."""
        try:
            with self._storage_lock:
                state = load_projects_state()
                devices = state.get('devices', {})

                old = devices.get(old_device_id)
                new = devices.get(new_device_id)

                if isinstance(old, dict) and not isinstance(new, dict):
                    devices[new_device_id] = dict(old)
                    devices.pop(old_device_id, None)
                elif isinstance(old, dict) and isinstance(new, dict):
                    # Merge projects if needed
                    if not new.get('projects') and old.get('projects'):
                        new['projects'] = old.get('projects')
                    devices.pop(old_device_id, None)
                    devices[new_device_id] = new

                if isinstance(devices.get(new_device_id), dict):
                    devices[new_device_id].update({
                        'id': new_device_id,
                        'name': device_name,
                        'ip': ip,
                        'last_seen': time.time()
                    })

                save_projects_state({'devices': devices})
        except Exception:
            pass

    def stop(self):
        self.running = False

    def get_connected_devices(self):
        """Get list of recently connected devices (within last 5 minutes)."""
        now = time.time()
        active = {}
        with self._devices_lock:
            for did, info in list(self.connected_devices.items()):
                if now - info['last_seen'] < 300:  # 5 minutes
                    active[did] = info
        return active


def load_projects_state():
    """Load per-device projects state from local file."""
    try:
        if os.path.exists(PROJECTS_FILE):
            with open(PROJECTS_FILE, 'r') as f:
                data = json.load(f)

                # v2 schema (multi-device)
                if isinstance(data, dict) and isinstance(data.get('devices'), dict):
                    devices = {str(k): v for k, v in data.get('devices', {}).items() if isinstance(v, dict)}
                    return {
                        'version': int(data.get('version', 2)),
                        'updated': float(data.get('updated', 0)),
                        'devices': devices
                    }

                # v1 schema (single list of projects)
                legacy_projects = data.get('projects', []) if isinstance(data, dict) else []
                return {
                    'version': 1,
                    'updated': float(data.get('updated', 0)) if isinstance(data, dict) else 0.0,
                    'devices': {
                        'legacy': {
                            'id': 'legacy',
                            'name': 'Legacy',
                            'ip': None,
                            'last_seen': float(data.get('updated', 0)) if isinstance(data, dict) else 0.0,
                            'projects': legacy_projects if isinstance(legacy_projects, list) else []
                        }
                    }
                }
    except Exception:
        pass

    return {'version': 2, 'updated': 0.0, 'devices': {}}


def save_projects_state(state):
    """Persist per-device projects state to local file."""
    try:
        os.makedirs(os.path.dirname(PROJECTS_FILE), exist_ok=True)
        payload = {
            'version': 2,
            'updated': time.time(),
            'devices': state.get('devices', {})
        }
        with open(PROJECTS_FILE, 'w') as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def load_notes_state():
    """Load per-device notes state from local file."""
    try:
        if os.path.exists(NOTES_FILE):
            with open(NOTES_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get('devices'), dict):
                    devices = {str(k): v for k, v in data.get('devices', {}).items() if isinstance(v, dict)}
                    return {
                        'version': int(data.get('version', 1)),
                        'updated': float(data.get('updated', 0)),
                        'devices': devices
                    }
    except Exception:
        pass
    return {'version': 1, 'updated': 0.0, 'devices': {}}


def save_notes_state(state):
    """Persist per-device notes state to local file."""
    try:
        os.makedirs(os.path.dirname(NOTES_FILE), exist_ok=True)
        payload = {
            'version': 1,
            'updated': time.time(),
            'devices': state.get('devices', {})
        }
        with open(NOTES_FILE, 'w') as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def get_note_content_from_cache(note_id):
    """Get note content from local cache if available."""
    try:
        state = load_notes_state()
        devices = state.get('devices', {})
        for device_id, device_info in devices.items():
            for note in device_info.get('notes', []):
                if note.get('id') == note_id:
                    content = note.get('content')
                    if content:
                        return {
                            'id': note_id,
                            'title': note.get('title', 'Untitled'),
                            'content': content,
                            'updatedAt': note.get('updatedAt', 0)
                        }
    except Exception:
        pass
    return None


def is_pc_path(path):
    """Check if path is a PC path (starts with drive letter)."""
    if not path:
        return False
    # Windows drive letter pattern (C:\, D:\, etc.)
    if len(path) >= 2 and path[1] == ':':
        return True
    # Unix absolute path
    if path.startswith('/') and not path.startswith('/storage/'):
        return True
    return False


def open_folder(path):
    """Open folder in file explorer."""
    try:
        if IS_WINDOWS:
            os.startfile(path)
        elif platform.system() == 'Darwin':
            subprocess.run(['open', path])
        else:
            subprocess.run(['xdg-open', path])
    except Exception:
        pass


class CompanionRelayServer(threading.Thread):
    """TCP server that relays messages between Claude Code plugin and Android app.

    This enables the Claude Code Companion feature:
    - Plugin sends approval requests, session events to Android
    - Android sends approval responses, user replies back to plugin
    - Clipboard sync for reply injection
    """

    def __init__(self, on_status_change=None):
        super().__init__(daemon=True)
        self.running = True
        self.sock = None
        self.on_status_change = on_status_change

        # Connected clients
        self._plugin_conn = None  # Connection from Claude Code plugin
        self._device_conns = {}   # device_id -> connection
        self._conns_lock = threading.Lock()

        # Pending messages for offline devices
        self._pending_messages = {}  # device_id -> [messages]
        self._pending_lock = threading.Lock()

    def run(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(('0.0.0.0', COMPANION_PORT))
            self.sock.listen(5)
            self.sock.settimeout(1.0)

            log.info(f"Companion relay server listening on port {COMPANION_PORT}")
            if self.on_status_change:
                self.on_status_change("listening", COMPANION_PORT)

            while self.running:
                try:
                    conn, addr = self.sock.accept()
                    threading.Thread(target=self._handle_connection, args=(conn, addr), daemon=True).start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        log.error(f"Companion accept error: {e}")
        except Exception as e:
            log.error(f"Companion server error: {e}")
        finally:
            if self.sock:
                self.sock.close()

    def stop(self):
        self.running = False
        with self._conns_lock:
            if self._plugin_conn:
                try:
                    self._plugin_conn.close()
                except:
                    pass
            for conn in self._device_conns.values():
                try:
                    conn.close()
                except:
                    pass

    def _handle_connection(self, conn, addr):
        """Handle incoming connection (either plugin or device)."""
        log.info(f"Companion connection from {addr}")
        client_type = None
        device_id = None

        try:
            conn.settimeout(None)  # No timeout for persistent connections

            while self.running:
                # Read length-prefixed message
                length_bytes = conn.recv(4)
                if not length_bytes or len(length_bytes) < 4:
                    break

                length = int.from_bytes(length_bytes, 'big')
                if length <= 0 or length > 1_000_000:
                    log.warning(f"Invalid message length from {addr}: {length}")
                    break

                data = b''
                while len(data) < length:
                    chunk = conn.recv(min(length - len(data), 8192))
                    if not chunk:
                        break
                    data += chunk

                if len(data) < length:
                    break

                try:
                    message = json.loads(data.decode('utf-8'))
                    msg_type = message.get('type', '')

                    # Handle handshake to identify client type
                    if msg_type == 'handshake':
                        device_id = message.get('deviceId')
                        if device_id:
                            # This is an Android device
                            client_type = 'device'
                            with self._conns_lock:
                                self._device_conns[device_id] = conn
                            log.info(f"Device connected: {device_id}")

                            # Send any pending messages
                            self._send_pending_messages(device_id, conn)

                            # Send handshake ack
                            self._send_to_conn(conn, {'type': 'handshake_ack', 'success': True})
                        else:
                            # This is the plugin
                            client_type = 'plugin'
                            with self._conns_lock:
                                self._plugin_conn = conn
                            log.info("Plugin connected")
                            self._send_to_conn(conn, {'type': 'handshake_ack', 'success': True})

                    # Handle approval response from device
                    elif msg_type == 'approval_response':
                        self._relay_to_plugin(message)

                    # Handle user input/reply from device
                    elif msg_type == 'user_input':
                        payload = message.get('payload', {})
                        text = payload.get('text', '')
                        action = payload.get('action', '')

                        if action == 'clipboard_sync' and text:
                            # Copy to Windows clipboard
                            self._copy_to_clipboard(text)
                            log.info(f"Reply copied to clipboard: {text[:50]}...")

                        # Also relay to plugin
                        self._relay_to_plugin(message)

                    # Handle messages from plugin to device
                    elif msg_type in ['approval_request', 'session_start', 'session_complete',
                                     'session_end', 'notification']:
                        target_device = message.get('deviceId')
                        self._relay_to_device(target_device, message)

                    else:
                        log.debug(f"Unknown message type: {msg_type}")

                except json.JSONDecodeError:
                    log.warning(f"Invalid JSON from {addr}")
                except Exception as e:
                    log.error(f"Error processing message: {e}")

        except Exception as e:
            log.debug(f"Companion connection closed: {e}")
        finally:
            # Clean up connection
            with self._conns_lock:
                if client_type == 'plugin' and self._plugin_conn == conn:
                    self._plugin_conn = None
                    log.info("Plugin disconnected")
                elif client_type == 'device' and device_id:
                    self._device_conns.pop(device_id, None)
                    log.info(f"Device disconnected: {device_id}")
            try:
                conn.close()
            except:
                pass

    def _relay_to_plugin(self, message):
        """Relay message to the connected plugin."""
        with self._conns_lock:
            if self._plugin_conn:
                try:
                    self._send_to_conn(self._plugin_conn, message)
                    return True
                except Exception as e:
                    log.error(f"Failed to relay to plugin: {e}")
                    self._plugin_conn = None
        return False

    def _relay_to_device(self, device_id, message):
        """Relay message to a connected device, or queue if offline."""
        with self._conns_lock:
            # Try to send to any connected device if no specific target
            if not device_id:
                for did, conn in self._device_conns.items():
                    try:
                        self._send_to_conn(conn, message)
                        log.info(f"Relayed to device: {did}")
                        return True
                    except:
                        pass

            # Send to specific device
            conn = self._device_conns.get(device_id)
            if conn:
                try:
                    self._send_to_conn(conn, message)
                    return True
                except Exception as e:
                    log.error(f"Failed to relay to device: {e}")
                    self._device_conns.pop(device_id, None)

        # Queue message for when device reconnects
        with self._pending_lock:
            if device_id not in self._pending_messages:
                self._pending_messages[device_id] = []
            self._pending_messages[device_id].append(message)
            # Keep only last 50 messages per device
            if len(self._pending_messages[device_id]) > 50:
                self._pending_messages[device_id] = self._pending_messages[device_id][-50:]

        log.info(f"Message queued for offline device: {device_id or 'any'}")
        return False

    def _send_pending_messages(self, device_id, conn):
        """Send any pending messages to a newly connected device."""
        with self._pending_lock:
            messages = self._pending_messages.pop(device_id, [])
        for msg in messages:
            try:
                self._send_to_conn(conn, msg)
            except:
                break

    def _send_to_conn(self, conn, message):
        """Send length-prefixed JSON message to connection."""
        data = json.dumps(message).encode('utf-8')
        conn.send(len(data).to_bytes(4, 'big'))
        conn.send(data)

    def _copy_to_clipboard(self, text):
        """Copy text to Windows clipboard."""
        if IS_WINDOWS:
            try:
                import ctypes
                from ctypes import wintypes

                # Open clipboard
                ctypes.windll.user32.OpenClipboard(0)
                ctypes.windll.user32.EmptyClipboard()

                # Allocate global memory
                text_bytes = text.encode('utf-16le') + b'\x00\x00'
                hGlobal = ctypes.windll.kernel32.GlobalAlloc(0x0042, len(text_bytes))
                ptr = ctypes.windll.kernel32.GlobalLock(hGlobal)
                ctypes.memmove(ptr, text_bytes, len(text_bytes))
                ctypes.windll.kernel32.GlobalUnlock(hGlobal)

                # Set clipboard data (CF_UNICODETEXT = 13)
                ctypes.windll.user32.SetClipboardData(13, hGlobal)
                ctypes.windll.user32.CloseClipboard()

                log.info("Text copied to clipboard")
            except Exception as e:
                log.error(f"Failed to copy to clipboard: {e}")
        else:
            # On non-Windows, try pyperclip or xclip
            try:
                import subprocess
                if platform.system() == 'Darwin':
                    subprocess.run(['pbcopy'], input=text.encode(), check=True)
                else:
                    subprocess.run(['xclip', '-selection', 'clipboard'], input=text.encode(), check=True)
            except Exception as e:
                log.warning(f"Clipboard copy not available: {e}")


class DiscoveryServer(threading.Thread):
    """UDP server that responds to discovery requests."""

    def __init__(self, connection_info):
        super().__init__(daemon=True)
        self.connection_info = connection_info
        self.running = True
        self.sock = None

    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind(('', DISCOVERY_PORT))
            self.sock.settimeout(1.0)
            while self.running:
                try:
                    data, addr = self.sock.recvfrom(1024)
                    if data.startswith(DISCOVERY_MAGIC):
                        response = DISCOVERY_MAGIC + json.dumps(self.connection_info).encode()
                        self.sock.sendto(response, addr)
                except socket.timeout:
                    continue
                except Exception:
                    pass
        finally:
            if self.sock:
                self.sock.close()

    def stop(self):
        self.running = False

    def update_info(self, info):
        self.connection_info = info


class ShadowBridgeApp:
    """Main application - compact popup style."""

    def __init__(self):
        set_app_user_model_id("ShadowBridge")
        self.root = tk.Tk()
        self.root.title(APP_NAME)
        self.root.configure(bg=COLORS['bg_dark'])
        self.root.overrideredirect(False)

        # Apply modern Windows theme if available
        if HAS_SV_TTK:
            sv_ttk.set_theme("dark")

        # Set window icon from logo.png
        icon_path = get_app_icon_path()
        if icon_path and HAS_PIL:
            try:
                icon_img = Image.open(icon_path)
                icon_img = icon_img.resize((64, 64), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
                self.app_icon = ImageTk.PhotoImage(icon_img)
                self.root.iconphoto(True, self.app_icon)
            except:
                pass

        # Window sizing - fixed compact size, bottom-right position
        self.root.update_idletasks()
        try:
            dpi = self.root.winfo_fpixels('1i')
            scale = dpi / 96.0  # 96 is standard DPI
        except:
            scale = 1.0

        # Fixed compact size for quick connect utility
        self.window_width = int(375 * scale)
        self.window_height = int(780 * scale)

        # Get screen size and taskbar offset
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        # Position at bottom-right, above taskbar
        x = screen_w - self.window_width - 20
        y = screen_h - self.window_height - 100  # Above taskbar

        # Set geometry - fixed size, bottom-right
        self.root.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")
        self.root.resizable(False, False)  # No resizing

        # State - auto-detect SSH port
        detected_port = find_ssh_port()
        self.ssh_port = detected_port if detected_port else 22
        self.is_broadcasting = False
        self.discovery_server = None
        self.data_receiver = None
        self.companion_relay = None
        self.web_process = None
        self.tray_icon = None
        self.selected_device_id = '__ALL__'  # '__ALL__' or a device_id
        self.devices = load_projects_state().get('devices', {})  # device_id -> {name, ip, last_seen, projects}
        self.notes_devices = load_notes_state().get('devices', {})  # device_id -> {name, ip, last_seen, notes}
        self.selected_notes_device_id = '__ALL__'
        self._device_menu_updating = False
        self._tool_poll_job = None

        # Setup modern styles
        self.setup_styles()

        # Build UI
        self.create_widgets()

        # Force geometry AFTER widgets are created (no saved state - always bottom-right)
        self.root.update_idletasks()
        self.root.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")
        self.root.after(100, lambda: apply_windows_11_theme(self.root))

        # Start data receiver for project sync
        self.start_data_receiver()

        # Handle close and minimize
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Unmap>", self.on_minimize)

        # Initial updates
        self.root.after(100, self.update_qr_code)
        self.root.after(200, self.update_status)
        self.root.after(500, self.auto_start_broadcast)
        self.root.after(1000, self.auto_start_web_dashboard)

    def setup_styles(self):
        """Configure ttk styles for modern M3-inspired look."""
        style = ttk.Style()
        style.theme_use('clam')

        # Scrollbar styling
        style.configure(
            "Custom.Vertical.TScrollbar",
            background=COLORS['bg_elevated'],
            troughcolor=COLORS['bg_surface'],
            bordercolor=COLORS['bg_surface'],
            lightcolor=COLORS['bg_elevated'],
            darkcolor=COLORS['bg_elevated'],
            arrowcolor=COLORS['bg_surface'],
            relief='flat',
            borderwidth=0,
            width=10
        )
        style.map("Custom.Vertical.TScrollbar",
            background=[('active', COLORS['accent']), ('!active', COLORS['bg_elevated'])],
            troughcolor=[('active', COLORS['bg_surface']), ('!active', COLORS['bg_surface'])]
        )
        try:
            style.layout(
                "Custom.Vertical.TScrollbar",
                [
                    ("Vertical.Scrollbar.trough", {
                        "children": [
                            ("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})
                        ],
                        "sticky": "nswe"
                    })
                ]
            )
        except Exception:
            pass

        # Entry styling
        style.configure(
            "Custom.TEntry",
            fieldbackground=COLORS['bg_input'],
            foreground=COLORS['text'],
            bordercolor=COLORS['border'],
            insertcolor=COLORS['text']
        )

        # Combobox styling
        style.configure(
            "Custom.TCombobox",
            fieldbackground=COLORS['bg_input'],
            background=COLORS['bg_elevated'],
            foreground=COLORS['text'],
            arrowcolor=COLORS['text_dim'],
            bordercolor=COLORS['border']
        )
        style.map("Custom.TCombobox",
            fieldbackground=[('readonly', COLORS['bg_input'])],
            foreground=[('readonly', COLORS['text'])]
        )

    def create_widgets(self):
        """Create compact single-column layout for Quick Connect and CLI tools."""
        # Simple non-scrollable content frame
        left_inner = tk.Frame(self.root, bg=COLORS['bg_surface'], padx=20, pady=16)
        left_inner.pack(fill=tk.BOTH, expand=True)

        # --- LEFT PANE CONTENT ---
        def add_divider(container, top=6, bottom=6):
            divider = tk.Frame(container, bg=COLORS['divider'], height=1)
            divider.pack(fill=tk.X, pady=(top, bottom))
            return divider

        # Header with title
        header = tk.Frame(left_inner, bg=COLORS['bg_surface'])
        header.pack(fill=tk.X, pady=(0, 12))

        tk.Label(
            header, text="ShadowBridge", bg=COLORS['bg_surface'],
            fg=COLORS['text'], font=('Segoe UI', 18, 'bold')
        ).pack(side=tk.LEFT)

        # Version badge
        version_label = tk.Label(
            header, text=f"v{APP_VERSION}", bg=COLORS['accent_container'],
            fg=COLORS['accent_light'], font=('Segoe UI', 8), padx=6, pady=2
        )
        version_label.pack(side=tk.LEFT, padx=(8, 0))

        # Host info row with broadcast status
        device_row = tk.Frame(left_inner, bg=COLORS['bg_surface'])
        device_row.pack(fill=tk.X, pady=(0, 12))

        # Host stack on left (HOST label above hostname)
        host_stack = tk.Frame(device_row, bg=COLORS['bg_surface'])
        host_stack.pack(side=tk.LEFT)
        tk.Label(
            host_stack, text="HOST", bg=COLORS['bg_surface'],
            fg=COLORS['text_muted'], font=('Segoe UI', 8), anchor='w'
        ).pack(anchor='w')
        tk.Label(
            host_stack, text=f"{socket.gethostname()}", bg=COLORS['bg_surface'],
            fg=COLORS['text_secondary'], font=('Segoe UI', 9), anchor='w'
        ).pack(anchor='w')

        # Status stack on right (device name above broadcast status)
        status_stack = tk.Frame(device_row, bg=COLORS['bg_surface'])
        status_stack.pack(side=tk.RIGHT)

        # Connected device name (above broadcast)
        self.connected_device_label = tk.Label(
            status_stack, text="No device", bg=COLORS['bg_surface'],
            fg=COLORS['text_dim'], font=('Segoe UI', 8), anchor='e'
        )
        self.connected_device_label.pack(anchor='e')

        # Broadcast status below device name
        self.broadcast_status_label = tk.Label(
            status_stack, text="● Broadcasting", bg=COLORS['bg_surface'],
            fg=COLORS['success'], font=('Segoe UI', 8), anchor='e'
        )
        self.broadcast_status_label.pack(anchor='e')

        add_divider(left_inner, 6, 12)

        # QR Code card - elevated with subtle border
        qr_card = tk.Frame(left_inner, bg=COLORS['bg_card'], padx=16, pady=12)
        qr_card.pack(fill=tk.X, pady=(0, 10))
        # Add top highlight line for depth effect
        qr_highlight = tk.Frame(qr_card, bg=COLORS['border_light'], height=1)
        qr_highlight.pack(fill=tk.X, side=tk.TOP)

        # QR Label title
        qr_title = tk.Label(
            qr_card, text="QUICK CONNECT", bg=COLORS['bg_card'],
            fg=COLORS['text_muted'], font=('Segoe UI', 8, 'bold')
        )
        qr_title.pack(anchor='w', pady=(0, 8))

        self.qr_label = tk.Label(qr_card, bg=COLORS['bg_card'], text="Loading...")
        self.qr_label.pack(pady=(0, 12))

        # Connection info - styled row
        info_row = tk.Frame(qr_card, bg=COLORS['bg_elevated'], padx=12, pady=8)
        info_row.pack(fill=tk.X)

        self.ip_label = tk.Label(
            info_row, text=f"{get_local_ip()}", bg=COLORS['bg_elevated'],
            fg=COLORS['accent_light'], font=('Consolas', 11, 'bold')
        )
        self.ip_label.pack(side=tk.LEFT)

        self.ssh_label = tk.Label(
            info_row, text="SSH: ...", bg=COLORS['bg_elevated'],
            fg=COLORS['text_dim'], font=('Consolas', 9)
        )
        self.ssh_label.pack(side=tk.RIGHT)

        add_divider(left_inner, 8, 12)

        # Tools section - elevated card
        tools_card = tk.Frame(left_inner, bg=COLORS['bg_card'], padx=16, pady=10)
        tools_card.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            tools_card, text="CLI TOOLS", bg=COLORS['bg_card'],
            fg=COLORS['text_muted'], font=('Segoe UI', 8, 'bold')
        ).pack(anchor='w', pady=(0, 10))

        # Tool buttons grid
        self.tool_buttons = {}
        self.tool_status = {}
        self._tool_specs = {}
        self._tool_install_procs = {}  # tool_id -> subprocess.Popen
        self._tool_install_lock = threading.Lock()

        tools = [
            ("Claude Code", "claude", {
                "type": "npm",
                "commands": ["npm install -g @anthropic-ai/claude-code"],
                "uninstall_commands": ["npm uninstall -g @anthropic-ai/claude-code"],
            }),
            ("Codex", "codex", {
                "type": "npm",
                "commands": ["npm install -g @openai/codex"],
                "uninstall_commands": ["npm uninstall -g @openai/codex"],
            }),
            ("Gemini", "gemini", {
                "type": "npm",
                "commands": ["npm install -g @google/gemini-cli"],
                "uninstall_commands": ["npm uninstall -g @google/gemini-cli"],
            }),
            ("Aider", "aider", {
                "type": "pip",
                "commands": ["py -m pip install -U aider-chat", "python -m pip install -U aider-chat"],
                "uninstall_commands": ["py -m pip uninstall -y aider-chat", "python -m pip uninstall -y aider-chat"],
                "fallback_url": "https://aider.chat/"
            }),
            ("Ollama", "ollama", {
                "type": "winget",
                "commands": ["winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements"],
                "uninstall_commands": ["winget uninstall -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements"],
                "fallback_url": "https://ollama.com/download/windows"
            }),
        ]

        for name, tool_id, spec in tools:
            self._tool_specs[tool_id] = spec
            row = tk.Frame(tools_card, bg=COLORS['bg_card'])
            row.pack(fill=tk.X, pady=3)

            # Status indicator with glow effect container
            status_frame = tk.Frame(row, bg=COLORS['bg_card'])
            status_frame.pack(side=tk.LEFT, padx=(0, 10))

            status_canvas = tk.Canvas(status_frame, width=10, height=10, bg=COLORS['bg_card'], highlightthickness=0)
            status_canvas.pack(pady=4)
            status_canvas.create_oval(1, 1, 9, 9, fill=COLORS['text_muted'], outline='')
            self.tool_status[tool_id] = status_canvas

            # Tool name with secondary text color
            tk.Label(
                row, text=name, bg=COLORS['bg_card'], fg=COLORS['text_secondary'],
                font=('Segoe UI', 9), width=12, anchor='w'
            ).pack(side=tk.LEFT)

            # Modern button styling
            btn = tk.Button(
                row, text="Install", bg=COLORS['bg_elevated'], fg=COLORS['text'],
                font=('Segoe UI', 8), relief='flat', cursor='hand2',
                width=14, pady=3, activebackground=COLORS['accent'],
                activeforeground='white', bd=0, highlightthickness=0,
                command=lambda s=spec, n=name, t=tool_id: self.install_tool(t, n, s)
            )
            btn.pack(side=tk.RIGHT)
            # Hover effects - handle both Install and Uninstall states
            def on_enter(e, b=btn):
                if b.cget('text') == 'Install':
                    b.configure(bg=COLORS['accent_hover'], fg='white')
                elif b.cget('text') == 'Uninstall':
                    b.configure(bg='#ff4444', fg='white')
            def on_leave(e, b=btn):
                if b.cget('text') == 'Install':
                    b.configure(bg=COLORS['bg_elevated'], fg=COLORS['text'])
                elif b.cget('text') == 'Uninstall':
                    b.configure(bg=COLORS['error'], fg='white')
            btn.bind('<Enter>', on_enter)
            btn.bind('<Leave>', on_leave)
            self.tool_buttons[tool_id] = btn

        # Divider
        tk.Frame(tools_card, bg=COLORS['border'], height=1).pack(fill=tk.X, pady=(10, 8))

        # Tailscale row
        ts_row = tk.Frame(tools_card, bg=COLORS['bg_card'])
        ts_row.pack(fill=tk.X, pady=3)

        ts_status_frame = tk.Frame(ts_row, bg=COLORS['bg_card'])
        ts_status_frame.pack(side=tk.LEFT, padx=(0, 10))

        self.tailscale_status = tk.Canvas(ts_status_frame, width=10, height=10, bg=COLORS['bg_card'], highlightthickness=0)
        self.tailscale_status.pack(pady=4)
        self.tailscale_status.create_oval(1, 1, 9, 9, fill=COLORS['text_muted'], outline='')

        tk.Label(
            ts_row, text="Tailscale", bg=COLORS['bg_card'], fg=COLORS['text_secondary'],
            font=('Segoe UI', 9), width=12, anchor='w'
        ).pack(side=tk.LEFT)

        self.tailscale_btn = tk.Button(
            ts_row, text="Install", bg=COLORS['bg_elevated'], fg=COLORS['text'],
            font=('Segoe UI', 8), relief='flat', cursor='hand2',
            width=14, pady=3, activebackground=COLORS['accent'],
            activeforeground='white', bd=0, highlightthickness=0,
            command=self.install_tailscale
        )
        self.tailscale_btn.pack(side=tk.RIGHT)
        self.tailscale_btn.bind('<Enter>', lambda e: self.tailscale_btn.configure(bg=COLORS['accent'], fg='white'))
        self.tailscale_btn.bind('<Leave>', lambda e: self.tailscale_btn.configure(bg=COLORS['bg_elevated'], fg=COLORS['text']) if self.tailscale_btn.cget('text') == 'Install' else None)

        # Check tool status
        self.root.after(300, self.check_tools)

        # Bottom section (Broadcast & Settings) - In scrollable content
        bottom = tk.Frame(left_inner, bg=COLORS['bg_surface'])
        bottom.pack(fill=tk.X, pady=(16, 0))

        # Web Dashboard button (primary action)
        self.web_dashboard_btn = tk.Button(
            bottom, text="Open Web Dashboard", bg=COLORS['accent'], fg='#ffffff',
            font=('Segoe UI', 10, 'bold'), relief='flat', cursor='hand2',
            padx=20, pady=8, bd=0, highlightthickness=0,
            activebackground=COLORS['accent_hover'], activeforeground='white',
            command=self.launch_web_dashboard
        )
        self.web_dashboard_btn.pack(fill=tk.X)
        self.web_dashboard_btn.bind('<Enter>', lambda e: self.web_dashboard_btn.configure(bg=COLORS['accent_hover']))
        self.web_dashboard_btn.bind('<Leave>', lambda e: self.web_dashboard_btn.configure(bg=COLORS['accent']))

        # Exit button
        exit_btn = tk.Button(
            bottom, text="Exit", bg=COLORS['bg_elevated'], fg=COLORS['text'],
            font=('Segoe UI', 10), relief='flat', cursor='hand2',
            padx=20, pady=8, bd=0, highlightthickness=0,
            activebackground=COLORS['error'], activeforeground='white',
            command=self.force_exit
        )
        exit_btn.pack(fill=tk.X, pady=(6, 8))
        exit_btn.bind('<Enter>', lambda e: exit_btn.configure(bg=COLORS['error'], fg='white'))
        exit_btn.bind('<Leave>', lambda e: exit_btn.configure(bg=COLORS['bg_elevated'], fg=COLORS['text']))

        # Options row with better styling
        opts = tk.Frame(bottom, bg=COLORS['bg_surface'])
        opts.pack(fill=tk.X)

        if IS_WINDOWS:
            self.startup_var = tk.BooleanVar(value=is_startup_enabled())
            startup_cb = tk.Checkbutton(
                opts, text="Start with Windows", variable=self.startup_var,
                command=self.toggle_startup, bg=COLORS['bg_surface'], fg=COLORS['text_secondary'],
                selectcolor=COLORS['bg_elevated'], activebackground=COLORS['bg_surface'],
                activeforeground=COLORS['text'], font=('Segoe UI', 9), cursor='hand2',
                highlightthickness=0, bd=0
            )
            startup_cb.pack(side=tk.LEFT)

        # Help link with hover effect
        help_link = tk.Label(
            opts, text="SSH Help", bg=COLORS['bg_surface'],
            fg=COLORS['accent_light'], font=('Segoe UI', 9), cursor='hand2'
        )
        help_link.pack(side=tk.RIGHT)
        help_link.bind('<Button-1>', lambda e: self.show_ssh_help())
        help_link.bind('<Enter>', lambda e: help_link.configure(fg=COLORS['accent_hover']))
        help_link.bind('<Leave>', lambda e: help_link.configure(fg=COLORS['accent_light']))

        # Start periodic web server status check
        self.root.after(1000, self.check_web_server_status)

    def _create_tooltip(self, widget, text):
        """Create a simple tooltip that appears on hover."""
        tooltip = None

        def show_tooltip(event):
            nonlocal tooltip
            if tooltip:
                return
            x = widget.winfo_rootx() + 20
            y = widget.winfo_rooty() + widget.winfo_height() + 5
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{x}+{y}")
            label = tk.Label(
                tooltip, text=text, bg=COLORS['bg_elevated'], fg=COLORS['text_secondary'],
                font=('Segoe UI', 8), padx=6, pady=3, relief='solid', borderwidth=1
            )
            label.pack()

        def hide_tooltip(event):
            nonlocal tooltip
            if tooltip:
                tooltip.destroy()
                tooltip = None

        widget.bind('<Enter>', lambda e: widget.after(500, lambda: show_tooltip(e) if widget.winfo_containing(widget.winfo_pointerx(), widget.winfo_pointery()) == widget else None), add='+')
        widget.bind('<Leave>', hide_tooltip, add='+')

    def update_status(self):
        """Update status indicators."""
        ssh_ok = check_ssh_running(self.ssh_port)

        # Update SSH label with port info
        if ssh_ok:
            self.ssh_label.configure(text=f"SSH :{self.ssh_port} ✓", fg=COLORS['success'])
        else:
            self.ssh_label.configure(text=f"SSH :{self.ssh_port} ✗", fg=COLORS['error'])

        now = time.time()
        active_devices = [
            d for d in (self.devices or {}).values()
            if now - float(d.get('last_seen', 0) or 0) < 300
        ]

        # Update broadcast status label based on connection state
        if hasattr(self, 'broadcast_status_label'):
            if active_devices:
                self.broadcast_status_label.configure(text=f"● {len(active_devices)} Connected", fg=COLORS['success'])
            elif self.is_broadcasting and ssh_ok:
                self.broadcast_status_label.configure(text="● Broadcasting", fg=COLORS['success'])
            elif ssh_ok:
                self.broadcast_status_label.configure(text="● Ready", fg=COLORS['warning'])
            else:
                self.broadcast_status_label.configure(text="● No SSH", fg=COLORS['error'])

        # Update connected devices display
        self.refresh_connected_devices_ui()

        self.root.after(3000, self.update_status)

    def start_data_receiver(self):
        """Start TCP server to receive project and notes data from Android app."""
        try:
            self.data_receiver = DataReceiver(
                on_data_received=self.on_projects_received,
                on_device_connected=self.on_device_connected,
                on_notes_received=self.on_notes_received,
                on_key_approval_needed=self.on_key_approval_needed
            )
            self.data_receiver.start()
            log.info(f"Data receiver started on port {DATA_PORT}")
        except Exception as e:
            log.error(f"Failed to start data receiver: {e}")

        # Also start companion relay for Claude Code plugin
        self.start_companion_relay()

    def start_companion_relay(self):
        """Start TCP relay server for Claude Code Companion feature."""
        try:
            self.companion_relay = CompanionRelayServer(
                on_status_change=self.on_companion_status_change
            )
            self.companion_relay.start()
            log.info(f"Companion relay started on port {COMPANION_PORT}")
        except Exception as e:
            log.error(f"Failed to start companion relay: {e}")

    def on_companion_status_change(self, status, port):
        """Called when companion relay status changes."""
        log.info(f"Companion relay: {status} on port {port}")

    def on_device_connected(self, device_id, device_name, ip):
        """Called when a device connects and sends data."""
        self.connected_device = {
            'id': device_id,
            'name': device_name,
            'ip': ip,
            'connected_at': time.time()
        }
        # Schedule UI update on main thread
        self.root.after(0, self.update_status)

    def on_projects_received(self, projects):
        """Called when projects data is received from Android app."""
        self.projects = projects
        # Schedule UI update on main thread
        self.root.after(0, self.refresh_projects_ui)

    def refresh_projects_ui(self):
        """Refresh the projects list UI."""
        # Clear existing project widgets
        for widget in self.projects_container.winfo_children():
            widget.destroy()

        # Update count
        self.projects_count_label.configure(text=f"{len(self.projects)}")

        if not self.projects:
            tk.Label(
                self.projects_container, text="No projects synced yet.\nProjects will appear here when\nyour Android app syncs.",
                bg=COLORS['bg_card'], fg=COLORS['text_dim'],
                font=('Segoe UI', 9), justify='center'
            ).pack(pady=20)
            return

        # Add project rows
        for project in self.projects:
            self._add_project_row(project)

    def _add_project_row_legacy(self, project):
        """Legacy: Add a single expandable project card to the UI."""
        name = project.get('name', 'Unnamed')
        path = project.get('path', '')

        # Main card container
        card = tk.Frame(self.projects_container, bg=COLORS['bg_input'], padx=8, pady=6)
        card.pack(fill=tk.X, pady=2)
        card._is_expanded = False

        # Header row (clickable)
        header = tk.Frame(card, bg=COLORS['bg_input'], cursor='hand2')
        header.pack(fill=tk.X)

        # Expand indicator
        expand_label = tk.Label(
            header, text="▶", bg=COLORS['bg_input'], fg=COLORS['text_dim'],
            font=('Segoe UI', 8), width=2
        )
        expand_label.pack(side=tk.LEFT)

        info_frame = tk.Frame(header, bg=COLORS['bg_input'])
        info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        title_label = tk.Label(
            info_frame, text=name, bg=COLORS['bg_input'], fg=COLORS['text'],
            font=('Segoe UI', 9, 'bold'), anchor='w', cursor='hand2'
        )
        title_label.pack(anchor='w')

        # Content frame (initially hidden)
        content_frame = tk.Frame(card, bg=COLORS['bg_card'], padx=8, pady=8)

        # Determine path display and whether it's accessible
        is_pc = is_pc_path(path) and path
        exists = is_pc and (os.path.isdir(path) or os.path.exists(path))

        if is_pc:
            path_display = path
            path_color = COLORS['text'] if exists else COLORS['text_dim']
        else:
            path_display = f"Phone: {path}" if path else "Phone only"
            path_color = COLORS['warning']

        # Path label in content
        tk.Label(
            content_frame, text="Path:", bg=COLORS['bg_card'], fg=COLORS['text_dim'],
            font=('Segoe UI', 8), anchor='w'
        ).pack(anchor='w')

        tk.Label(
            content_frame, text=path_display, bg=COLORS['bg_card'], fg=path_color,
            font=('Consolas', 9), anchor='w', wraplength=350
        ).pack(anchor='w', pady=(0, 8))

        # Open button inside expanded content (only for PC paths that exist)
        if is_pc and exists:
            open_btn = tk.Button(
                content_frame, text="Open in Explorer", bg=COLORS['accent'], fg='#ffffff',
                font=('Segoe UI', 9), relief='flat', cursor='hand2',
                padx=12, pady=4, bd=0, highlightthickness=0,
                activebackground=COLORS['accent_hover'], activeforeground='white',
                command=lambda p=path: open_folder(p)
            )
            open_btn.pack(anchor='w')
            open_btn.bind('<Enter>', lambda e, b=open_btn: b.configure(bg=COLORS['accent_hover']))
            open_btn.bind('<Leave>', lambda e, b=open_btn: b.configure(bg=COLORS['accent']))
        elif is_pc and not exists:
            tk.Label(
                content_frame, text="(Path not found on this PC)", bg=COLORS['bg_card'],
                fg=COLORS['text_muted'], font=('Segoe UI', 8, 'italic'), anchor='w'
            ).pack(anchor='w')

        def toggle_expand(event=None):
            if card._is_expanded:
                content_frame.pack_forget()
                expand_label.configure(text="▶")
                card._is_expanded = False
            else:
                content_frame.pack(fill=tk.X, pady=(6, 0))
                expand_label.configure(text="▼")
                card._is_expanded = True

        # Bind click events to header elements
        for widget in [header, expand_label, title_label, info_frame]:
            widget.bind('<Button-1>', toggle_expand)

        # Hover effect on header
        def on_enter(e):
            header.configure(bg=COLORS['bg_elevated'])
            expand_label.configure(bg=COLORS['bg_elevated'])
            info_frame.configure(bg=COLORS['bg_elevated'])
            title_label.configure(bg=COLORS['bg_elevated'])
            for child in info_frame.winfo_children():
                child.configure(bg=COLORS['bg_elevated'])

        def on_leave(e):
            header.configure(bg=COLORS['bg_input'])
            expand_label.configure(bg=COLORS['bg_input'])
            info_frame.configure(bg=COLORS['bg_input'])
            title_label.configure(bg=COLORS['bg_input'])
            for child in info_frame.winfo_children():
                child.configure(bg=COLORS['bg_input'])

        header.bind('<Enter>', on_enter)
        header.bind('<Leave>', on_leave)

    def check_tools(self):
        """Check which tools are installed."""
        tool_names = {
            "claude": "Claude Code",
            "codex": "Codex",
            "gemini": "Gemini",
            "aider": "Aider",
            "ollama": "Ollama",
        }

        def check():
            # Check CLI tools
            for tool_id, canvas in self.tool_status.items():
                with self._tool_install_lock:
                    # Don't stomp the UI if an install/uninstall is currently in progress.
                    if tool_id in self._tool_install_procs:
                        continue
                installed = check_tool_installed(tool_id)
                color = COLORS['success'] if installed else COLORS['text_dim']
                name = tool_names.get(tool_id, tool_id)

                def update_ui(c=canvas, col=color, t=tool_id, n=name, inst=installed):
                    self._update_tool_dot(c, col)
                    # Button state: Install vs Uninstall
                    if inst:
                        self.tool_buttons[t].configure(
                            text="Uninstall",
                            state='normal',
                            bg=COLORS['error'],
                            fg='#ffffff',
                            activebackground=COLORS['error']
                        )
                        self.tool_buttons[t].configure(
                            command=lambda tid=t, nm=n: self.uninstall_tool(tid, nm)
                        )
                    else:
                        self.tool_buttons[t].configure(
                            text="Install",
                            state='normal',
                            bg=COLORS['accent'],
                            fg='#ffffff',
                            activebackground=COLORS['accent_hover']
                        )
                        self.tool_buttons[t].configure(
                            command=lambda tid=t, nm=n: self.install_tool(tid, nm, self._tool_specs.get(tid, {}))
                        )

                self.root.after(0, update_ui)

            # Check Tailscale
            ts_ip = get_tailscale_ip()
            def update_tailscale():
                if ts_ip:
                    self.tailscale_btn.configure(
                        text=f"{ts_ip}",
                        state='disabled',
                        bg=COLORS['bg_elevated'],
                        fg='#ffffff',
                        disabledforeground='#ffffff'
                    )
                    self.tailscale_status.delete('all')
                    self.tailscale_status.create_oval(0, 0, 8, 8, fill=COLORS['success'], outline='')
            self.root.after(0, update_tailscale)

        threading.Thread(target=check, daemon=True).start()
        if self._tool_poll_job:
            try:
                self.root.after_cancel(self._tool_poll_job)
            except Exception:
                pass
        self._tool_poll_job = self.root.after(30000, self.check_tools)

    def _update_tool_dot(self, canvas, color):
        canvas.delete('all')
        canvas.create_oval(0, 0, 8, 8, fill=color, outline='')

    def get_connection_info(self):
        """Get connection info with ALL fallback options for auto-reconnection."""
        all_ips = get_all_ips()
        tailscale_ip = get_tailscale_ip()
        local_ip = get_local_ip()
        hostname_local = get_hostname_local()

        # Build list of hosts to try (in priority order)
        # Android app will try these in order until one works
        hosts_to_try = []

        # 1. Tailscale (most stable - works across networks)
        if tailscale_ip:
            hosts_to_try.append({"host": tailscale_ip, "type": "tailscale", "stable": True})

        # 2. Hostname.local (stable on local network even if IP changes)
        if hostname_local:
            hosts_to_try.append({"host": hostname_local, "type": "mdns", "stable": True})

        # 3. All local IPs
        for ip in all_ips.get('local', []):
            hosts_to_try.append({"host": ip, "type": "local", "stable": False})

        # 4. VPN IPs
        for ip in all_ips.get('vpn', []):
            hosts_to_try.append({"host": ip, "type": "vpn", "stable": False})

        # 5. Other IPs
        for ip in all_ips.get('other', []):
            hosts_to_try.append({"host": ip, "type": "other", "stable": False})

        # Primary host (best available)
        primary_host = tailscale_ip or local_ip
        mode = "tailscale" if tailscale_ip else "local"

        return {
            "type": "shadowai_connect",
            "version": 3,  # New version with multi-host support
            "mode": mode,
            "host": primary_host,  # Primary for QR display
            "port": self.ssh_port,
            "username": get_username(),
            "hostname": socket.gethostname(),
            "hostname_local": hostname_local,
            "hosts": hosts_to_try,  # All hosts to try in order
            "local_ip": local_ip,
            "tailscale_ip": tailscale_ip,
            "timestamp": int(time.time())
        }

    def update_qr_code(self):
        """Generate and display QR code."""
        if not HAS_QRCODE or not HAS_PIL:
            self.qr_label.configure(text="QR unavailable")
            return

        try:
            info = self.get_connection_info()
            info_json = json.dumps(info)
            encoded = base64.urlsafe_b64encode(info_json.encode()).decode()
            qr_data = f"shadowai://connect?data={encoded}"

            qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=8, border=2)
            qr.add_data(qr_data)
            qr.make(fit=True)

            img = qr.make_image(fill_color=COLORS['text'], back_color=COLORS['bg_card'])
            img = img.resize((270, 270), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

            self.qr_image = ImageTk.PhotoImage(img)
            self.qr_label.configure(image=self.qr_image, text="")
        except Exception as e:
            self.qr_label.configure(text="QR Error")

    def auto_start_broadcast(self):
        """Auto-start broadcasting if SSH is running."""
        if check_ssh_running(self.ssh_port):
            self.start_broadcast()

    def _get_tools_dir(self):
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))

    def _get_web_server_command(self, open_browser: bool):
        if getattr(sys, 'frozen', False):
            cmd = [sys.executable, "--web-server"]
        else:
            cmd = [sys.executable, os.path.abspath(__file__), "--web-server"]
        if not open_browser:
            cmd.append("--no-browser")
        return cmd

    def auto_start_web_dashboard(self):
        """Auto-start web dashboard server (without opening browser)."""
        try:
            import urllib.request
            urllib.request.urlopen("http://127.0.0.1:6767", timeout=1)
            # Already running
            return
        except:
            pass

        # Start the server silently
        try:
            tools_dir = self._get_tools_dir()
            if not getattr(sys, 'frozen', False):
                web_folder = os.path.join(tools_dir, "web")
                if not os.path.exists(web_folder):
                    log.warning("Web dashboard folder not found, skipping auto-start")
                    return

            # Launch as detached process
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0
            # Use DETACHED_PROCESS to fully separate from parent
            creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            cmd = self._get_web_server_command(open_browser=False)
            web_log = open(WEB_LOG_FILE, "a", encoding="utf-8", errors="replace")
            self.web_process = subprocess.Popen(
                cmd,
                cwd=tools_dir,
                startupinfo=startupinfo,
                creationflags=creation_flags,
                stdin=subprocess.DEVNULL,
                stdout=web_log,
                stderr=web_log,
                close_fds=True
            )
            web_log.close()
            log.info(f"Web dashboard auto-started (PID: {self.web_process.pid})")
        except Exception as e:
            log.error(f"Failed to auto-start web dashboard: {e}")

    def toggle_broadcast(self):
        """Toggle broadcasting."""
        if self.is_broadcasting:
            self.stop_broadcast()
        else:
            self.start_broadcast()

    def launch_web_dashboard(self):
        """Launch the web dashboard in browser."""
        web_port = 6767
        web_url = f"http://127.0.0.1:{web_port}"

        # Check if server is already running
        try:
            import urllib.request
            urllib.request.urlopen(web_url, timeout=1)
            # Server is running, just open browser
            webbrowser.open(web_url)
            return
        except:
            pass

        # Start the web server
        try:
            tools_dir = self._get_tools_dir()
            if not getattr(sys, 'frozen', False):
                web_folder = os.path.join(tools_dir, "web")
                if not os.path.exists(web_folder):
                    messagebox.showerror(
                        "Web Dashboard Missing",
                        f"Web dashboard folder not found.\n\n"
                        f"Expected at:\n{web_folder}"
                    )
                    return

            # Launch web server in background
            if IS_WINDOWS:
                # Use CREATE_NO_WINDOW + DETACHED_PROCESS to fully hide console
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0  # SW_HIDE
                creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                cmd = self._get_web_server_command(open_browser=False)
                web_log = open(WEB_LOG_FILE, "a", encoding="utf-8", errors="replace")
                self.web_process = subprocess.Popen(
                    cmd,
                    cwd=tools_dir,
                    startupinfo=startupinfo,
                    creationflags=creation_flags,
                    stdin=subprocess.DEVNULL,
                    stdout=web_log,
                    stderr=web_log,
                    close_fds=True
                )
                web_log.close()
            else:
                cmd = self._get_web_server_command(open_browser=False)
                web_log = open(WEB_LOG_FILE, "a", encoding="utf-8", errors="replace")
                self.web_process = subprocess.Popen(
                    cmd,
                    cwd=tools_dir,
                    stdout=web_log,
                    stderr=web_log
                )
                web_log.close()

            log.info(f"Web dashboard started (PID: {self.web_process.pid})")

            # Monitor and open browser when server is ready
            def monitor_and_open():
                time.sleep(1.5)
                # Check if server is responding (more reliable than process check with DETACHED)
                try:
                    import urllib.request
                    urllib.request.urlopen("http://127.0.0.1:6767", timeout=3)
                    # Server is running, open browser
                    webbrowser.open(web_url)
                except Exception as e:
                    log.warning(f"Web server not responding yet: {e}")
                    # Wait a bit more and try again
                    time.sleep(2)
                    try:
                        urllib.request.urlopen("http://127.0.0.1:6767", timeout=3)
                        webbrowser.open(web_url)
                    except:
                        log.error("Web dashboard failed to start")
                        self.root.after(0, lambda: messagebox.showerror(
                            "Web Dashboard Error",
                            "Web server failed to start. Check if port 6767 is available."
                        ))

            threading.Thread(target=monitor_and_open, daemon=True).start()

        except Exception as e:
            log.error(f"Failed to launch web dashboard: {e}")
            import traceback
            log.error(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to launch web dashboard:\n{e}")

    def check_web_server_status(self):
        """Check if web dashboard server is running and update status indicator."""
        if not hasattr(self, 'web_status_dot') or self.web_status_dot is None:
            # Widget not created, skip status check
            self.root.after(3000, self.check_web_server_status)
            return

        try:
            import urllib.request
            urllib.request.urlopen("http://127.0.0.1:6767", timeout=1)
            # Server is running
            self.web_status_dot.delete('dot')
            self.web_status_dot.create_oval(1, 1, 9, 9, fill=COLORS['success'], outline='', tags='dot')
            self.web_dashboard_btn.configure(text="Open Web Dashboard")
        except Exception:
            # Server not running
            self.web_status_dot.delete('dot')
            self.web_status_dot.create_oval(1, 1, 9, 9, fill=COLORS['text_muted'], outline='', tags='dot')
            self.web_dashboard_btn.configure(text="Launch Web Dashboard")

        # Schedule next check
        self.root.after(3000, self.check_web_server_status)

    def start_broadcast(self):
        """Start discovery server."""
        try:
            # Ensure firewall rule exists for discovery
            setup_firewall_rule()

            self.discovery_server = DiscoveryServer(self.get_connection_info())
            self.discovery_server.start()
            self.is_broadcasting = True
            # Update status label
            if hasattr(self, 'broadcast_status_label'):
                self.broadcast_status_label.configure(text="● Broadcasting", fg=COLORS['success'])
        except Exception:
            pass

    def stop_broadcast(self):
        """Stop discovery server."""
        if self.discovery_server:
            self.discovery_server.stop()
            self.discovery_server = None
        self.is_broadcasting = False
        # Update status label
        if hasattr(self, 'broadcast_status_label'):
            self.broadcast_status_label.configure(text="● Stopped", fg=COLORS['text_muted'])

    def install_tool(self, tool_id, name, spec):
        """Install a tool (npm/pip/winget)."""
        with self._tool_install_lock:
            if tool_id in self._tool_install_procs:
                return
        btn = self.tool_buttons[tool_id]
        original_text = btn.cget('text')
        btn.configure(text="Stop", state='normal', bg=COLORS['warning'])

        def do_install():
            install_type = (spec or {}).get("type", "npm")
            commands = (spec or {}).get("commands", [])
            fallback_url = (spec or {}).get("fallback_url", None)

            if install_type == "npm":
                if not check_npm_installed():
                    self.root.after(0, lambda: self._prompt_nodejs_install(tool_id, name, original_text))
                    return
            elif install_type == "pip":
                if not check_python_pip_installed():
                    self.root.after(0, lambda: self._prompt_python_install(tool_id, name, original_text))
                    return
            elif install_type == "winget":
                if not check_winget_installed():
                    self.root.after(0, lambda: self._prompt_direct_download(tool_id, name, original_text, fallback_url))
                    return

            try:
                flags = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
                timeout_s = 900 if install_type == "winget" else 300

                last_error = "Unknown error"
                for command in commands:
                    proc = subprocess.Popen(
                        command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        creationflags=flags
                    )
                    with self._tool_install_lock:
                        self._tool_install_procs[tool_id] = proc

                    try:
                        stdout, stderr = proc.communicate(timeout=timeout_s)
                        rc = proc.returncode
                    except subprocess.TimeoutExpired:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        stdout, stderr = proc.communicate(timeout=5)
                        rc = proc.returncode
                        raise subprocess.TimeoutExpired(command, timeout_s)
                    finally:
                        with self._tool_install_lock:
                            self._tool_install_procs.pop(tool_id, None)

                    if rc == 0 or check_tool_installed(tool_id):
                        self.root.after(0, lambda: self._mark_installed(tool_id))
                        return

                    stderr = (stderr or "").strip()
                    stdout = (stdout or "").strip()
                    last_error = (stderr or stdout or last_error)[:200]

                self.root.after(0, lambda: self._mark_failed(tool_id, name, last_error))
            except subprocess.TimeoutExpired:
                self.root.after(0, lambda: self._mark_failed(tool_id, name, "Installation timed out"))
            except Exception as e:
                self.root.after(0, lambda: self._mark_failed(tool_id, name, str(e)[:100]))

        threading.Thread(target=do_install, daemon=True).start()
        btn.configure(command=lambda tid=tool_id, nm=name: self.stop_tool_install(tid, nm))

    def stop_tool_install(self, tool_id, name):
        """Stop/cancel an in-progress install."""
        proc = None
        with self._tool_install_lock:
            proc = self._tool_install_procs.get(tool_id)

        if proc is None:
            return

        try:
            if IS_WINDOWS:
                try:
                    subprocess.run(
                        f"taskkill /PID {proc.pid} /T /F",
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=10,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                except Exception:
                    pass

            try:
                proc.terminate()
            except Exception:
                pass

            try:
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        finally:
            with self._tool_install_lock:
                self._tool_install_procs.pop(tool_id, None)
            self._mark_failed(tool_id, name, "Cancelled")

    def _mark_installed(self, tool_id):
        """Mark a tool as installed (refreshes buttons)."""
        self._update_tool_dot(self.tool_status[tool_id], COLORS['success'])
        # NPM global bin dir is frequently missing from PATH on Windows; fix it after npm installs.
        spec = (getattr(self, "_tool_specs", {}) or {}).get(tool_id, {}) or {}
        if IS_WINDOWS and spec.get("type") == "npm":
            npm_dir = os.path.join(os.environ.get("APPDATA", ""), "npm")
            if npm_dir and os.path.isdir(npm_dir):
                ensure_windows_user_path_contains(npm_dir)
        self.check_tools()

    def _mark_failed(self, tool_id, name, error):
        """Mark installation as failed."""
        btn = self.tool_buttons[tool_id]
        btn.configure(text="Install", state='normal', bg=COLORS['accent'])
        btn.configure(command=lambda tid=tool_id, nm=name: self.install_tool(tid, nm, self._tool_specs.get(tid, {})))
        if error and error != "Cancelled":
            messagebox.showerror("Install Failed", f"Failed to install {name}:\n\n{error}")
        self.check_tools()

    def _prompt_nodejs_install(self, tool_id, name, original_text):
        """Prompt user to install Node.js."""
        btn = self.tool_buttons[tool_id]
        btn.configure(text=original_text, state='normal', bg=COLORS['accent'])

        if messagebox.askyesno(
            "Node.js Required",
            f"npm is required to install {name}.\n\n"
            "Would you like to download Node.js now?\n\n"
            "(After installing, restart ShadowBridge and try again)"
        ):
            webbrowser.open("https://nodejs.org/en/download/")

    def uninstall_tool(self, tool_id, name):
        """Uninstall a tool (npm/pip/winget)."""
        with self._tool_install_lock:
            if tool_id in self._tool_install_procs:
                return

        spec = self._tool_specs.get(tool_id, {}) or {}
        install_type = spec.get("type", "npm")
        commands = spec.get("uninstall_commands", [])
        fallback_url = spec.get("fallback_url", None)

        btn = self.tool_buttons.get(tool_id)
        if btn is None:
            return

        btn.configure(text="Uninstalling...", state='disabled', bg=COLORS['warning'])

        def do_uninstall():
            if not commands:
                self.root.after(0, lambda: self._mark_failed(tool_id, name, "No uninstall command configured"))
                return

            if install_type == "npm" and not check_npm_installed():
                self.root.after(0, lambda: self._prompt_nodejs_install(tool_id, name, "Uninstall"))
                return
            if install_type == "pip" and not check_python_pip_installed():
                self.root.after(0, lambda: self._prompt_python_install(tool_id, name, "Uninstall"))
                return
            if install_type == "winget" and not check_winget_installed():
                self.root.after(0, lambda: self._prompt_direct_download(tool_id, name, "Uninstall", fallback_url))
                return

            flags = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
            timeout_s = 900 if install_type == "winget" else 300

            last_error = "Unknown error"
            for command in commands:
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    creationflags=flags
                )
                with self._tool_install_lock:
                    self._tool_install_procs[tool_id] = proc

                try:
                    stdout, stderr = proc.communicate(timeout=timeout_s)
                    rc = proc.returncode
                except subprocess.TimeoutExpired:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    stdout, stderr = proc.communicate(timeout=5)
                    rc = proc.returncode
                    raise
                finally:
                    with self._tool_install_lock:
                        self._tool_install_procs.pop(tool_id, None)

                if rc == 0 and not check_tool_installed(tool_id):
                    self.root.after(0, lambda: self.check_tools())
                    return

                stderr = (stderr or "").strip()
                stdout = (stdout or "").strip()
                last_error = (stderr or stdout or last_error)[:200]

            self.root.after(0, lambda: messagebox.showerror("Uninstall Failed", f"Failed to uninstall {name}:\n\n{last_error}"))
            self.root.after(0, lambda: self.check_tools())

        threading.Thread(target=do_uninstall, daemon=True).start()

    def _prompt_python_install(self, tool_id, name, original_text):
        """Prompt user to install Python (for pip installs)."""
        btn = self.tool_buttons[tool_id]
        btn.configure(text=original_text, state='normal', bg=COLORS['accent'])

        if messagebox.askyesno(
            "Python Required",
            f"Python + pip is required to install {name}.\n\n"
            "Would you like to download Python now?\n\n"
            "(After installing, restart ShadowBridge and try again)"
        ):
            webbrowser.open("https://www.python.org/downloads/")

    def _prompt_direct_download(self, tool_id, name, original_text, url):
        """Prompt user to download an installer directly."""
        btn = self.tool_buttons[tool_id]
        btn.configure(text=original_text, state='normal', bg=COLORS['accent'])

        if not url:
            messagebox.showinfo("Install", f"Please install {name} manually.")
            return

        if messagebox.askyesno(
            "Download Required",
            f"{name} couldn't be installed automatically.\n\nOpen the download page?"
        ):
            webbrowser.open(url)

    def install_tailscale(self):
        """Install Tailscale for stable connections."""
        # Check if already installed
        if get_tailscale_ip():
            self.tailscale_btn.configure(text="✓ Connected", state='disabled', bg=COLORS['success'])
            self.tailscale_status.delete('all')
            self.tailscale_status.create_oval(0, 0, 8, 8, fill=COLORS['success'], outline='')
            messagebox.showinfo("Tailscale", "Tailscale is already installed and connected!")
            return

        self.tailscale_btn.configure(text="Installing...", state='disabled', bg=COLORS['warning'])

        def do_install():
            try:
                import urllib.request
                import tempfile

                # Download Tailscale installer
                url = "https://pkgs.tailscale.com/stable/tailscale-setup-latest.exe"
                temp_dir = tempfile.gettempdir()
                installer_path = os.path.join(temp_dir, "tailscale-setup.exe")

                self.root.after(0, lambda: self.tailscale_btn.configure(text="Downloading..."))

                urllib.request.urlretrieve(url, installer_path)

                self.root.after(0, lambda: self.tailscale_btn.configure(text="Running installer..."))

                # Run installer (will prompt for admin)
                if IS_WINDOWS:
                    os.startfile(installer_path)
                else:
                    subprocess.run(["open", installer_path])

                self.root.after(0, lambda: self._tailscale_install_done())

            except Exception as e:
                self.root.after(0, lambda: self._tailscale_install_failed(str(e)))

        threading.Thread(target=do_install, daemon=True).start()

    def _tailscale_install_done(self):
        """Tailscale installer launched."""
        self.tailscale_btn.configure(text="Restart after setup", state='normal', bg=COLORS['accent'])
        messagebox.showinfo(
            "Tailscale Setup",
            "Tailscale installer launched!\n\n"
            "1. Complete the installation\n"
            "2. Sign in to Tailscale\n"
            "3. Install Tailscale on your phone too\n"
            "4. Restart ShadowBridge\n\n"
            "Your devices will get stable 100.x.x.x IPs that never change!"
        )

    def _tailscale_install_failed(self, error):
        """Tailscale install failed."""
        self.tailscale_btn.configure(text="Tailscale", state='normal', bg=COLORS['success'])
        messagebox.showerror("Install Failed", f"Failed to download Tailscale:\n\n{error}\n\nVisit tailscale.com/download")

    def toggle_startup(self):
        """Toggle Windows startup."""
        enabled = self.startup_var.get()
        if not set_startup_enabled(enabled):
            self.startup_var.set(not enabled)

    def show_ssh_help(self):
        """Show SSH help dialog."""
        help_win = tk.Toplevel(self.root)
        help_win.title("SSH Setup")
        help_win.geometry("400x300")
        help_win.configure(bg=COLORS['bg_dark'])
        help_win.transient(self.root)

        text = """Windows SSH Setup:

1. Open Settings → Apps → Optional Features
2. Click "Add a feature"
3. Find and install "OpenSSH Server"
4. Open Services (Win+R → services.msc)
5. Find "OpenSSH SSH Server"
6. Set to "Automatic" and click "Start"

Or run in PowerShell (Admin):
  Start-Service sshd
  Set-Service -Name sshd -StartupType Automatic"""

        tk.Label(
            help_win, text=text, bg=COLORS['bg_dark'], fg=COLORS['text'],
            font=('Consolas', 9), justify='left', padx=20, pady=20
        ).pack(fill=tk.BOTH, expand=True)

    def _save_window_state(self):
        """Save window position and size to file."""
        try:
            geometry = self.root.geometry()
            # Parse geometry string like "1180x820+100+50"
            import re
            match = re.match(r'(\d+)x(\d+)\+(-?\d+)\+(-?\d+)', geometry)
            if match:
                state = {
                    'width': int(match.group(1)),
                    'height': int(match.group(2)),
                    'x': int(match.group(3)),
                    'y': int(match.group(4))
                }
                with open(WINDOW_STATE_FILE, 'w') as f:
                    json.dump(state, f)
        except Exception as e:
            log.debug(f"Failed to save window state: {e}")

    def _load_window_state(self):
        """Load and apply saved window position and size."""
        try:
            if os.path.exists(WINDOW_STATE_FILE):
                with open(WINDOW_STATE_FILE, 'r') as f:
                    state = json.load(f)
                w = state.get('width', self.window_width)
                h = state.get('height', self.window_height)
                x = state.get('x', 100)
                y = state.get('y', 100)
                # Validate position is on screen
                screen_w = self.root.winfo_screenwidth()
                screen_h = self.root.winfo_screenheight()
                if x < 0 or x > screen_w - 100:
                    x = 100
                if y < 0 or y > screen_h - 100:
                    y = 100
                self.root.geometry(f"{w}x{h}+{x}+{y}")
                return True
        except Exception as e:
            log.debug(f"Failed to load window state: {e}")
        return False

    def minimize_to_tray(self):
        """Minimize to system tray."""
        if not HAS_TRAY or not HAS_PIL:
            return

        self._save_window_state()
        self.root.withdraw()

        icon_image = create_app_icon(64)
        if icon_image and icon_image.mode == 'RGBA':
            background = Image.new('RGB', icon_image.size, COLORS['bg_dark'])
            background.paste(icon_image, mask=icon_image.split()[3])
            icon_image = background

        def on_show(icon, item):
            icon.stop()
            self.root.after(0, self._restore_from_tray)

        def on_exit(icon, item):
            icon.stop()
            self.root.after(0, self.quit_app)

        menu = pystray.Menu(
            pystray.MenuItem("Show", on_show, default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Exit", on_exit)
        )

        self.tray_icon = pystray.Icon(APP_NAME, icon_image, APP_NAME, menu)
        self.tray_thread = threading.Thread(target=self.tray_icon.run)
        self.tray_thread.start()

    def _restore_from_tray(self):
        """Restore from tray."""
        self.tray_icon = None
        self.root.deiconify()
        self.root.lift()

    def quit_app(self):
        """Quit application."""
        self._save_window_state()
        self.stop_broadcast()
        if self.data_receiver:
            try:
                self.data_receiver.stop()
            except:
                pass
        if self.companion_relay:
            try:
                self.companion_relay.stop()
            except:
                pass
        if self.web_process:
            try:
                self.web_process.terminate()
                log.info("Web dashboard process terminated")
            except:
                pass
        if self.tray_icon:
            try:
                self.tray_icon.stop()
            except:
                pass
        self.root.quit()
        self.root.destroy()

    def force_exit(self):
        """Force exit the application immediately."""
        self.stop_broadcast()
        if self.data_receiver:
            try:
                self.data_receiver.stop()
            except:
                pass
        if self.companion_relay:
            try:
                self.companion_relay.stop()
            except:
                pass
        if self.web_process:
            try:
                self.web_process.terminate()
            except:
                pass
        if self.tray_icon:
            try:
                self.tray_icon.stop()
            except:
                pass
        self.root.destroy()
        os._exit(0)  # Force exit

    def on_minimize(self, event=None):
        """Handle minimize - go to system tray instead of taskbar."""
        if event and self.root.state() == 'iconic':
            # Window was minimized, send to tray instead
            self.root.after(10, self._do_minimize_to_tray)

    def _do_minimize_to_tray(self):
        """Actually minimize to tray after event processing."""
        if HAS_TRAY and HAS_PIL and not self.tray_icon:
            self.minimize_to_tray()

    def on_close(self):
        """Handle window close - minimize to tray by default."""
        if HAS_TRAY and HAS_PIL:
            # Minimize to tray by default (no dialog)
            self.minimize_to_tray()
        else:
            # No tray support, just exit
            self.quit_app()

    def run(self):
        """Run the app."""
        self.root.mainloop()

    # =======================
    # Multi-device Projects UI
    # =======================

    def _get_active_devices(self):
        now = time.time()
        devices_sorted = sorted(
            (self.devices or {}).values(),
            key=lambda d: d.get('last_seen', 0),
            reverse=True
        )
        return [d for d in devices_sorted if now - float(d.get('last_seen', 0) or 0) < 300]

    def refresh_connected_devices_ui(self):
        """Refresh the connected device label in the header."""
        if not hasattr(self, "connected_device_label"):
            return

        active = self._get_active_devices()

        if not active:
            self.connected_device_label.configure(text="No device", fg=COLORS['text_dim'])
            return

        # Show first connected device name
        device = active[0]
        name = device.get('name') or device.get('id') or "Unknown"
        if len(active) > 1:
            self.connected_device_label.configure(text=f"● {name} +{len(active)-1}", fg=COLORS['success'])
        else:
            self.connected_device_label.configure(text=f"● {name}", fg=COLORS['success'])

    def refresh_project_device_menu(self):
        """Refresh the Projects device selector menu."""
        if not hasattr(self, "project_device_menu"):
            return

        if self._device_menu_updating:
            return

        self._device_menu_updating = True
        try:
            menu = self.project_device_menu["menu"]
            menu.delete(0, "end")

            self._device_label_to_id = {}
            id_to_label = {}

            def add_option(label, device_id_value):
                self._device_label_to_id[label] = device_id_value
                id_to_label[device_id_value] = label
                menu.add_command(label=label, command=lambda l=label: self.project_device_var.set(l))

            add_option("All devices", "__ALL__")

            devices_sorted = sorted(
                (self.devices or {}).values(),
                key=lambda d: d.get('last_seen', 0),
                reverse=True
            )
            for device in devices_sorted:
                did = device.get('id')
                if not did:
                    continue
                name = device.get('name') or did
                ip = device.get('ip') or ""
                label = f"{name}{f' ({ip})' if ip else ''}"
                add_option(label, did)

            # Keep selection stable by device_id
            desired = self.selected_device_id if getattr(self, "selected_device_id", "__ALL__") else "__ALL__"
            self.project_device_var.set(id_to_label.get(desired, "All devices"))
        finally:
            self._device_menu_updating = False

    def on_project_device_selected(self):
        """Handle selection change in the Projects device selector."""
        if self._device_menu_updating:
            return
        if not hasattr(self, "project_device_var"):
            return

        label = self.project_device_var.get() or "All devices"
        self.selected_device_id = getattr(self, "_device_label_to_id", {}).get(label, "__ALL__")
        self.refresh_projects_ui()

    def on_device_connected(self, device_id, device_name, ip):
        """Called when a device connects and sends data."""
        device = (self.devices or {}).get(device_id, {})
        device.update({
            'id': device_id,
            'name': device_name,
            'ip': ip,
            'last_seen': time.time(),
            'projects': device.get('projects', [])
        })
        self.devices[device_id] = device

        self.refresh_connected_devices_ui()
        self.refresh_project_device_menu()
        self.update_status()

    def on_projects_received(self, device_id, projects):
        """Called when projects data is received from Android app."""
        if not isinstance(projects, list):
            return

        device = (self.devices or {}).get(device_id, {})
        device.update({
            'id': device_id,
            'name': device.get('name', device_id),
            'ip': device.get('ip', None),
            'last_seen': time.time(),
            'projects': projects
        })
        self.devices[device_id] = device

        self.refresh_connected_devices_ui()
        self.refresh_project_device_menu()
        self.refresh_projects_ui()
        self.update_status()

        # Notify web dashboard for real-time sync
        self._notify_web_dashboard_projects_updated(device_id)

    def _notify_web_dashboard_projects_updated(self, device_id):
        """Notify web dashboard that projects have been updated."""
        def do_notify():
            try:
                import urllib.request
                import json as json_module
                data = json_module.dumps({'device_id': device_id}).encode('utf-8')
                req = urllib.request.Request(
                    'http://127.0.0.1:6767/api/projects/sync',
                    data=data,
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                urllib.request.urlopen(req, timeout=2)
                log.debug(f"Notified web dashboard of projects update for {device_id}")
            except Exception as e:
                log.debug(f"Could not notify web dashboard: {e}")

        threading.Thread(target=do_notify, daemon=True).start()

    def on_key_approval_needed(self, device_id, device_name, ip):
        """Called when a new device requests SSH key installation - requires user approval."""
        log.info(f"Key approval needed for: {device_name} ({device_id}) from {ip}")
        # Schedule dialog on main thread
        self.root.after(0, lambda: self._show_key_approval_dialog(device_id, device_name, ip))

    def _show_key_approval_dialog(self, device_id, device_name, ip):
        """Show a dialog asking user to approve SSH key installation."""
        try:
            import tkinter.messagebox as messagebox

            # Create approval dialog
            result = messagebox.askyesno(
                "SSH Key Approval Required",
                f"A new device wants to install an SSH key:\n\n"
                f"Device: {device_name}\n"
                f"IP Address: {ip}\n\n"
                f"This will allow the device to connect via SSH.\n\n"
                f"Do you want to approve this device?",
                icon='warning'
            )

            if result:
                # User approved
                if hasattr(self, 'data_receiver') and self.data_receiver:
                    response = self.data_receiver.approve_device(device_id)
                    if response.get('success'):
                        messagebox.showinfo("Approved", f"SSH key installed for {device_name}")
                    else:
                        messagebox.showwarning("Note", f"Device approved but key installation pending.\n{response.get('message', '')}")
            else:
                # User rejected
                if hasattr(self, 'data_receiver') and self.data_receiver:
                    self.data_receiver.reject_device(device_id)
                messagebox.showinfo("Rejected", f"SSH key request from {device_name} was rejected.")

        except Exception as e:
            log.error(f"Error showing key approval dialog: {e}")

    def refresh_projects_ui(self):
        """Refresh the projects list UI."""
        if not hasattr(self, "projects_container"):
            return

        for widget in self.projects_container.winfo_children():
            widget.destroy()

        projects = []
        if getattr(self, "selected_device_id", "__ALL__") == '__ALL__':
            for device in (self.devices or {}).values():
                for project in device.get('projects', []) or []:
                    if isinstance(project, dict):
                        p = dict(project)
                        p['_device_name'] = device.get('name') or device.get('id')
                        projects.append(p)
        else:
            device = (self.devices or {}).get(self.selected_device_id) or {}
            projects = device.get('projects', []) if isinstance(device.get('projects', []), list) else []

        if hasattr(self, "projects_count_label"):
            self.projects_count_label.configure(text=f"{len(projects)}")

        if not projects:
            tk.Label(
                self.projects_container,
                text="No project folders synced yet.\nProjects will appear here after your phone syncs.",
                bg=COLORS['bg_card'], fg=COLORS['text_dim'],
                font=('Segoe UI', 9),
                justify='center',
                anchor='center'
            ).pack(fill=tk.BOTH, expand=True, pady=20)
            return

        # Sort by last edited (most recent first)
        def get_project_timestamp(p):
            return p.get('updated_at') or p.get('updatedAt') or p.get('lastAccessed') or p.get('last_accessed') or 0
        projects.sort(key=get_project_timestamp, reverse=True)

        for project in projects:
            self._add_project_row(project)

    def _add_project_row(self, project):
        """Add a single project row to the UI. Double-click to open."""
        name = project.get('name', 'Unnamed')
        path = (
            project.get('path')
            or project.get('workingDirectory')
            or project.get('working_directory')
            or project.get('dir')
            or ''
        )
        device_name = project.get('_device_name')

        # Check if this is an openable PC path
        is_openable = is_pc_path(path) and path and (os.path.isdir(path) or os.path.exists(path))

        row = tk.Frame(self.projects_container, bg=COLORS['bg_input'], padx=8, pady=6,
                       cursor='hand2' if is_openable else 'arrow')
        row.pack(fill=tk.X, pady=2)

        info_frame = tk.Frame(row, bg=COLORS['bg_input'], cursor='hand2' if is_openable else 'arrow')
        info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        name_label = tk.Label(
            info_frame, text=name, bg=COLORS['bg_input'], fg=COLORS['text'],
            font=('Segoe UI', 9, 'bold'), anchor='w', cursor='hand2' if is_openable else 'arrow'
        )
        name_label.pack(anchor='w')

        if device_name and getattr(self, "selected_device_id", "__ALL__") == '__ALL__':
            tk.Label(
                info_frame, text=str(device_name), bg=COLORS['bg_input'], fg=COLORS['text_dim'],
                font=('Segoe UI', 8), anchor='w', cursor='hand2' if is_openable else 'arrow'
            ).pack(anchor='w')

        if is_pc_path(path) and path:
            path_display = path
            path_color = COLORS['text_dim'] if is_openable else COLORS['text_muted']
        else:
            path_display = f"Phone: {path}" if path else "Phone only"
            path_color = COLORS['warning']

        path_label = tk.Label(
            info_frame, text=path_display, bg=COLORS['bg_input'], fg=path_color,
            font=('Consolas', 8), anchor='w', wraplength=250, cursor='hand2' if is_openable else 'arrow'
        )
        path_label.pack(anchor='w')

        # Double-click to open folder
        if is_openable:
            def on_double_click(event, p=path):
                open_folder(p)

            for widget in [row, info_frame, name_label, path_label]:
                widget.bind('<Double-Button-1>', on_double_click)

            # Hover effect
            def on_enter(e):
                row.configure(bg=COLORS['bg_elevated'])
                info_frame.configure(bg=COLORS['bg_elevated'])
                for child in info_frame.winfo_children():
                    child.configure(bg=COLORS['bg_elevated'])

            def on_leave(e):
                row.configure(bg=COLORS['bg_input'])
                info_frame.configure(bg=COLORS['bg_input'])
                for child in info_frame.winfo_children():
                    child.configure(bg=COLORS['bg_input'])

            row.bind('<Enter>', on_enter)
            row.bind('<Leave>', on_leave)

            # Tooltip on hover
            self._create_tooltip(row, "Double-click to open")

    # =======================
    # Notes UI Methods
    # =======================

    def on_notes_received(self, device_id, notes):
        """Called when notes data is received from Android app."""
        if not isinstance(notes, list):
            return

        device = (self.notes_devices or {}).get(device_id, {})
        device.update({
            'id': device_id,
            'name': device.get('name', device_id),
            'ip': device.get('ip', None),
            'last_seen': time.time(),
            'notes': notes
        })
        self.notes_devices[device_id] = device

        self.refresh_notes_device_menu()
        self.refresh_notes_ui()

        # Notify web dashboard for real-time sync
        self._notify_web_dashboard_notes_updated(device_id)

    def _notify_web_dashboard_notes_updated(self, device_id):
        """Notify web dashboard that notes have been updated (triggers WebSocket broadcast)."""
        def do_notify():
            try:
                import urllib.request
                import json as json_module
                data = json_module.dumps({'device_id': device_id}).encode('utf-8')
                req = urllib.request.Request(
                    'http://127.0.0.1:6767/api/notes/sync',
                    data=data,
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                urllib.request.urlopen(req, timeout=2)
                log.debug(f"Notified web dashboard of notes update for {device_id}")
            except Exception as e:
                log.debug(f"Could not notify web dashboard: {e}")

        # Run in background thread to not block
        threading.Thread(target=do_notify, daemon=True).start()

    def refresh_notes_ui(self):
        """Refresh the notes list UI."""
        if not hasattr(self, "notes_container"):
            return

        for widget in self.notes_container.winfo_children():
            widget.destroy()

        notes = []
        if getattr(self, "selected_notes_device_id", "__ALL__") == '__ALL__':
            for device in (self.notes_devices or {}).values():
                for note in device.get('notes', []) or []:
                    if isinstance(note, dict):
                        n = dict(note)
                        n['_device_name'] = device.get('name') or device.get('id')
                        n['_device_id'] = device.get('id')
                        n['_device_ip'] = device.get('ip')
                        n['_note_content_port'] = device.get('note_content_port')
                        notes.append(n)
        else:
            device = (self.notes_devices or {}).get(self.selected_notes_device_id) or {}
            for note in device.get('notes', []) or []:
                if isinstance(note, dict):
                    n = dict(note)
                    n['_device_name'] = device.get('name') or device.get('id')
                    n['_device_id'] = device.get('id')
                    n['_device_ip'] = device.get('ip')
                    n['_note_content_port'] = device.get('note_content_port')
                    notes.append(n)

        if hasattr(self, "notes_count_label"):
            self.notes_count_label.configure(text=f"{len(notes)}")

        if not notes:
            tk.Label(
                self.notes_container,
                text="No notes synced yet.\nNotes will appear here after your phone syncs.",
                bg=COLORS['bg_card'], fg=COLORS['text_dim'],
                font=('Segoe UI', 9),
                justify='center',
                anchor='center'
            ).pack(fill=tk.BOTH, expand=True, pady=20)
            return

        # Sort by last edited (most recent first)
        def get_note_timestamp(n):
            return n.get('updated_at') or n.get('updatedAt') or n.get('created_at') or n.get('createdAt') or 0
        notes.sort(key=get_note_timestamp, reverse=True)

        for note in notes:
            self._add_note_row(note)

    def _add_note_row(self, note):
        """Add a single expandable note card to the UI."""
        title = note.get('title', 'Untitled')
        note_id = note.get('id', '')
        device_name = note.get('_device_name')
        device_ip = note.get('_device_ip')
        device_id = note.get('_device_id')
        note_port = note.get('_note_content_port')

        # Main card container
        card = tk.Frame(self.notes_container, bg=COLORS['bg_input'], padx=8, pady=6)
        card.pack(fill=tk.X, pady=2)
        card._is_expanded = False
        card._content_loaded = False
        card._note_content = ""

        # Header row (clickable)
        header = tk.Frame(card, bg=COLORS['bg_input'], cursor='hand2')
        header.pack(fill=tk.X)

        # Expand indicator
        expand_label = tk.Label(
            header, text="▶", bg=COLORS['bg_input'], fg=COLORS['text_dim'],
            font=('Segoe UI', 8), width=2
        )
        expand_label.pack(side=tk.LEFT)

        info_frame = tk.Frame(header, bg=COLORS['bg_input'])
        info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        title_label = tk.Label(
            info_frame, text=title, bg=COLORS['bg_input'], fg=COLORS['text'],
            font=('Segoe UI', 9, 'bold'), anchor='w', cursor='hand2'
        )
        title_label.pack(anchor='w')

        if device_name and getattr(self, "selected_notes_device_id", "__ALL__") == '__ALL__':
            tk.Label(
                info_frame, text=str(device_name), bg=COLORS['bg_input'], fg=COLORS['text_dim'],
                font=('Segoe UI', 8), anchor='w'
            ).pack(anchor='w')

        # Content frame (initially hidden)
        content_frame = tk.Frame(card, bg=COLORS['bg_card'], padx=8, pady=8)

        # Loading label
        loading_label = tk.Label(
            content_frame, text="Loading...", bg=COLORS['bg_card'], fg=COLORS['text_dim'],
            font=('Segoe UI', 9, 'italic')
        )

        # Button bar for copy action (stored on card for access in _display_note_content)
        button_bar = tk.Frame(content_frame, bg=COLORS['bg_card'])
        card._button_bar = button_bar

        def copy_note_content():
            """Copy note content to clipboard."""
            if card._note_content:
                self.root.clipboard_clear()
                self.root.clipboard_append(card._note_content)
                self.root.update()  # Required for clipboard to work
                # Show feedback
                copy_btn.configure(text="Copied!", fg=COLORS['success'])
                self.root.after(1500, lambda: copy_btn.configure(text="Copy", fg='#ffffff'))

        copy_btn = tk.Button(
            button_bar, text="Copy", bg=COLORS['accent'], fg='#ffffff',
            font=('Segoe UI', 8), relief='flat', cursor='hand2',
            padx=8, pady=2, bd=0, highlightthickness=0,
            activebackground=COLORS['accent_hover'], activeforeground='white',
            command=copy_note_content
        )
        copy_btn.pack(side=tk.RIGHT, padx=2)

        # Content text widget with scrollbar
        content_text = tk.Text(
            content_frame, bg=COLORS['bg_card'], fg=COLORS['text'],
            font=('Consolas', 9), wrap=tk.WORD, relief='flat',
            height=10, padx=4, pady=4, bd=0, highlightthickness=0,
            insertbackground=COLORS['text']
        )
        content_text.configure(state='disabled')  # Read-only

        # Right-click context menu for copy
        context_menu = tk.Menu(content_text, tearoff=0, bg=COLORS['bg_elevated'],
                               fg=COLORS['text'], activebackground=COLORS['accent'],
                               activeforeground='white', bd=0)
        context_menu.add_command(label="Copy All", command=copy_note_content)
        context_menu.add_command(label="Copy Selection", command=lambda: self._copy_selection(content_text))

        def show_context_menu(event):
            try:
                context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                context_menu.grab_release()

        content_text.bind('<Button-3>', show_context_menu)

        # Also enable Ctrl+C for copying selection
        def on_ctrl_c(event):
            self._copy_selection(content_text)
            return 'break'
        content_text.bind('<Control-c>', on_ctrl_c)

        def toggle_expand(event=None):
            if card._is_expanded:
                # Collapse
                content_frame.pack_forget()
                expand_label.configure(text="▶")
                card._is_expanded = False
            else:
                # Expand
                content_frame.pack(fill=tk.X, pady=(6, 0))
                expand_label.configure(text="▼")
                card._is_expanded = True

                # Load content if not already loaded
                if not card._content_loaded:
                    loading_label.pack(anchor='w')
                    self._fetch_note_content(
                        note_id, device_ip, note_port, device_id,
                        card, content_text, loading_label
                    )

        # Bind click events to header elements (single-click to expand)
        for widget in [header, expand_label, title_label, info_frame]:
            widget.bind('<Button-1>', toggle_expand)

        # Double-click to open in external editor
        def on_double_click(event):
            self.open_note(note_id, device_ip, title, note_port, device_id)

        for widget in [header, expand_label, title_label, info_frame]:
            widget.bind('<Double-Button-1>', on_double_click)

        # Tooltip
        self._create_tooltip(header, "Double-click to edit")

        # Hover effect on header
        def on_enter(e):
            header.configure(bg=COLORS['bg_elevated'])
            expand_label.configure(bg=COLORS['bg_elevated'])
            info_frame.configure(bg=COLORS['bg_elevated'])
            title_label.configure(bg=COLORS['bg_elevated'])
            for child in info_frame.winfo_children():
                child.configure(bg=COLORS['bg_elevated'])

        def on_leave(e):
            header.configure(bg=COLORS['bg_input'])
            expand_label.configure(bg=COLORS['bg_input'])
            info_frame.configure(bg=COLORS['bg_input'])
            title_label.configure(bg=COLORS['bg_input'])
            for child in info_frame.winfo_children():
                child.configure(bg=COLORS['bg_input'])

        header.bind('<Enter>', on_enter)
        header.bind('<Leave>', on_leave)

    def _fetch_note_content(self, note_id, device_ip, note_port, device_id, card, content_text, loading_label):
        """Fetch note content from local cache or device and display inline."""
        # Check local cache first (instant loading)
        cached = get_note_content_from_cache(note_id)
        if cached and cached.get('content'):
            self._display_note_content(card, content_text, loading_label, cached['content'])
            return

        if not device_ip:
            self._show_note_error(content_text, loading_label, "Device IP not available")
            return

        def fetch():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                port = NOTE_CONTENT_PORT
                if isinstance(note_port, int) and 1 <= note_port <= 65535:
                    port = note_port
                elif isinstance(device_id, str):
                    package_name = device_id.split(':')[-1]
                    if package_name.endswith('.release6'):
                        port = NOTE_CONTENT_PORT + 1
                    elif 'debug' in package_name:
                        port = NOTE_CONTENT_PORT + 2
                sock.connect((device_ip, port))

                request = json.dumps({'action': 'fetch_note', 'note_id': note_id}).encode('utf-8')
                sock.send(len(request).to_bytes(4, 'big'))
                sock.send(request)

                response_len = int.from_bytes(sock.recv(4), 'big')
                if response_len <= 0 or response_len > 1000000:
                    raise Exception("Invalid response length")

                response_data = b''
                while len(response_data) < response_len:
                    chunk = sock.recv(min(4096, response_len - len(response_data)))
                    if not chunk:
                        break
                    response_data += chunk

                sock.close()

                response = json.loads(response_data.decode('utf-8'))
                if not response.get('success'):
                    raise Exception(response.get('message', 'Failed to fetch note'))

                content = response.get('content', '')

                # Update UI on main thread
                self.root.after(0, lambda: self._display_note_content(
                    card, content_text, loading_label, content
                ))

            except socket.timeout:
                self.root.after(0, lambda: self._show_note_error(
                    content_text, loading_label, "Connection timeout - is the app open?"
                ))
            except (ConnectionRefusedError, ConnectionResetError, OSError) as e:
                self.root.after(0, lambda: self._show_note_error(
                    content_text, loading_label, "Could not connect to device"
                ))
            except Exception as e:
                self.root.after(0, lambda: self._show_note_error(
                    content_text, loading_label, f"Error: {str(e)}"
                ))

        threading.Thread(target=fetch, daemon=True).start()

    def _copy_selection(self, text_widget):
        """Copy selected text from a text widget to clipboard."""
        try:
            selection = text_widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            if selection:
                self.root.clipboard_clear()
                self.root.clipboard_append(selection)
                self.root.update()
        except tk.TclError:
            pass  # No selection

    def _display_note_content(self, card, content_text, loading_label, content):
        """Display fetched note content in the text widget."""
        loading_label.pack_forget()
        # Show the button bar (stored on card)
        if hasattr(card, '_button_bar'):
            card._button_bar.pack(fill=tk.X, pady=(0, 4))
        content_text.pack(fill=tk.BOTH, expand=True)
        content_text.configure(state='normal')
        content_text.delete('1.0', tk.END)
        content_text.insert('1.0', content)
        content_text.configure(state='disabled')
        card._content_loaded = True
        card._note_content = content

    def _show_note_error(self, content_text, loading_label, message):
        """Show error message in note content area."""
        loading_label.pack_forget()
        content_text.pack(fill=tk.BOTH, expand=True)
        content_text.configure(state='normal')
        content_text.delete('1.0', tk.END)
        content_text.insert('1.0', f"[Error] {message}")
        content_text.configure(state='disabled', fg=COLORS['error'] if 'error' in COLORS else '#ff6b6b')

    def open_note(self, note_id, device_ip, title, note_port=None, device_id=None):
        """Fetch note content from device and open in default editor."""
        if not device_ip:
            messagebox.showerror("Error", "Device IP not available.\nReconnect your device.")
            return

        def fetch_and_open():
            try:
                # Connect to device's NoteContentServer
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                port = NOTE_CONTENT_PORT
                if isinstance(note_port, int) and 1 <= note_port <= 65535:
                    port = note_port
                elif isinstance(device_id, str):
                    package_name = device_id.split(':')[-1]
                    if package_name.endswith('.release6'):
                        port = NOTE_CONTENT_PORT + 1
                    elif 'debug' in package_name:
                        port = NOTE_CONTENT_PORT + 2
                sock.connect((device_ip, port))

                # Send request
                request = json.dumps({'action': 'fetch_note', 'note_id': note_id}).encode('utf-8')
                sock.send(len(request).to_bytes(4, 'big'))
                sock.send(request)

                # Read response
                response_len = int.from_bytes(sock.recv(4), 'big')
                if response_len <= 0 or response_len > 1000000:
                    raise Exception("Invalid response length")

                response_data = b''
                while len(response_data) < response_len:
                    chunk = sock.recv(min(4096, response_len - len(response_data)))
                    if not chunk:
                        break
                    response_data += chunk

                sock.close()

                response = json.loads(response_data.decode('utf-8'))
                if not response.get('success'):
                    raise Exception(response.get('message', 'Failed to fetch note'))

                content = response.get('content', '')
                note_title = response.get('title', title)

                # Write to temp file and open
                safe_title = "".join(c for c in note_title if c.isalnum() or c in ' -_').strip()[:50]
                temp_dir = os.path.join(tempfile.gettempdir(), 'shadowai_notes')
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, f"{safe_title}_{note_id[:8]}.txt")

                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                if IS_WINDOWS:
                    os.startfile(temp_path)
                elif platform.system() == 'Darwin':
                    subprocess.run(['open', temp_path])
                else:
                    subprocess.run(['xdg-open', temp_path])

                log.info(f"Opened note: {note_title}")

            except socket.timeout:
                self.root.after(0, lambda: messagebox.showerror(
                    "Connection Timeout",
                    "Could not connect to device.\nMake sure the Shadow app is open."
                ))
            except (ConnectionRefusedError, ConnectionResetError, OSError) as e:
                log.error(f"Failed to connect to device: {e}")
                self.root.after(0, lambda: messagebox.showerror(
                    "Device Not Connected",
                    "Could not connect to the device.\n\nMake sure:\n- Your phone is on the same network\n- The Shadow app is open\n- The device hasn't gone to sleep"
                ))
            except Exception as e:
                log.error(f"Failed to open note: {e}")
                self.root.after(0, lambda: messagebox.showerror(
                    "Error",
                    f"Failed to open note:\n{str(e)}"
                ))

        # Run in background thread to avoid blocking UI
        threading.Thread(target=fetch_and_open, daemon=True).start()

    def refresh_notes_device_menu(self):
        """Refresh the Notes device selector menu."""
        if not hasattr(self, "notes_device_menu"):
            return

        if self._device_menu_updating:
            return

        self._device_menu_updating = True
        try:
            menu = self.notes_device_menu["menu"]
            menu.delete(0, "end")

            self._notes_device_label_to_id = {}
            id_to_label = {}

            def add_option(label, device_id_value):
                self._notes_device_label_to_id[label] = device_id_value
                id_to_label[device_id_value] = label
                menu.add_command(label=label, command=lambda l=label: self.notes_device_var.set(l))

            add_option("All devices", "__ALL__")

            for device_id, device in (self.notes_devices or {}).items():
                if isinstance(device, dict):
                    name = device.get('name') or device_id
                    label = f"{name}"
                    add_option(label, device_id)

            # Keep selection stable by device_id
            desired = self.selected_notes_device_id if getattr(self, "selected_notes_device_id", "__ALL__") else "__ALL__"
            self.notes_device_var.set(id_to_label.get(desired, "All devices"))
        finally:
            self._device_menu_updating = False

    def on_notes_device_selected(self):
        """Handle selection change in the Notes device selector."""
        if self._device_menu_updating:
            return
        if not hasattr(self, "notes_device_var"):
            return

        label = self.notes_device_var.get() or "All devices"
        self.selected_notes_device_id = getattr(self, "_notes_device_label_to_id", {}).get(label, "__ALL__")
        self.refresh_notes_ui()


def check_single_instance():
    """Check if another instance is already running. Returns lock socket if successful."""
    try:
        # Try to bind to a specific port - if it fails, another instance is running
        lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lock_socket.bind(('127.0.0.1', 19286))  # Use a dedicated port for instance lock
        lock_socket.listen(1)
        return lock_socket
    except socket.error:
        return None


def run_web_dashboard_server(open_browser: bool):
    """Run the web dashboard server (used by the --web-server mode)."""
    # Single-instance check using socket lock on port 6766
    lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock_socket.bind(('127.0.0.1', 6766))
        lock_socket.listen(1)
    except OSError:
        # Another instance is running - just open browser to existing
        print("Web dashboard already running, opening browser...")
        if open_browser:
            webbrowser.open("http://127.0.0.1:6767")
        return

    try:
        from web.app import create_app, socketio

        host = "127.0.0.1"
        port = 6767

        def open_browser_delayed():
            time.sleep(1.5)
            webbrowser.open(f"http://{host}:{port}")

        if open_browser:
            thread = threading.Thread(target=open_browser_delayed, daemon=True)
            thread.start()

        app = create_app()

        # Use socketio.run if available, otherwise fall back to Flask's app.run
        if getattr(app, 'socketio_enabled', False) and socketio is not None:
            socketio.run(
                app,
                host=host,
                port=port,
                debug=False,
                use_reloader=False,
                log_output=True
            )
        else:
            # Fallback for PyInstaller frozen builds where socketio may be disabled
            app.run(
                host=host,
                port=port,
                debug=False,
                use_reloader=False
            )
    except Exception:
        try:
            import traceback
            with open(WEB_LOG_FILE, "a", encoding="utf-8", errors="replace") as log_file:
                log_file.write("\n=== Web dashboard crash ===\n")
                log_file.write(traceback.format_exc())
                log_file.write("\n")
        except Exception:
            pass
        raise


def main():
    """Main entry point."""
    if WEB_SERVER_MODE:
        open_browser = "--no-browser" not in sys.argv
        run_web_dashboard_server(open_browser=open_browser)
        return

    start_minimized = "--minimized" in sys.argv

    # Check for existing instance
    lock_socket = check_single_instance()
    if lock_socket is None:
        # Another instance is already running
        if IS_WINDOWS:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, "ShadowBridge is already running.\n\nCheck your system tray.", "ShadowBridge", 0x40)
        else:
            print("ShadowBridge is already running.")
        sys.exit(0)

    if not HAS_PIL:
        print("Missing pillow. Install with: pip install pillow")
        sys.exit(1)

    log.info(f"ShadowBridge v{APP_VERSION} starting...")
    app = ShadowBridgeApp()
    log.info("ShadowBridge initialized")

    if start_minimized and HAS_TRAY:
        app.root.after(500, app.minimize_to_tray)

    app.run()


if __name__ == "__main__":
    main()
