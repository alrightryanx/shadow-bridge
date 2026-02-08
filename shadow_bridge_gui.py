#!/usr/bin/env python3
"""
ShadowBridge GUI - Desktop Quick Connect Tool
----------------------------------------------
A polished Windows GUI app that helps connect Shadow Android app to this PC.

Features:
- Compact popup-style UI near system tray
- QR code for instant app configuration
- Network discovery for automatic PC finding
- One-click Tools installation (Claude Code, Codex, Gemini)

Usage:
    python shadow_bridge_gui.py

Requirements:
    pip install qrcode pillow pystray
"""

import os
import sys
from pathlib import Path

# ---- Dynamic Path Detection for Relocated Modules ----
# web/ and ouroboros/ have been moved to the private shadow-android repo.
# Add shadow-android to sys.path so imports like `from web.services...` work.
def _setup_module_paths():
    """Add shadow-android directory to Python path for relocated modules."""
    # Determine base directory (where shadow-bridge and shadow-android live)
    if getattr(sys, 'frozen', False):
        # Running as compiled exe
        base_dir = Path(sys.executable).parent.parent
    else:
        # Running as script
        base_dir = Path(__file__).parent.parent

    # shadow-android should be a sibling directory
    shadow_android_dir = base_dir / "shadow-android"

    if shadow_android_dir.exists():
        # Add to front of path so it takes precedence
        shadow_android_str = str(shadow_android_dir)
        if shadow_android_str not in sys.path:
            sys.path.insert(0, shadow_android_str)
    else:
        # Fallback: check common locations
        for fallback in [
            Path("C:/shadow/shadow-android"),
            Path.home() / "shadow" / "shadow-android",
        ]:
            if fallback.exists():
                fallback_str = str(fallback)
                if fallback_str not in sys.path:
                    sys.path.insert(0, fallback_str)
                break

# Must be called before importing from web/ or ouroboros/
_setup_module_paths()

import json
import time
import socket
import platform
import subprocess
import threading
import shlex
import base64
import webbrowser
import logging
from logging.handlers import RotatingFileHandler
import tempfile
import ctypes
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

# Import Zeroconf for mDNS discovery
try:
    from zeroconf import IPVersion, ServiceInfo, Zeroconf

    HAS_ZEROCONF = True
except ImportError:
    HAS_ZEROCONF = False

# Import Ouroboros Sentinel
try:
    from ouroboros.sentinel import Sentinel

    HAS_SENTINEL = True
except ImportError:
    HAS_SENTINEL = False

# Import CLI handlers
from shadow_bridge.cli import (
    run_image_command,
    run_video_command,
    run_audio_command,
    run_assembly_command,
    run_browser_command,
)

# Import data service for bi-directional sync (web -> Android)
try:
    from web.services.data_service import (
        get_pending_sync_items,
        mark_items_synced,
        save_sessions_from_device,
        save_cards_from_device,
        save_collections_from_device,
        append_session_message,
        upsert_session,
    )

    SYNC_SERVICE_AVAILABLE = True
except ImportError:
    SYNC_SERVICE_AVAILABLE = False

# Import SingleInstance for robust multi-method instance locking
try:
    from shadow_bridge.utils.singleton import SingleInstance

    HAS_SINGLETON = True
except ImportError:
    HAS_SINGLETON = False
# Import effects for ping
try:
    from shadow_bridge.utils.effects import ConfettiEffect, play_ping_sound

    HAS_EFFECTS = True
except ImportError:
    HAS_EFFECTS = False


# Check for command modes
IMAGE_MODE = len(sys.argv) > 1 and sys.argv[1] == "image"
VIDEO_MODE = len(sys.argv) > 1 and sys.argv[1] == "video"
AUDIO_MODE = len(sys.argv) > 1 and sys.argv[1] == "audio"
ASSEMBLY_MODE = len(sys.argv) > 1 and sys.argv[1] == "assembly"
BROWSER_MODE = len(sys.argv) > 1 and sys.argv[1] == "browser"
WEB_SERVER_MODE = "--web-server" in sys.argv
DEBUG_BUILD = "--debug" in sys.argv
AIDEV_MODE = "--aidev" in sys.argv
AGENT_MODE = "--mode" in sys.argv and "agent" in sys.argv
PING_MODE = "--ping" in sys.argv
AUTO_INSTALL = "--auto-install" in sys.argv
HEADLESS_MODE = "--headless" in sys.argv
TRUST_ALL = "--trust-all" in sys.argv
TEST_MODE = os.environ.get("SHADOWAI_TESTING") == "1" or "PYTEST_CURRENT_TEST" in os.environ
if AGENT_MODE:
    AIDEV_MODE = True

# Environment Tier Detection
if AIDEV_MODE:
    ENVIRONMENT = "AIDEV"
    DATA_PORT = 19304
    WEB_PORT = 6769
    COMPANION_PORT = 19306
    DB_NAME = "shadow_aidev.db"
elif DEBUG_BUILD:
    ENVIRONMENT = "DEBUG"
    DATA_PORT = 19294
    WEB_PORT = 6768
    COMPANION_PORT = 19296
    DB_NAME = "shadow_debug.db"
else:
    ENVIRONMENT = "RELEASE"
    DATA_PORT = 19284
    WEB_PORT = 6767
    COMPANION_PORT = 19286
    DB_NAME = "shadow_ai.db"

SSH_KEY_PREFIX = f"# Shadow {ENVIRONMENT} device:"

# CLI Dispatch Logic
if IMAGE_MODE:
    from shadow_bridge.cli import run_image_command

    # Locate the image CLI executable
    install_path = os.path.dirname(os.path.abspath(__file__))
    image_cli_path = os.path.join(install_path, "shadow-image-cli.exe")
    run_image_command(sys.argv[1:], image_cli_path, install_path)
    sys.exit(0)

if VIDEO_MODE:
    from shadow_bridge.cli import run_video_command

    run_video_command(sys.argv[1:])
    sys.exit(0)

if AUDIO_MODE:
    from shadow_bridge.cli import run_audio_command

    run_audio_command(sys.argv[1:])
    sys.exit(0)

if ASSEMBLY_MODE:
    from shadow_bridge.cli import run_assembly_command

    run_assembly_command(sys.argv[1:])
    sys.exit(0)

if BROWSER_MODE:
    from shadow_bridge.cli import run_browser_command

    run_browser_command(sys.argv[1:])
    sys.exit(0)


def is_admin():
    """Check if running with administrator privileges."""
    if platform.system() != "Windows":
        return True  # Non-Windows doesn't need this check
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except (AttributeError, OSError):
        return False


def run_as_admin():
    """Re-launch the script with administrator privileges."""
    if platform.system() != "Windows":
        return False

    try:
        # Get the path to Python executable and this script
        if getattr(sys, "frozen", False):
            # Running as compiled exe
            script = sys.executable
            params = " ".join(sys.argv[1:])
        else:
            # Running as Python script
            script = sys.executable
            params = " ".join([f'"{sys.argv[0]}"'] + sys.argv[1:])

        # Use ShellExecuteW to request elevation
        ret = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",  # Request elevation
            script,
            params,
            None,
            1,  # SW_SHOWNORMAL
        )

        # If ShellExecuteW returns > 32, it succeeded
        return ret > 32
    except Exception as e:
        print(f"Failed to elevate: {e}")
        return False


# Auto-elevate on Windows if not already admin
if (
    platform.system() == "Windows"
    and not is_admin()
    and not WEB_SERVER_MODE
    and not (IMAGE_MODE or VIDEO_MODE or AUDIO_MODE or ASSEMBLY_MODE or BROWSER_MODE)
    and not TEST_MODE
):
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
                0x10 | 0x40000,  # MB_ICONERROR | MB_TOPMOST
            )
        except (AttributeError, OSError) as e:
            print(f"WARNING: Could not elevate ({e}). Please restart as Administrator.")
        sys.exit(1)


# Setup logging to file
HOME_DIR = os.path.expanduser("~")
LOG_DIR = os.path.join(HOME_DIR, ".shadowai_aidev" if AIDEV_MODE else ".shadowai")
LOG_FILE = os.path.join(LOG_DIR, "shadowbridge.log")
WEB_LOG_FILE = os.path.join(LOG_DIR, "shadowbridge_web.log")
os.makedirs(LOG_DIR, exist_ok=True)

# Configure rotating logs to prevent massive file growth
logging.basicConfig(
    level=logging.DEBUG if DEBUG_BUILD else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RotatingFileHandler(
            LOG_FILE,
            mode="a",
            maxBytes=5 * 1024 * 1024,
            backupCount=2,
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("ShadowBridge")

# Silence noisy libraries
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("zeroconf").setLevel(logging.WARNING)
if not DEBUG_BUILD:
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.INFO)

# Platform detection
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    import winreg
    import winsound

# Configuration
DISCOVERY_PORT = 19283
if ENVIRONMENT == "DEBUG":
    DISCOVERY_PORT = 19293
elif ENVIRONMENT == "AIDEV":
    DISCOVERY_PORT = 19303

DISCOVERY_MAGIC = b"SHADOWAI_DISCOVER"
# DATA_PORT, WEB_PORT, COMPANION_PORT are set above based on environment
NOTE_CONTENT_PORT = 19285
if ENVIRONMENT == "DEBUG":
    NOTE_CONTENT_PORT = 19295
elif ENVIRONMENT == "AIDEV":
    NOTE_CONTENT_PORT = 19305

APP_NAME = f"ShadowBridge{ENVIRONMENT}" if ENVIRONMENT != "RELEASE" else "ShadowBridge"
APP_VERSION = "1.209"
SYNC_SCHEMA_VERSION = 2
SYNC_SCHEMA_MIN_VERSION = 1
# Windows Registry path for autostart
PROJECTS_FILE = os.path.join(HOME_DIR, ".shadowai", "projects.json")
NOTES_FILE = os.path.join(HOME_DIR, ".shadowai", "notes.json")
AGENTS_FILE = os.path.join(HOME_DIR, ".shadowai", "agents.json")
TASKS_FILE = os.path.join(HOME_DIR, ".shadowai", "tasks.json")
AUTOMATIONS_FILE = os.path.join(HOME_DIR, ".shadowai", "automations.json")
WINDOW_STATE_FILE = os.path.join(HOME_DIR, ".shadowai", "window_state.json")
SETTINGS_FILE = os.path.join(HOME_DIR, ".shadowai", "settings.json")

if IS_WINDOWS:
    INSTALL_PATH_FILE = os.path.join(
        os.environ.get("APPDATA", os.path.expanduser("~")),
        "ShadowBridge",
        "install_path.txt",
    )
else:
    INSTALL_PATH_FILE = os.path.join(HOME_DIR, ".shadowai", "install_path.txt")


def register_install_path():
    """
    Write the ShadowBridge install directory to a known location.
    This allows the Android app to find us via SSH for image generation.
    Also copies shadow_image_cli.py to the install directory if running as exe.
    """
    try:
        # Get the directory where this script/exe is located
        if getattr(sys, "frozen", False):
            # Running as compiled exe
            install_dir = os.path.dirname(sys.executable)
        else:
            # Running as Python script
            install_dir = os.path.dirname(os.path.abspath(__file__))

        # Ensure directory exists
        os.makedirs(os.path.dirname(INSTALL_PATH_FILE), exist_ok=True)

        # Write the path
        with open(INSTALL_PATH_FILE, "w", encoding="utf-8") as f:
            f.write(install_dir)

        log.info(f"Registered install path: {install_dir}")

        # Copy shadow_image_cli.py to install directory if it doesn't exist there
        # This is needed when running as frozen exe (dist folder)
        cli_script = os.path.join(install_dir, "shadow_image_cli.py")
        if not os.path.exists(cli_script):
            # Try to find it in common locations
            source_locations = [
                os.path.join(
                    os.path.dirname(install_dir), "shadow_image_cli.py"
                ),  # Parent dir (for dist/)
                os.path.join(install_dir, "..", "shadow_image_cli.py"),
            ]
            for source in source_locations:
                if os.path.exists(source):
                    import shutil

                    shutil.copy2(source, cli_script)
                    log.info(f"Copied shadow_image_cli.py to {install_dir}")
                    break
    except Exception as e:
        log.warning(f"Failed to register install path: {e}")


# GitHub release URL for auto-updates
GITHUB_REPO = "alrightryanx/shadow-bridge"
GITHUB_RELEASES_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

# Theme colors - M3 Dark Theme with Claude Terracotta
COLORS = {
    # Backgrounds - warm dark tones
    "bg_dark": "#1a1714",
    "bg_surface": "#211d19",
    "bg_card": "#252019",
    "bg_elevated": "#2a251f",
    "bg_input": "#2a251f",
    # Accent colors - Claude terracotta
    "accent": "#D97757",
    "accent_hover": "#ff8a65",  # Brighter orange for clear hover state
    "accent_light": "#e8967a",
    "accent_container": "#3d2a20",
    # Status colors - softer, M3 style
    "success": "#81c784",
    "success_dim": "#2e4a2f",
    "warning": "#ffb74d",
    "warning_dim": "#4a3a1f",
    "error": "#e8967a",
    "error_dim": "#3d2a22",
    # Text colors
    "text": "#faf6f1",
    "text_secondary": "#b8a99a",
    "text_dim": "#8a7a6a",
    "text_muted": "#5a4a3a",
    # Borders and dividers
    "border": "#3d352c",
    "border_light": "#4d453c",
    "divider": "#2a251f",
}

# Enable DPI awareness on Windows BEFORE importing tkinter
if platform.system() == "Windows":
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
        root.update_idletasks()
        # Get actual window handle (GA_ROOT = 2)
        hwnd = ctypes.windll.user32.GetAncestor(root.winfo_id(), 2)

        # Dark title bar (DWMWA_USE_IMMERSIVE_DARK_MODE)
        use_dark = ctypes.c_int(1)
        for attr in (20, 19):  # 20 is Win10 20H1+, 19 is older fallback
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(use_dark), ctypes.sizeof(use_dark)
            )

        # Mica backdrop (Windows 11) - DWMWA_SYSTEMBACKDROP_TYPE = 38
        # backdrop = ctypes.c_int(2)  # DWMSBT_MAINWINDOW (Mica)
        # ctypes.windll.dwmapi.DwmSetWindowAttribute(
        #     hwnd, 38, ctypes.byref(backdrop), ctypes.sizeof(backdrop)
        # )
        log.info(f"[OK] Applied Windows 11 dark theme to HWND {hwnd}")
    except Exception as e:
        log.warning(f"Could not apply Windows 11 theme: {e}")


# Try to import optional dependencies
try:
    import tkinter as tk
    from tkinter import ttk, messagebox

    HAS_TK = True
except ImportError:
    HAS_TK = False
    log.critical("tkinter not available - cannot start GUI")
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


_cached_all_ips = None
_last_ips_refresh = 0


def get_all_ips():
    """Get all available IP addresses from all network interfaces - cached for 30s."""
    global _cached_all_ips, _last_ips_refresh

    now = time.time()
    # Reduced from 30s to 10s for more responsive IP changes on dynamic networks
    if _cached_all_ips and (now - _last_ips_refresh < 10):
        return _cached_all_ips

    ips = {
        "local": [],  # Private LAN IPs (192.168.x.x, 10.x.x.x, 172.16-31.x.x)
        "tailscale": [],  # Tailscale IPs (100.x.x.x)
        "vpn": [],  # Common VPN ranges
        "other": [],  # Other IPs
    }

    try:
        # Get all IPs from hostname
        hostname = socket.gethostname()
        all_ips = socket.getaddrinfo(hostname, None, socket.AF_INET)

        for item in all_ips:
            ip = item[4][0]
            if ip.startswith("127."):
                continue  # Skip loopback

            # Categorize IP
            if ip.startswith("100.64.") or ip.startswith("100."):
                ips["tailscale"].append(ip)
            elif ip.startswith("10.") and (
                ip.startswith("10.8.") or ip.startswith("10.9.")
            ):
                ips["vpn"].append(ip)  # Common OpenVPN range
            elif (
                ip.startswith("192.168.")
                or ip.startswith("10.")
                or ip.startswith("172.")
            ):
                ips["local"].append(ip)
            else:
                ips["other"].append(ip)
        log.debug(f"get_all_ips: Found {len(all_ips)} IPs via getaddrinfo")
    except Exception as e:
        log.warning(f"get_all_ips: getaddrinfo failed: {e}")

    # Also try socket method for primary IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)  # Fast response timeout
        s.connect(("8.8.8.8", 80))
        primary_ip = s.getsockname()[0]
        s.close()

        # Add to appropriate category if not already there
        found = False
        for category in ips.values():
            if primary_ip in category:
                found = True
                break
        if not found and not primary_ip.startswith("127."):
            if (
                primary_ip.startswith("192.168.")
                or primary_ip.startswith("10.")
                or primary_ip.startswith("172.")
            ):
                ips["local"].insert(0, primary_ip)
                log.debug(f"get_all_ips: Primary LAN IP detected: {primary_ip}")
            else:
                ips["other"].insert(0, primary_ip)
                log.debug(f"get_all_ips: Primary other IP detected: {primary_ip}")
    except Exception as e:
        log.warning(f"get_all_ips: UDP socket detection failed: {e}")

    _cached_all_ips = ips
    _last_ips_refresh = now
    return ips


def get_local_ip(wait=False):
    """Get primary local IP address - non-blocking cached version unless wait=True."""
    global _cached_local_ip

    if _cached_local_ip and _cached_local_ip != "Detecting...":
        return _cached_local_ip

    # Return placeholder while detecting in background if not already done
    if not hasattr(get_local_ip, "_detecting"):
        get_local_ip._detecting = True

        def detect():
            global _cached_local_ip
            candidate_ip = None

            try:
                # 1. Prefer IPv4 Route (Best for legacy SSH/QR codes)
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.settimeout(0.2)  # Reduced from 1.0s for faster response
                    s.connect(("8.8.8.8", 80))
                    ip = s.getsockname()[0]
                    s.close()
                    # Prioritize LAN over Tailscale for the primary displayed IP
                    if (
                        not ip.startswith("127.")
                        and not ip.startswith("169.254.")
                        and not ip.startswith("100.")
                    ):
                        candidate_ip = ip
                except Exception:
                    pass

                # 2. Tailscale Fallback (If LAN unavailable or route points there)
                if not candidate_ip:
                    ts_ip = get_tailscale_ip()
                    if ts_ip and ts_ip != "Detecting...":
                        candidate_ip = ts_ip

                # 3. IPv6 Route (If IPv4 unavailable)
                if not candidate_ip:
                    try:
                        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
                        s.settimeout(0.2)  # Reduced from 1.0s for faster response
                        s.connect(("2001:4860:4860::8888", 80))
                        ip = s.getsockname()[0]
                        s.close()
                        # Exclude Link-Local (fe80:) as they require Scope IDs which are hard for clients
                        if not ip.startswith("fe80:") and not ip.startswith("::1"):
                            candidate_ip = ip
                    except Exception:
                        pass

                # 4. Fallback: Iterate all interfaces (AF_UNSPEC for v4 and v6)
                if not candidate_ip:
                    try:
                        hostname = socket.gethostname()
                        # AF_UNSPEC gets both v4 and v6
                        all_ips = socket.getaddrinfo(
                            hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
                        )

                        ipv4_candidates = []
                        ipv6_candidates = []

                        for item in all_ips:
                            ip = item[4][0]
                            # Skip loopback
                            if ip.startswith("127.") or ip == "::1":
                                continue
                            # Skip APIPA
                            if ip.startswith("169.254."):
                                continue

                            if ":" in ip:  # IPv6
                                ipv6_candidates.append(ip)
                            else:  # IPv4
                                ipv4_candidates.append(ip)

                        # Preference logic:
                        # 1. 192.168.x.x (Standard Home)
                        # 2. 10.x.x.x (Corporate/VPN)
                        # 3. 172.x.x.x (Docker/Corporate - lower priority as often virtual)
                        # 4. Global IPv6
                        # 5. Tailscale (100.x.x.x)
                        # 6. Other IPv4

                        prio_192 = [
                            i for i in ipv4_candidates if i.startswith("192.168.")
                        ]
                        prio_10 = [
                            i
                            for i in ipv4_candidates
                            if i.startswith("10.") and not i.startswith("100.")
                        ]
                        prio_172 = [i for i in ipv4_candidates if i.startswith("172.")]
                        prio_ts = [i for i in ipv4_candidates if i.startswith("100.")]

                        if prio_192:
                            candidate_ip = prio_192[0]
                        elif prio_10:
                            candidate_ip = prio_10[0]
                        elif ipv6_candidates:
                            global_v6 = [
                                i for i in ipv6_candidates if not i.startswith("fe80:")
                            ]
                            candidate_ip = (
                                global_v6[0] if global_v6 else ipv6_candidates[0]
                            )
                        elif prio_172:
                            candidate_ip = prio_172[0]
                        elif prio_ts:
                            candidate_ip = prio_ts[0]
                        elif ipv4_candidates:
                            candidate_ip = ipv4_candidates[0]

                    except Exception:
                        pass

                # 5. Final Fallback
                if not candidate_ip:
                    candidate_ip = "127.0.0.1"

                _cached_local_ip = candidate_ip
                log.info(
                    f"Detected primary Local IP (LAN prioritized): {_cached_local_ip}"
                )

            except Exception as e:
                log.error(f"IP detection failed: {e}")
                _cached_local_ip = "127.0.0.1"
            finally:
                get_local_ip._detecting = False

        if wait:
            detect()
        else:
            threading.Thread(target=detect, daemon=True).start()

    if wait and (not _cached_local_ip or _cached_local_ip == "Detecting..."):
        pass

    return _cached_local_ip


_cached_local_ip = None


_cached_tailscale_ip = None


def get_tailscale_ip():
    """Get Tailscale IP if available (stable 100.x.x.x address) - non-blocking cached version."""
    global _cached_tailscale_ip

    # Return cached value if we have it
    if _cached_tailscale_ip:
        return _cached_tailscale_ip if _cached_tailscale_ip != "Detecting..." else None

    # First check our detected IPs (fast, non-blocking)
    all_ips = get_all_ips()
    if all_ips["tailscale"]:
        _cached_tailscale_ip = all_ips["tailscale"][0]
        return _cached_tailscale_ip

    # Try tailscale CLI in background if not already detecting
    if not hasattr(get_tailscale_ip, "_detecting"):
        get_tailscale_ip._detecting = True
        _cached_tailscale_ip = "Detecting..."

        def detect():
            global _cached_tailscale_ip
            try:
                # Try tailscale CLI
                result = subprocess.run(
                    ["tailscale", "ip", "-4"],
                    capture_output=True,
                    text=True,
                    timeout=0.3,  # Reduced from 0.5s to 0.3s for faster response
                    creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0,
                )
                if result.returncode == 0:
                    ip = result.stdout.strip().split("\n")[0]
                    if ip.startswith("100."):
                        _cached_tailscale_ip = ip
                        log.info(f"Tailscale IP detected: {ip}")
                        return
            except Exception:
                pass

            # If we get here, detection failed
            _cached_tailscale_ip = None
            # Reset detecting flag after some time to allow retry (increased from 60s to 120s)
            time.sleep(120)
            delattr(get_tailscale_ip, "_detecting")

        threading.Thread(target=detect, daemon=True).start()

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
            ["zerotier-cli", "listnetworks"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0,
        )
        if result.returncode == 0 and "200" in result.stdout:
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


_cached_ssh_port = None


def find_ssh_port():
    """Find which port SSH is running on - non-blocking cached version."""
    global _cached_ssh_port

    if _cached_ssh_port:
        return _cached_ssh_port if _cached_ssh_port != "Detecting..." else None

    # Start background detection if not already running
    if not hasattr(find_ssh_port, "_detecting"):
        find_ssh_port._detecting = True
        _cached_ssh_port = "Detecting..."

        def detect():
            global _cached_ssh_port
            try:
                # Fast path in thread: Check common ports directly
                common_ports = [22, 2222, 2269, 22022, 8022]
                for port in common_ports:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(
                        0.02
                    )  # Reduced from 0.05s to 0.02s for faster response
                    try:
                        if sock.connect_ex(("127.0.0.1", port)) == 0:
                            sock.close()
                            _cached_ssh_port = port
                            log.info(f"SSH port detected: {port}")
                            return
                    except Exception:
                        pass
                    finally:
                        sock.close()

                # Slow path: Check sshd_config
                sshd_config_paths = [
                    r"C:\ProgramData\ssh\sshd_config",
                    r"C:\Windows\System32\OpenSSH\sshd_config",
                    os.path.expanduser("~/.ssh/sshd_config"),
                ]
                for config_path in sshd_config_paths:
                    try:
                        if os.path.exists(config_path):
                            with open(config_path, "r", encoding="utf-8") as f:
                                for line in f:
                                    line = line.strip()
                                    if line.startswith("Port ") and not line.startswith(
                                        "#"
                                    ):
                                        try:
                                            port = int(line.split()[1])
                                            # Verify it's listening
                                            sock = socket.socket(
                                                socket.AF_INET, socket.SOCK_STREAM
                                            )
                                            sock.settimeout(
                                                0.05
                                            )  # Reduced from 0.1s to 0.05s for faster response
                                            if (
                                                sock.connect_ex(("127.0.0.1", port))
                                                == 0
                                            ):
                                                sock.close()
                                                _cached_ssh_port = port
                                                log.info(
                                                    f"SSH port found in config: {port}"
                                                )
                                                return
                                            sock.close()
                                        except ValueError:
                                            pass
                    except Exception:
                        continue

                # Fallback: netstat (slowest)
                try:
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    result = subprocess.run(
                        ["netstat", "-an"],
                        capture_output=True,
                        text=True,
                        timeout=2,  # Reduced from 5s to 2s for faster response
                        startupinfo=startupinfo,
                    )
                    if result.returncode == 0:
                        for line in result.stdout.splitlines():
                            if "TCP" in line and "LISTENING" in line:
                                parts = line.split()
                                local_addr = parts[1]
                                if ":" in local_addr:
                                    port_str = local_addr.split(":")[-1]
                                    try:
                                        port = int(port_str)
                                        if (
                                            port in [22, 2222, 2269, 22022, 8022]
                                            or port > 1000
                                        ):
                                            # Double check this port
                                            sock = socket.socket(
                                                socket.AF_INET, socket.SOCK_STREAM
                                            )
                                            sock.settimeout(
                                                0.05
                                            )  # Reduced from 0.1s to 0.05s for faster response
                                            if (
                                                sock.connect_ex(("127.0.0.1", port))
                                                == 0
                                            ):
                                                sock.close()
                                                _cached_ssh_port = port
                                                log.info(
                                                    f"SSH port found via netstat: {port}"
                                                )
                                                return
                                            sock.close()
                                    except ValueError:
                                        continue
                except Exception:
                    pass
            finally:
                if _cached_ssh_port == "Detecting...":
                    _cached_ssh_port = None
                # Allow retry after some time (increased from 60s to 120s)
                time.sleep(120)
                delattr(find_ssh_port, "_detecting")

        threading.Thread(target=detect, daemon=True).start()

    return None


_cached_ssh_status = None
_last_ssh_check_time = 0


def check_ssh_running(port=22):
    """Check if SSH server is running - non-blocking cached version."""
    global _cached_ssh_status, _last_ssh_check_time

    now = time.time()
    # Return cached value if it's fresh (increased from 15s to 30s to reduce checks)
    if _cached_ssh_status is not None and (now - _last_ssh_check_time < 30):
        return _cached_ssh_status

    # Start background detection if not already running
    if not hasattr(check_ssh_running, "_detecting") or not check_ssh_running._detecting:
        check_ssh_running._detecting = True

        def detect():
            global _cached_ssh_status, _last_ssh_check_time
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)  # Reduced from 0.3s to 0.1s for faster response
            try:
                result = sock.connect_ex(("127.0.0.1", port))
                _cached_ssh_status = result == 0
                _last_ssh_check_time = time.time()
            except Exception:
                _cached_ssh_status = False
            finally:
                sock.close()
                check_ssh_running._detecting = False

        threading.Thread(target=detect, daemon=True).start()

    # Return last known status or False while first/new check runs
    return _cached_ssh_status if _cached_ssh_status is not None else False


def setup_firewall_rule():
    """Add Windows Firewall rules for necessary ports."""
    if not IS_WINDOWS:
        return True
    try:
        # Rules to ensure are present: (Name, Protocol, Port)
        required_rules = [
            ("ShadowBridge Discovery", "UDP", "19283"),
            ("ShadowBridge Data Receiver", "TCP", str(DATA_PORT)),
            ("ShadowBridge Companion", "TCP", str(COMPANION_PORT)),
            ("ShadowBridge Dashboard", "TCP", str(WEB_PORT)),
        ]

        # Check existing rules
        check = subprocess.run(
            ["netsh", "advfirewall", "firewall", "show", "rule", "name=all"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        existing_rules = check.stdout

        all_success = True
        for name, proto, port in required_rules:
            if name not in existing_rules:
                log.info(f"Adding firewall rule: {name} ({proto} {port})")
                result = subprocess.run(
                    [
                        "netsh",
                        "advfirewall",
                        "firewall",
                        "add",
                        "rule",
                        f"name={name}",
                        "dir=in",
                        "action=allow",
                        f"protocol={proto}",
                        f"localport={port}",
                    ],
                    capture_output=True,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                if result.returncode != 0:
                    all_success = False
                    log.error(f"Failed to add firewall rule {name}: {result.stderr}")

        return all_success
    except Exception as e:
        log.error(f"Firewall setup error: {e}")
        return False


def check_npm_installed():
    """Check if npm is installed."""
    try:
        result = subprocess.run(
            ["npm", "--version"],
            capture_output=True,
            text=True,
            timeout=3,  # Reduced from 10s to 3s for faster response
            creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0,
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
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            env_key_path,
            0,
            winreg.KEY_READ | winreg.KEY_SET_VALUE,
        )
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
                    proc_parts = {
                        os.path.normpath(p.strip())
                        for p in proc_path.split(";")
                        if p.strip()
                    }
                    if entry not in proc_parts:
                        os.environ["PATH"] = (
                            (proc_path + ";" + entry) if proc_path else entry
                        )
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
                    None,
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
            ["winget", "--version"],
            capture_output=True,
            text=True,
            timeout=3,  # Reduced from 10s to 3s for faster response
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_python_pip_installed():
    """Check if pip is available via either the Windows 'py' launcher or 'python'."""
    # Use array form for each candidate to avoid shell=True
    candidates = [
        ["py", "-m", "pip", "--version"],
        ["python", "-m", "pip", "--version"],
    ]
    flags = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
    for cmd in candidates:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3,
                creationflags=flags,  # Reduced from 10s to 3s
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
        "claude": ["claude", "claude-code"],
        "codex": ["codex"],
        "gemini": ["gemini", "gemini-cli"],
        "aider": ["aider"],
        "ollama": ["ollama"],
        "opencode": ["opencode"],
    }

    commands_to_try = tool_commands.get(tool_name, [tool_name])

    # Check npm global directories directly on Windows
    if IS_WINDOWS:
        npm_paths = [
            os.path.join(os.environ.get("APPDATA", ""), "npm"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "npm"),
            os.path.join(
                os.environ.get("USERPROFILE", ""), "AppData", "Roaming", "npm"
            ),
        ]
        for cmd in commands_to_try:
            for npm_path in npm_paths:
                # Check for .cmd file (Windows npm installs)
                cmd_path = os.path.join(npm_path, f"{cmd}.cmd")
                if os.path.isfile(cmd_path):
                    return True
                # Check for .ps1 file
                ps1_path = os.path.join(npm_path, f"{cmd}.ps1")
                if os.path.isfile(ps1_path):
                    return True
                # Check for plain executable
                exe_path = os.path.join(npm_path, f"{cmd}.exe")
                if os.path.isfile(exe_path):
                    return True

    # Use shutil.which instead of shell 'where' command (safer)
    import shutil

    for cmd in commands_to_try:
        if shutil.which(cmd):
            return True

        # Try running with --version (array form, no shell=True)
        try:
            result = subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=3,  # Reduced from 10s to 3s for faster fallback checks
                creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0,
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
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, get_startup_registry_key(), 0, winreg.KEY_READ
        )
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
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            get_startup_registry_key(),
            0,
            winreg.KEY_SET_VALUE,
        )
        try:
            if enabled:
                exe_path = sys.executable if getattr(sys, "frozen", False) else None
                if exe_path:
                    winreg.SetValueEx(
                        key, APP_NAME, 0, winreg.REG_SZ, f'"{exe_path}" --minimized'
                    )
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


# ============ Settings Management ============


def load_settings():
    """Load settings from JSON file."""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log.warning(f"Failed to load settings: {e}")
    return {"auto_update": True}  # Default settings


def save_settings(settings):
    """Save settings to JSON file."""
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        log.warning(f"Failed to save settings: {e}")
        return False


def is_auto_update_enabled():
    """Check if auto-update is enabled."""
    settings = load_settings()
    return settings.get("auto_update", True)


def set_auto_update_enabled(enabled):
    """Enable or disable auto-update."""
    settings = load_settings()
    settings["auto_update"] = enabled
    return save_settings(settings)


# ============ Auto-Update System ============


def check_for_updates():
    """Check GitHub releases for a newer version. Returns (has_update, latest_version, download_url) or (False, None, None) on error."""
    try:
        import urllib.request
        import ssl

        # Create SSL context that works with GitHub
        ctx = ssl.create_default_context()

        req = urllib.request.Request(
            GITHUB_RELEASES_API, headers={"User-Agent": f"ShadowBridge/{APP_VERSION}"}
        )

        with urllib.request.urlopen(req, timeout=10, context=ctx) as response:
            data = json.loads(response.read().decode("utf-8"))

        latest_tag = data.get("tag_name", "").lstrip("v")

        # Find the EXE asset
        download_url = None
        for asset in data.get("assets", []):
            if asset.get("name", "").endswith(".exe"):
                download_url = asset.get("browser_download_url")
                break

        # Compare versions (simple string comparison works for X.XXX format)
        if latest_tag and latest_tag > APP_VERSION:
            return True, latest_tag, download_url

        return False, latest_tag, None

    except Exception as e:
        log.warning(f"Failed to check for updates: {e}")
        return False, None, None


def download_update(download_url, progress_callback=None):
    """Download update to temp folder. Returns path to downloaded file or None on error."""
    try:
        import urllib.request
        import ssl

        ctx = ssl.create_default_context()

        req = urllib.request.Request(
            download_url, headers={"User-Agent": f"ShadowBridge/{APP_VERSION}"}
        )

        # Download to temp directory
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "ShadowBridge_update.exe")

        with urllib.request.urlopen(req, timeout=120, context=ctx) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 8192

            with open(temp_path, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_size:
                        progress_callback(downloaded / total_size)

        return temp_path

    except Exception as e:
        log.error(f"Failed to download update: {e}")
        return None


def apply_update(update_path):
    """Apply the downloaded update. Creates a batch script to replace EXE after app closes."""
    try:
        if not os.path.exists(update_path):
            return False

        current_exe = sys.executable if getattr(sys, "frozen", False) else None
        if not current_exe:
            log.warning("Cannot apply update: not running as frozen executable")
            return False

        # Create batch script to replace EXE
        batch_path = os.path.join(tempfile.gettempdir(), "shadowbridge_update.bat")

        batch_content = f'''@echo off
echo Updating ShadowBridge...
timeout /t 2 /nobreak >nul

:waitloop
tasklist /FI "IMAGENAME eq ShadowBridge.exe" 2>NUL | find /I /N "ShadowBridge.exe">NUL
if "%ERRORLEVEL%"=="0" (
    timeout /t 1 /nobreak >nul
    goto waitloop
)

copy /Y "{update_path}" "{current_exe}"
if errorlevel 1 (
    echo Update failed!
    pause
    exit /b 1
)

del "{update_path}"
start "" "{current_exe}"
del "%~f0"
'''

        with open(batch_path, "w", encoding="utf-8") as f:
            f.write(batch_content)

        # Launch the batch script (hidden)
        subprocess.Popen(
            ["cmd", "/c", batch_path],
            creationflags=subprocess.CREATE_NO_WINDOW,
            shell=False,
        )

        return True

    except Exception as e:
        log.error(f"Failed to apply update: {e}")
        return False


def get_app_icon_path():
    """Get the path to the app icon."""
    # Check various locations for logo.png
    possible_paths = [
        Path(__file__).parent / "logo.png",
        Path(sys.executable).parent / "logo.png"
        if getattr(sys, "frozen", False)
        else None,
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
            img = img.resize(
                (size, size),
                Image.Resampling.LANCZOS
                if hasattr(Image, "Resampling")
                else Image.LANCZOS,
            )
            return img.convert("RGBA")
        except Exception:
            pass

    # Fallback: create programmatic icon
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([2, 2, size - 2, size - 2], fill="#1a1714", outline="#D97757", width=2)
    center = size // 2
    points = [
        (center + 8, center - 18),
        (center - 4, center - 2),
        (center + 4, center - 2),
        (center - 8, center + 18),
        (center + 4, center + 2),
        (center - 4, center + 2),
    ]
    draw.polygon(points, fill="#D97757")
    return img


import upnpy
import requests


class UPnPManager:
    """
    Handles automatic port forwarding via UPnP and Public IP discovery.
    Ensures SSH (22) and Signaling (19287) are reachable from outside.
    """

    def __init__(self):
        self.upnp = upnpy.UPnP()
        self.public_ip = None
        self.is_mapped = False
        self.mappings = [
            {"port": 22, "proto": "TCP", "desc": "Shadow SSH"},
            {"port": 19287, "proto": "UDP", "desc": "Shadow Signaling"},
        ]

    def discover_and_map(self):
        """Discover UPnP devices and map required ports."""
        threading.Thread(target=self._run_discovery, daemon=True).start()

    def _run_discovery(self):
        try:
            # 1. Discover Public IP (Reliable way)
            try:
                response = requests.get("https://api.ipify.org?format=json", timeout=5)
                if response.status_code == 200:
                    self.public_ip = response.json().get("ip")
                    log.info(f"Public IP discovered: {self.public_ip}")
            except Exception as e:
                log.warning(f"Could not discover Public IP: {e}")

            # 2. UPnP Port Mapping
            devices = self.upnp.discover()
            if not devices:
                log.info("No UPnP devices found on network.")
                return

            # Get the IGD (Internet Gateway Device)
            device = self.upnp.get_igd()
            if not device:
                log.info("No Internet Gateway Device found for UPnP.")
                return

            # Get the AddPortMapping service
            service = device.get_service("WANIPConnection.1") or device.get_service(
                "WANPPPConnection.1"
            )
            if not service:
                log.info("UPnP Connection service not found.")
                return

            local_ip = get_local_ip()
            success_count = 0

            for mapping in self.mappings:
                try:
                    # Check if already mapped (best effort)
                    # We try to add it; if it exists, most routers just update/ignore
                    service.AddPortMapping(
                        NewRemoteHost="",
                        NewExternalPort=mapping["port"],
                        NewProtocol=mapping["proto"],
                        NewInternalPort=mapping["port"],
                        NewInternalClient=local_ip,
                        NewEnabled=1,
                        NewPortMappingDescription=mapping["desc"],
                        NewLeaseDuration=0,  # Permanent until reboot
                    )
                    log.info(
                        f"UPnP: Mapped {mapping['proto']} {mapping['port']} to {local_ip}"
                    )
                    success_count += 1
                except Exception as e:
                    log.warning(f"UPnP mapping failed for {mapping['port']}: {e}")

            self.is_mapped = success_count > 0
            if self.is_mapped:
                log.info(
                    f"UPnP auto-mapping successful ({success_count}/{len(self.mappings)})"
                )

        except Exception as e:
            log.error(f"UPnP discovery error: {e}")

    def get_public_ip(self):
        return self.public_ip


class SignalingService(threading.Thread):
    """
    UDP Signaling Service for rapid roaming and instant discovery.
    Listens for heartbeats from the Android app and maintains active
    NAT mappings to ensure SSH can connect instantly.
    """

    PORT = 19287
    MAGIC = b"SHADOW_SIGNAL_V1"

    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.sock = None
        self.active_devices = {}  # device_id -> {last_ip, last_seen, last_latency}
        self._lock = threading.Lock()

    def run(self):
        log.info(f"Starting Signaling Service on UDP {self.PORT}")
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(("", self.PORT))
            self.sock.settimeout(1.0)

            while self.running:
                try:
                    data, addr = self.sock.recvfrom(1024)
                    if data.startswith(self.MAGIC):
                        self._handle_signal(data, addr)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        log.error(f"Signaling error: {e}")
                    time.sleep(1)
        except Exception as e:
            log.error(f"Signaling service failed to start: {e}")

    def _handle_signal(self, data, addr):
        try:
            # Format: MAGIC | device_id | status_json
            parts = data[len(self.MAGIC) :].split(b"|", 1)
            if len(parts) < 2:
                return

            device_id = parts[0].decode("utf-8")
            status_json = json.loads(parts[1].decode("utf-8"))

            with self._lock:
                is_new = device_id not in self.active_devices
                self.active_devices[device_id] = {
                    "ip": addr[0],
                    "last_seen": time.time(),
                    "status": status_json,
                }

            # Send immediate ACK to help Android app measure RTT and confirm NAT mapping
            ack_data = (
                self.MAGIC
                + b"ACK|"
                + json.dumps({"timestamp": time.time()}).encode("utf-8")
            )
            self.sock.sendto(ack_data, addr)

            if is_new:
                log.info(
                    f"Instant discovery: New device {device_id} signaled from {addr[0]}"
                )
        except Exception as e:
            log.debug(f"Failed to handle signal: {e}")

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()


class DataReceiver(threading.Thread):
    """TCP server that receives project and notes data from Android app.

    Security Features:
    - Rate limiting: Max 5 connection attempts per IP per minute
    - Known device tracking: Only known devices can sync without approval
    - SSH key approval: New device keys require user confirmation
    """

    # Rate limiting settings
    RATE_LIMIT_WINDOW = 60  # seconds
    RATE_LIMIT_MAX = 100  # max attempts per window (increased for retry flows)

    # SECURITY: Limit concurrent connections to prevent resource exhaustion
    MAX_WORKERS = 10

    def __init__(
        self,
        on_data_received,
        on_device_connected,
        on_notes_received=None,
        on_sessions_received=None,
        on_cards_received=None,
        on_collections_received=None,
        on_key_approval_needed=None,
    ):
        super().__init__(daemon=True)
        self.on_data_received = on_data_received
        self.on_device_connected = on_device_connected
        self.on_notes_received = on_notes_received
        self.on_sessions_received = on_sessions_received
        self.on_cards_received = on_cards_received
        self.on_collections_received = on_collections_received
        self.on_key_approval_needed = (
            on_key_approval_needed  # Callback for key approval UI
        )
        self.running = True
        self.sock = None
        self.bind_failed = False  # Track if port binding failed
        self.bind_error = None  # Store error message
        self.connected_devices = {}  # device_id -> {name, ip, last_seen}
        self.ip_to_device_id = {}  # ip -> device_id
        self._devices_lock = threading.Lock()
        self._storage_lock = threading.Lock()
        # SECURITY: Use thread pool instead of unbounded daemon threads
        self._executor = ThreadPoolExecutor(
            max_workers=self.MAX_WORKERS, thread_name_prefix="DataReceiver"
        )

        # Security: Rate limiting
        self._rate_limit_tracker = {}  # ip -> [timestamps]
        self._rate_limit_lock = threading.Lock()

        # Security: Pending key approvals
        self._pending_keys = {}  # device_id -> {public_key, device_name, ip, timestamp}
        self._approved_devices = set()  # Set of approved device_ids
        self._load_approved_devices()

        # Transcript watching for bidirectional session sync
        self._transcript_watcher_thread = None
        self._transcript_path = None
        self._transcript_last_pos = 0
        self._transcript_stop_event = threading.Event()

    def _load_approved_devices(self):
        """Load list of approved device IDs from disk."""
        try:
            approved_file = os.path.join(
                os.environ.get("USERPROFILE", os.path.expanduser("~")),
                ".shadowai",
                "approved_devices.json",
            )
            if os.path.exists(approved_file):
                with open(approved_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._approved_devices = set(data.get("approved", []))
                    log.info(f"Loaded {len(self._approved_devices)} approved devices")
        except Exception as e:
            log.warning(f"Could not load approved devices: {e}")
            self._approved_devices = set()

    def _save_approved_devices(self):
        """Save list of approved device IDs to disk."""
        try:
            approved_file = os.path.join(
                os.environ.get("USERPROFILE", os.path.expanduser("~")),
                ".shadowai",
                "approved_devices.json",
            )
            os.makedirs(os.path.dirname(approved_file), exist_ok=True)
            with open(approved_file, "w", encoding="utf-8") as f:
                json.dump({"approved": list(self._approved_devices)}, f, indent=2)
        except Exception as e:
            log.warning(f"Could not save approved devices: {e}")

    def _start_transcript_watcher(self, transcript_path):
        """Start watching a transcript file for new content."""
        if not transcript_path or not os.path.exists(transcript_path):
            log.warning(f"Transcript path invalid or doesn't exist: {transcript_path}")
            return

        # Stop any existing watcher
        self._stop_transcript_watcher()

        self._transcript_path = transcript_path
        self._transcript_last_pos = (
            os.path.getsize(transcript_path) if os.path.exists(transcript_path) else 0
        )
        self._transcript_stop_event.clear()

        self._transcript_watcher_thread = threading.Thread(
            target=self._watch_transcript, daemon=True
        )
        self._transcript_watcher_thread.start()
        log.info(f"Started transcript watcher for: {transcript_path}")

    def _stop_transcript_watcher(self):
        """Stop the transcript watcher."""
        self._transcript_stop_event.set()
        if (
            self._transcript_watcher_thread
            and self._transcript_watcher_thread.is_alive()
        ):
            self._transcript_watcher_thread.join(timeout=2)
        self._transcript_watcher_thread = None
        self._transcript_path = None
        log.info("Stopped transcript watcher")

    def _watch_transcript(self):
        """Watch transcript file for new content and relay to devices."""
        while not self._transcript_stop_event.is_set():
            try:
                if not self._transcript_path or not os.path.exists(
                    self._transcript_path
                ):
                    time.sleep(1)
                    continue

                current_size = os.path.getsize(self._transcript_path)
                if current_size > self._transcript_last_pos:
                    # New content available
                    with open(
                        self._transcript_path, "r", encoding="utf-8", errors="ignore"
                    ) as f:
                        f.seek(self._transcript_last_pos)
                        new_content = f.read()
                        self._transcript_last_pos = f.tell()

                    if new_content.strip():
                        self._parse_and_relay_transcript(new_content)

                time.sleep(0.5)  # Poll every 500ms

            except Exception as e:
                log.error(f"Transcript watcher error: {e}")
                time.sleep(1)

    def _parse_and_relay_transcript(self, content):
        """Parse transcript content and relay messages to connected devices."""
        try:
            # Claude Code transcripts are typically JSONL format
            # Each line is a JSON object with message data
            lines = content.strip().split("\n")

            for line in lines:
                if not line.strip():
                    continue

                try:
                    msg_data = json.loads(line)

                    # Extract message details
                    msg_type = msg_data.get("type", "")
                    role = msg_data.get("role", msg_data.get("sender", ""))
                    message_content = msg_data.get(
                        "content", msg_data.get("message", "")
                    )

                    # Handle different message formats
                    if isinstance(message_content, list):
                        # Content blocks format
                        text_parts = []
                        for block in message_content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    text_parts.append(block.get("text", ""))
                                elif block.get("type") == "tool_use":
                                    text_parts.append(
                                        f"[Tool: {block.get('name', 'unknown')}]"
                                    )
                            elif isinstance(block, str):
                                text_parts.append(block)
                        message_content = "\n".join(text_parts)

                    if not message_content:
                        continue

                    # Create relay message
                    relay_msg = {
                        "type": "session_message",
                        "id": f"msg_{int(time.time() * 1000)}",
                        "sessionId": self._pc_session.get("sessionId")
                        if self._pc_session
                        else None,
                        "timestamp": int(time.time() * 1000),
                        "payload": {
                            "role": role or "assistant",
                            "content": message_content[:5000],  # Limit size
                            "hostname": socket.gethostname(),
                        },
                    }

                    # Relay to all devices
                    self._relay_to_all_devices(relay_msg)
                    log.debug(
                        f"Relayed session message: {role} - {message_content[:50]}..."
                    )

                    # Detect file creation for Phase 4.3 requirement (File Notifications)
                    self._detect_file_creation(
                        message_content, sessionId=relay_msg.get("sessionId")
                    )

                except json.JSONDecodeError:
                    # Not JSON, might be plain text output
                    if len(line.strip()) > 10:
                        content = line.strip()
                        relay_msg = {
                            "type": "session_message",
                            "id": f"msg_{int(time.time() * 1000)}",
                            "sessionId": self._pc_session.get("sessionId")
                            if self._pc_session
                            else None,
                            "timestamp": int(time.time() * 1000),
                            "payload": {
                                "role": "assistant",
                                "content": content[:5000],
                                "hostname": socket.gethostname(),
                            },
                        }
                        self._relay_to_all_devices(relay_msg)
                        self._detect_file_creation(
                            content, sessionId=relay_msg.get("sessionId")
                        )

        except Exception as e:
            log.error(f"Failed to parse transcript content: {e}")

    def _detect_file_creation(self, content, sessionId=None):
        """Scan content for file creation and notify devices."""
        import re

        # Look for common file creation patterns
        # e.g. "Created file index.html", "Writing to styles.css", "Saved as image.png"
        file_patterns = [
            r"(?:Created|Saved|Writing to|Generated)\s+file\s+[`'\"\(]?([^\s`'\"\)\[\]]+\.[a-z0-9]{2,10})",
            r"(?:Created|Saved|Writing to|Generated)\s+[`'\"\(]?([^\s`'\"\)\[\]]+\.(?:html|png|jpg|pdf|md|txt|py|js|json))",
            r"\[Tool:\s+(?:Write|Edit)\]\s+[`'\"\(]?([^\s`'\"\)\[\]]+\.[a-z0-9]{2,10})",
        ]

        found_files = []
        for pattern in file_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found_files.extend(matches)

        found_files = list(set(found_files))
        if not found_files:
            return

        # Simple file list for the notification message
        file_list_str = ", ".join(found_files)
        if len(found_files) > 3:
            file_list_str = f"{len(found_files)} files (including {found_files[0]})"

        # Send aggregated notification message
        notif_msg = {
            "type": "notification",
            "id": f"notif_{int(time.time() * 1000)}",
            "sessionId": sessionId,
            "timestamp": int(time.time() * 1000),
            "payload": {
                "message": f"Assets ready: {file_list_str}",
                "notificationType": "file_ready",
                "summary": "Project Assets Generated",
                "files": found_files,
                "deepLink": f"shadow://project/{sessionId}"
                if sessionId
                else "shadow://projects",
                "hostname": socket.gethostname(),
            },
        }
        self._relay_to_all_devices(notif_msg)
        log.info(f"Sent aggregated notification for {len(found_files)} files")

    def _relay_to_all_devices(self, message):
        """Relay a message to all connected Android devices."""
        with self._conns_lock:
            for device_id, conn in list(self._device_conns.items()):
                try:
                    if conn:
                        self._send_to_conn(conn, message)
                except Exception as e:
                    log.debug(f"Failed to relay to device {device_id}: {e}")
                    # Remove broken connection
                    self._device_conns.pop(device_id, None)

    def approve_device(self, device_id):
        """Approve a device for SSH key installation."""
        self._approved_devices.add(device_id)
        self._save_approved_devices()
        log.info(f"Device approved: {device_id}")

        # If there's a pending key for this device, install it now
        if device_id in self._pending_keys:
            pending = self._pending_keys.pop(device_id)
            result = self._install_ssh_key(
                pending["public_key"], pending["device_name"]
            )
            log.info(f"Installed pending key for {device_id}: {result}")
            return result
        return {"success": True, "message": "Device approved"}

    def reject_device(self, device_id):
        """Reject a pending device key."""
        if device_id in self._pending_keys:
            del self._pending_keys[device_id]
            log.info(f"Device key rejected: {device_id}")

    def get_pending_keys(self):
        """Get list of pending key approvals."""
        return dict(self._pending_keys)

    def revoke_all_keys(self):
        """Remove all Shadow-added keys from authorized_keys files."""
        try:
            # List of authorized_keys files to check
            auth_keys_files = []
            home_dir = os.path.expanduser("~")
            ssh_dir = os.path.join(home_dir, ".ssh")
            auth_keys_files.append(os.path.join(ssh_dir, "authorized_keys"))
            if platform.system() == "Windows":
                auth_keys_files.append(
                    r"C:\ProgramData\ssh\administrators_authorized_keys"
                )

            removed_count = 0
            for auth_keys_file in auth_keys_files:
                if not os.path.exists(auth_keys_file):
                    continue

                try:
                    with open(auth_keys_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    new_lines = []
                    skip_next = False
                    for line in lines:
                        if skip_next:
                            skip_next = False
                            removed_count += 1
                            continue

                        if line.startswith("# Shadow device:"):
                            skip_next = True
                            continue

                        new_lines.append(line)

                    if len(new_lines) < len(lines):
                        with open(auth_keys_file, "w", encoding="utf-8") as f:
                            f.writelines(new_lines)
                        log.info(f"Cleaned Shadow keys from {auth_keys_file}")

                except Exception as e:
                    log.warning(f"Failed to clean {auth_keys_file}: {e}")

            # Also clear approved devices and pending keys
            with self._devices_lock:
                self._approved_devices.clear()
                self._save_approved_devices()
                self._pending_keys.clear()
                self.connected_devices.clear()

            return {
                "success": True,
                "message": f"Revoked {removed_count} keys and cleared pairing data.",
            }
        except Exception as e:
            return {"success": False, "message": f"Revocation failed: {str(e)}"}

    def _check_rate_limit(self, ip):
        """Check if IP is rate limited. Returns True if request is allowed."""
        # Skip rate limiting for localhost (internal web dashboard connections)
        if ip in ("127.0.0.1", "localhost", "::1"):
            return True

        # Skip rate limiting for approved devices (check if any approved device has this IP)
        with self._devices_lock:
            for device_id in self._approved_devices:
                device_info = self.connected_devices.get(device_id, {})
                if device_info.get("ip") == ip:
                    return True  # Approved device, no rate limit

        now = time.time()
        with self._rate_limit_lock:
            if ip not in self._rate_limit_tracker:
                self._rate_limit_tracker[ip] = []

            # Remove old timestamps outside the window
            self._rate_limit_tracker[ip] = [
                ts
                for ts in self._rate_limit_tracker[ip]
                if now - ts < self.RATE_LIMIT_WINDOW
            ]

            # Record this attempt BEFORE checking (fixes off-by-one bug)
            self._rate_limit_tracker[ip].append(now)

            # Check if over limit (now correctly enforces RATE_LIMIT_MAX)
            if len(self._rate_limit_tracker[ip]) > self.RATE_LIMIT_MAX:
                log.warning(f"Rate limit exceeded for {ip}")
                return False

            return True

    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind(("", DATA_PORT))
            self.sock.listen(5)
            self.sock.settimeout(1.0)
            log.info(f"[OK] DataReceiver listening on port {DATA_PORT}")
        except OSError as e:
            log.error(f"[ERR] FATAL: DataReceiver cannot bind to port {DATA_PORT}: {e}")
            log.error(f"Another application may be using port {DATA_PORT}")
            log.error("Device sync will NOT work until this is resolved")
            # Set flag for GUI to detect failure
            self.bind_failed = True
            self.bind_error = str(e)
            return  # Exit thread - cannot continue without port
        except Exception as e:
            log.error(f"[ERR] FATAL: DataReceiver unexpected error during bind: {e}")
            self.bind_failed = True
            self.bind_error = str(e)
            return

        # Port binding successful - mark as ready
        self.bind_failed = False
        self.bind_error = None

        try:
            while self.running:
                try:
                    conn, addr = self.sock.accept()
                    # SECURITY: Use thread pool with bounded size instead of unbounded threads
                    self._executor.submit(self._handle_client, conn, addr)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:  # Only log if not shutting down
                        log.warning(f"DataReceiver accept error: {e}")
        finally:
            # Gracefully shutdown thread pool with timeout to prevent hanging
            try:
                self._executor.shutdown(wait=True, timeout=5.0)
            except Exception:
                self._executor.shutdown(wait=False, cancel_futures=True)
            if self.sock:
                self.sock.close()

    def _handle_client(self, conn, addr):
        """Handle incoming data from Android app."""
        try:
            ip = addr[0]
            # Skip logging to reduce log spam - connections are now silent by default

            # Security: Rate limiting
            if not self._check_rate_limit(ip):
                log.warning(f"Rate limited connection from {ip}")
                self._send_response(
                    conn,
                    {"success": False, "message": "Rate limited. Try again later."},
                )
                return

            conn.settimeout(10)

            # Read length-prefixed message (4 bytes big-endian length + data)
            # Must read exactly 4 bytes for length prefix
            length_bytes = b""
            while len(length_bytes) < 4:
                chunk = conn.recv(4 - len(length_bytes))
                if not chunk:
                    # Silently close health check connections (localhost) to reduce log spam
                    # Only log unexpected connection failures from non-localhost
                    if len(length_bytes) == 0 and ip == "127.0.0.1":
                        pass  # Silently skip health check connections
                    else:
                        log.debug(
                            f"Connection closed while reading length prefix from {addr}"
                        )
                    return
                length_bytes += chunk

            # Skip verbose debug logging to reduce log spam
            pass

            msg_length = int.from_bytes(length_bytes, "big")
            if msg_length > 10 * 1024 * 1024:  # 10MB limit for sessions payloads
                # Only log error for non-localhost to reduce spam
                if ip != "127.0.0.1":
                    log.error(f"Message too large from {addr}: {msg_length} bytes")
                return

            data = b""
            while len(data) < msg_length:
                # Increase buffer size for better performance
                chunk = conn.recv(min(65536, msg_length - len(data)))  # 64KB buffer
                if not chunk:
                    # Skip logging to reduce spam
                    break
                data += chunk

            # Skip verbose debug logging to reduce log spam
            pass
            if len(data) < msg_length:
                # Only log error for non-localhost to reduce spam
                if ip != "127.0.0.1":
                    log.error(
                        f"From {addr}: Incomplete message - got {len(data)}/{msg_length} bytes"
                    )
            else:
                # Skip verbose debug logging to reduce log spam
                pass

            if data:
                try:
                    payload = json.loads(data.decode("utf-8"))
                    action = payload.get("action", "")
                    ip = addr[0]
                    raw_device_id = payload.get("device_id", None)
                    device_id = raw_device_id or self.ip_to_device_id.get(ip) or ip
                    try:
                        client_schema = int(payload.get("schema_version", SYNC_SCHEMA_MIN_VERSION))
                    except (TypeError, ValueError):
                        client_schema = SYNC_SCHEMA_MIN_VERSION

                    def send_sync_response(response_payload: dict):
                        response_payload["schema_version"] = SYNC_SCHEMA_VERSION
                        response_payload["schema_min_version"] = SYNC_SCHEMA_MIN_VERSION
                        compatible = SYNC_SCHEMA_MIN_VERSION <= client_schema <= SYNC_SCHEMA_VERSION
                        response_payload["schema_compatible"] = compatible
                        if not compatible:
                            if client_schema < SYNC_SCHEMA_MIN_VERSION:
                                response_payload["schema_warning"] = "Client sync schema too old; upgrade the app."
                            else:
                                response_payload["schema_warning"] = "Client sync schema is newer; some fields may be ignored."
                        self._send_response(conn, response_payload)

                    with self._devices_lock:
                        if raw_device_id:
                            self.ip_to_device_id[ip] = raw_device_id

                        existing = self.connected_devices.get(device_id)
                        device_name = payload.get("device_name", None)
                        if device_name is None:
                            device_name = existing.get("name") if existing else ip

                        # Merge legacy entry keyed by IP into stable device_id if needed.
                        if (
                            raw_device_id
                            and raw_device_id != ip
                            and ip in self.connected_devices
                            and device_id != ip
                        ):
                            legacy = self.connected_devices.pop(ip, {})
                            if not existing:
                                existing = legacy

                        self.connected_devices[device_id] = {
                            "name": device_name,
                            "ip": ip,
                            "last_seen": time.time(),
                        }

                    # If we learned a stable device_id after having only an IP-based id,
                    # migrate any previously saved projects to keep everything under one device.
                    if raw_device_id and raw_device_id != ip and device_id != ip:
                        self._migrate_saved_device(
                            old_device_id=ip,
                            new_device_id=device_id,
                            device_name=device_name,
                            ip=ip,
                        )

                    if self.on_device_connected:
                        self.on_device_connected(device_id, device_name, ip)

                    # Handle different actions
                    # Reduce log flooding: Only log non-ping actions at info level
                    if action != "ping":
                        log.debug(f"Action: {action}, Device: {device_name}")
                    else:
                        log.debug(f"Action: {action}, Device: {device_name}")
                    if action == "ping":
                        self._send_response(conn, {"success": True, "message": "PONG"})
                    elif action == "install_key":
                        # SSH key installation request - SECURITY: Requires approval for new devices AND new keys
                        public_key = payload.get("public_key", "")

                        # Check if key is already installed (regardless of device approval)
                        if self._is_key_installed(public_key):
                            log.info(f"SSH key already installed for: {device_name}")
                            response = {
                                "success": True,
                                "message": "Key already installed",
                            }
                            self._send_response(conn, response)
                        elif TRUST_ALL:
                            log.info(
                                f"TRUST_ALL enabled: auto-approving key from {device_name}"
                            )
                            result = self._install_ssh_key(public_key, device_name)
                            self._send_response(conn, result)
                        else:
                            # Queue for approval (even if device is known, a new key requires approval)
                            log.info(
                                f"Queueing SSH key for approval: {device_name} ({device_id})"
                            )
                            self._pending_keys[device_id] = {
                                "public_key": public_key,
                                "device_name": device_name,
                                "ip": ip,
                                "timestamp": time.time(),
                            }
                            # Notify UI for approval
                            if self.on_key_approval_needed:
                                self.on_key_approval_needed(device_id, device_name, ip)
                            response = {
                                "success": False,
                                "pending": True,
                                "message": "SSH key queued for approval. Please approve on the PC.",
                            }
                            self._send_response(conn, response)
                        log.info(f"Response sent to {addr}")
                    elif "projects" in payload:
                        # Handle projects data
                        ip_candidates = payload.get("ip_candidates")
                        note_content_port = payload.get("note_content_port")
                        self._save_projects(
                            device_id,
                            device_name,
                            ip,
                            payload["projects"],
                            ip_candidates,
                            note_content_port,
                        )
                        if self.on_data_received:
                            self.on_data_received(device_id, payload["projects"])

                        # Bi-directional sync: Include pending items from web
                        response = {"success": True, "message": "Projects synced"}
                        if SYNC_SERVICE_AVAILABLE:
                            pending = get_pending_sync_items(device_id)
                            if pending.get("projects"):
                                response["sync_to_device"] = {
                                    "projects": pending["projects"]
                                }
                                log.info(
                                    f"Including {len(pending['projects'])} pending projects for sync to device"
                                )
                        send_sync_response(response)
                    elif "agents" in payload:
                        # Handle agents data from Android device
                        self._save_agents(device_id, device_name, ip, payload["agents"])

                        # Sync to agent orchestrator for web API access
                        try:
                            from web.services.agent_orchestrator import get_orchestrator
                            get_orchestrator().sync_agents(device_id, payload["agents"])
                        except Exception as e:
                            log.warning(f"Agent orchestrator sync failed: {e}")

                        # Broadcast to WebSocket clients for real-time web dashboard updates
                        try:
                            from web.routes.websocket import broadcast_agents_updated
                            broadcast_agents_updated(device_id)
                        except Exception as e:
                            log.debug(f"WebSocket broadcast skipped: {e}")

                        # Build response with bridge-side agents for bidirectional sync
                        response = {"success": True, "message": "Agents synced"}
                        try:
                            from web.services.agent_orchestrator import get_all_agents
                            bridge_agents = get_all_agents()
                            if bridge_agents:
                                response["sync_to_device"] = {"agents": bridge_agents}
                        except Exception as e:
                            log.warning(f"Failed to include bridge agents in response: {e}")
                        send_sync_response(response)
                    elif "tasks" in payload:
                        # Handle tasks data
                        self._save_tasks(device_id, device_name, ip, payload["tasks"])
                        send_sync_response({"success": True, "message": "Tasks synced"})
                    elif "automations" in payload:
                        # Handle automations data
                        self._save_automations(
                            device_id, device_name, ip, payload["automations"]
                        )
                        # Bi-directional sync: Include pending automations from web
                        response = {"success": True, "message": "Automations synced"}
                        if SYNC_SERVICE_AVAILABLE:
                            pending = get_pending_sync_items(device_id)
                            if pending.get("automations"):
                                response["sync_to_device"] = {
                                    "automations": pending["automations"]
                                }
                                log.info(
                                    f"Including {len(pending['automations'])} pending automations for sync to device"
                                )
                        send_sync_response(response)
                    elif "notes" in payload:
                        # Handle notes data (titles only, content fetched on-demand)
                        ip_candidates = payload.get("ip_candidates")
                        note_content_port = payload.get("note_content_port")
                        self._save_notes(
                            device_id,
                            device_name,
                            ip,
                            payload["notes"],
                            ip_candidates,
                            note_content_port,
                        )
                        if self.on_notes_received:
                            self.on_notes_received(device_id, payload["notes"])

                        # Bi-directional sync: Include pending items from web
                        response = {"success": True, "message": "Notes synced"}
                        if SYNC_SERVICE_AVAILABLE:
                            pending = get_pending_sync_items(device_id)
                            if pending.get("notes"):
                                response["sync_to_device"] = {"notes": pending["notes"]}
                                log.info(
                                    f"Including {len(pending['notes'])} pending notes for sync to device"
                                )
                        send_sync_response(response)
                    elif action == "sync_sessions" or "sessions" in payload:
                        sessions_payload = payload.get("sessions", [])
                        if SYNC_SERVICE_AVAILABLE and isinstance(
                            sessions_payload, list
                        ):
                            save_sessions_from_device(
                                device_id, device_name, sessions_payload
                            )
                        if self.on_sessions_received:
                            self.on_sessions_received(device_id, sessions_payload)

                        response = {"success": True, "message": "Sessions synced"}
                        if SYNC_SERVICE_AVAILABLE:
                            pending = get_pending_sync_items(device_id)
                            if pending.get("sessions"):
                                response["sync_to_device"] = {
                                    "sessions": pending["sessions"]
                                }
                                log.info(
                                    f"Including {len(pending['sessions'])} pending sessions for sync to device"
                                )
                        send_sync_response(response)
                    elif action == "sync_cards" or "cards" in payload:
                        cards_payload = payload.get("cards", [])
                        if SYNC_SERVICE_AVAILABLE and isinstance(cards_payload, list):
                            save_cards_from_device(device_id, cards_payload)
                        if self.on_cards_received:
                            self.on_cards_received(device_id, cards_payload)

                        response = {"success": True, "message": "Cards synced"}
                        if SYNC_SERVICE_AVAILABLE:
                            pending = get_pending_sync_items(device_id)
                            if pending.get("cards"):
                                response["sync_to_device"] = {"cards": pending["cards"]}
                                log.info(
                                    f"Including {len(pending['cards'])} pending cards for sync to device"
                                )
                        send_sync_response(response)
                    elif action == "sync_collections" or "collections" in payload:
                        collections_payload = payload.get("collections", [])
                        if SYNC_SERVICE_AVAILABLE and isinstance(
                            collections_payload, list
                        ):
                            save_collections_from_device(device_id, collections_payload)
                        if self.on_collections_received:
                            self.on_collections_received(device_id, collections_payload)

                        response = {"success": True, "message": "Collections synced"}
                        if SYNC_SERVICE_AVAILABLE:
                            pending = get_pending_sync_items(device_id)
                            if pending.get("collections"):
                                response["sync_to_device"] = {
                                    "collections": pending["collections"]
                                }
                                log.info(
                                    f"Including {len(pending['collections'])} pending collections for sync to device"
                                )
                        send_sync_response(response)
                    elif action == "sync_confirm":
                        # Android confirming it received and saved web-created items
                        if SYNC_SERVICE_AVAILABLE:
                            synced_projects = payload.get("synced_projects", [])
                            synced_notes = payload.get("synced_notes", [])
                            synced_automations = payload.get("synced_automations", [])
                            synced_sessions = payload.get("synced_sessions", [])
                            if synced_projects:
                                mark_items_synced(
                                    device_id, "projects", synced_projects
                                )
                                log.info(
                                    f"Marked {len(synced_projects)} projects as synced"
                                )
                            if synced_notes:
                                mark_items_synced(device_id, "notes", synced_notes)
                                log.info(f"Marked {len(synced_notes)} notes as synced")
                            if synced_automations:
                                mark_items_synced(
                                    device_id, "automations", synced_automations
                                )
                                log.info(
                                    f"Marked {len(synced_automations)} automations as synced"
                                )
                            if synced_sessions:
                                mark_items_synced(
                                    device_id, "sessions", synced_sessions
                                )
                                log.info(
                                    f"Marked {len(synced_sessions)} sessions as synced"
                                )
                            synced_cards = payload.get("synced_cards", [])
                            synced_collections = payload.get("synced_collections", [])
                            if synced_cards:
                                mark_items_synced(
                                    device_id, "cards", synced_cards
                                )
                                log.info(
                                    f"Marked {len(synced_cards)} cards as synced"
                                )
                            if synced_collections:
                                mark_items_synced(
                                    device_id, "collections", synced_collections
                                )
                                log.info(
                                    f"Marked {len(synced_collections)} collections as synced"
                                )
                        send_sync_response({"success": True, "message": "Sync confirmed"})

                    elif action == "sync_audits":
                        try:
                            from web.services.audit_service import get_audit_service
                            svc = get_audit_service()
                            audit_entries = payload.get("audit_entries", [])
                            audit_traces = payload.get("audit_traces", [])
                            # Index traces by entry ID for lookup
                            traces_by_entry = {}
                            for trace in audit_traces:
                                eid = trace.get("audit_entry_id", "")
                                if eid:
                                    traces_by_entry.setdefault(eid, []).append(trace)
                            count = 0
                            for entry in audit_entries:
                                entry_id = entry.get("id", "")
                                # Embed traces into details so get_traces() can reconstruct them
                                details = entry.get("details") or {}
                                if isinstance(details, str):
                                    try:
                                        details = json.loads(details)
                                    except (json.JSONDecodeError, TypeError):
                                        details = {}
                                entry_traces = traces_by_entry.get(entry_id, [])
                                if entry_traces:
                                    details["traces"] = entry_traces
                                svc.log_audit(
                                    category=entry.get("category", "SYSTEM"),
                                    severity=entry.get("severity", "INFO"),
                                    summary=entry.get("summary", ""),
                                    action=entry.get("action", ""),
                                    resource=entry.get("related_entity_id", ""),
                                    device_id=device_id,
                                    details=details if details else None,
                                )
                                count += 1
                            log.info(
                                f"Synced {count} audit entries from {device_name} "
                                f"({len(audit_traces)} traces attached)"
                            )
                            send_sync_response(
                                {"success": True, "message": f"Synced {count} audit entries"}
                            )
                        except Exception as e:
                            log.error(f"sync_audits failed: {e}")
                            send_sync_response({"success": False, "error": str(e)})

                    elif action == "sync_settings":
                        try:
                            from web.services.data_service import save_device_settings
                            settings_payload = payload.get("settings", {})
                            save_device_settings(device_id, device_name, settings_payload)
                            log.info(
                                f"Synced settings from {device_name}: "
                                f"backend={settings_payload.get('active_backend')}, "
                                f"model={settings_payload.get('model')}"
                            )
                            send_sync_response({"success": True, "message": "Settings synced"})
                        except Exception as e:
                            log.error(f"sync_settings failed: {e}")
                            send_sync_response({"success": False, "error": str(e)})

                    elif action == "sync_pull":
                        # Phone requesting full data bundle (all pending items at once)
                        try:
                            from web.services.data_service import get_full_sync_bundle
                            bundle = get_full_sync_bundle(device_id)
                            log.info(
                                f"Serving full sync bundle to {device_name}"
                            )
                            send_sync_response({"success": True, "bundle": bundle})
                        except Exception as e:
                            log.error(f"sync_pull failed: {e}")
                            send_sync_response({"success": False, "error": str(e)})

                    # ========== Intelligence Layer: Agent Persistence ==========
                    elif action == "save_conversation":
                        try:
                            from web.services.state_store import get_state_store
                            store = get_state_store()
                            conv = payload.get("conversation", {})
                            messages = payload.get("messages", [])
                            conv["device_id"] = conv.get("device_id", device_id)
                            conv["device_name"] = conv.get("device_name", device_name)
                            store.save_conversation(conv)
                            if messages:
                                store.save_messages(conv.get("id", ""), messages)
                            # Auto-index in vector store
                            try:
                                from web.services.vector_store import get_vector_store_v2
                                vs = get_vector_store_v2()
                                if vs:
                                    title = conv.get("title", "")
                                    content = " ".join(m.get("content", "") for m in messages if m.get("content"))
                                    vs.index_document("conversation", conv.get("id", ""), title, content[:2000],
                                                      collection="conversations")
                            except Exception:
                                pass
                            self._send_response(conn, {"success": True, "message": "Conversation saved"})
                        except Exception as e:
                            log.error(f"save_conversation failed: {e}")
                            self._send_response(conn, {"success": False, "error": str(e)})

                    elif action == "get_conversations":
                        try:
                            from web.services.state_store import get_state_store
                            store = get_state_store()
                            result = store.get_conversations(
                                agent_id=payload.get("agent_id"),
                                project_id=payload.get("project_id"),
                                limit=payload.get("limit", 50),
                                offset=payload.get("offset", 0),
                            )
                            self._send_response(conn, {"success": True, "conversations": result})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    elif action == "get_conversation_messages":
                        try:
                            from web.services.state_store import get_state_store
                            store = get_state_store()
                            result = store.get_messages(
                                conversation_id=payload.get("conversation_id", ""),
                                limit=payload.get("limit", 100),
                                before_id=payload.get("before_id"),
                            )
                            self._send_response(conn, {"success": True, "messages": result})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    elif action == "sync_agent_memory":
                        try:
                            from web.services.state_store import get_state_store
                            store = get_state_store()
                            agent_id_mem = payload.get("agent_id", "")
                            memories = payload.get("memories", [])
                            result = store.sync_memories(agent_id_mem, memories)
                            self._send_response(conn, {"success": True, **result})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    elif action == "search_conversations":
                        try:
                            from web.services.state_store import get_state_store
                            store = get_state_store()
                            query = payload.get("query", "")
                            result = store.search_conversations(
                                query=query,
                                agent_id=payload.get("agent_id"),
                                limit=payload.get("limit", 20),
                            )
                            self._send_response(conn, {"success": True, "results": result})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    # ========== Intelligence Layer: Storage API ==========
                    elif action == "storage_list":
                        try:
                            from web.services.storage_service import get_storage_service
                            svc = get_storage_service()
                            result = svc.list_directory(
                                path=payload.get("path", ""),
                                recursive=payload.get("recursive", False),
                                filter_glob=payload.get("filter_glob"),
                                include_hidden=payload.get("include_hidden", False),
                            )
                            self._send_response(conn, {"success": True, "files": result})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    elif action == "storage_read":
                        try:
                            from web.services.storage_service import get_storage_service
                            svc = get_storage_service()
                            result = svc.read_file(
                                path=payload.get("path", ""),
                                encoding=payload.get("encoding", "utf-8"),
                                offset=payload.get("offset"),
                                limit=payload.get("limit"),
                            )
                            self._send_response(conn, {"success": True, **result})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    elif action == "storage_write":
                        try:
                            from web.services.storage_service import get_storage_service
                            svc = get_storage_service()
                            result = svc.write_file(
                                path=payload.get("path", ""),
                                content=payload.get("content", ""),
                                encoding=payload.get("encoding", "utf-8"),
                                create_dirs=payload.get("create_dirs", False),
                            )
                            self._send_response(conn, {"success": True, **result})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    elif action == "storage_delete":
                        try:
                            from web.services.storage_service import get_storage_service
                            svc = get_storage_service()
                            result = svc.delete_file(path=payload.get("path", ""))
                            self._send_response(conn, {"success": True, **result})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    elif action == "storage_search":
                        try:
                            from web.services.storage_service import get_storage_service
                            svc = get_storage_service()
                            result = svc.search_files(
                                query=payload.get("query", ""),
                                root=payload.get("root"),
                                name_pattern=payload.get("name_pattern"),
                                content_search=payload.get("content_search", False),
                                max_results=payload.get("max_results", 50),
                            )
                            self._send_response(conn, {"success": True, "results": result})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    elif action == "storage_stat":
                        try:
                            from web.services.storage_service import get_storage_service
                            svc = get_storage_service()
                            result = svc.get_file_stat(path=payload.get("path", ""))
                            self._send_response(conn, {"success": True, **result})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    elif action == "storage_roots":
                        try:
                            from web.services.storage_service import get_storage_service
                            svc = get_storage_service()
                            result = svc.get_allowed_roots()
                            self._send_response(conn, {"success": True, "roots": result})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    elif action == "storage_stream_read":
                        try:
                            from web.services.storage_service import get_storage_service
                            svc = get_storage_service()
                            result = svc.read_file_chunked(
                                path=payload.get("path", ""),
                                chunk_size=payload.get("chunk_size", 524288),
                                chunk_index=payload.get("chunk_index", 0),
                            )
                            self._send_response(conn, {"success": True, **result})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    elif action == "storage_stream_write":
                        try:
                            from web.services.storage_service import get_storage_service
                            svc = get_storage_service()
                            result = svc.write_file_chunked(
                                path=payload.get("path", ""),
                                chunk_data=payload.get("chunk_data", ""),
                                chunk_index=payload.get("chunk_index", 0),
                                total_chunks=payload.get("total_chunks", 1),
                                is_final=payload.get("is_final", False),
                            )
                            self._send_response(conn, {"success": True, **result})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    # ========== Intelligence Layer: Vector Store ==========
                    elif action == "vector_search":
                        try:
                            from web.services.vector_store import get_vector_store_v2
                            vs = get_vector_store_v2()
                            if not vs:
                                self._send_response(conn, {"success": False, "error": "Vector store not available"})
                            else:
                                results = vs.hybrid_search(
                                    query=payload.get("query", ""),
                                    collections=payload.get("collections"),
                                    limit=payload.get("limit", 10),
                                    min_score=payload.get("min_score", 0.3),
                                )
                                self._send_response(conn, {
                                    "success": True,
                                    "results": [{"id": r.id, "content": r.content[:500], "score": r.score,
                                                 "metadata": r.metadata, "source_type": r.source_type,
                                                 "title": r.title, "collection": r.collection} for r in results],
                                })
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    elif action == "vector_index_directory":
                        try:
                            from web.services.vector_store import get_vector_store_v2
                            vs = get_vector_store_v2()
                            if not vs:
                                self._send_response(conn, {"success": False, "error": "Vector store not available"})
                            else:
                                result = vs.index_directory(
                                    path=payload.get("path", ""),
                                    recursive=payload.get("recursive", True),
                                )
                                self._send_response(conn, {"success": True, **result})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    elif action == "vector_status":
                        try:
                            from web.services.vector_store import get_vector_store_v2
                            vs = get_vector_store_v2()
                            if not vs:
                                self._send_response(conn, {"success": False, "error": "Vector store not available"})
                            else:
                                stats = vs.get_collection_stats()
                                self._send_response(conn, {"success": True, "collections": stats})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    elif action == "vector_build_context":
                        try:
                            from web.services.vector_store import get_vector_store_v2
                            vs = get_vector_store_v2()
                            if not vs:
                                self._send_response(conn, {"success": False, "error": "Vector store not available"})
                            else:
                                context = vs.build_agent_context(
                                    query=payload.get("query", ""),
                                    agent_id=payload.get("agent_id"),
                                    project_id=payload.get("project_id"),
                                    max_tokens=payload.get("max_tokens", 4000),
                                )
                                self._send_response(conn, {"success": True, "context": context})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    # ========== Intelligence Layer: Ollama / LLM ==========
                    elif action == "ollama_status":
                        try:
                            from web.services.llm_gateway import get_intelligence_hub
                            hub = get_intelligence_hub()
                            status = hub.ollama_manager.detect()
                            gpu = hub.ollama_manager.get_gpu_info()
                            recommended = hub.ollama_manager.recommend_models()
                            self._send_response(conn, {
                                "success": True, **status,
                                "gpu": gpu, "recommended_models": recommended,
                            })
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    elif action == "ollama_pull":
                        try:
                            from web.services.llm_gateway import get_intelligence_hub
                            hub = get_intelligence_hub()
                            model_name = payload.get("model_name", "")
                            result = hub.ollama_manager.pull_model(model_name)
                            self._send_response(conn, {"success": True, **result})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    elif action == "ollama_chat":
                        try:
                            from web.services.llm_gateway import get_intelligence_hub
                            hub = get_intelligence_hub()
                            result = hub.ollama_chat(
                                messages=payload.get("messages", []),
                                model=payload.get("model"),
                                system_prompt=payload.get("system_prompt"),
                                use_rag=payload.get("use_rag", False),
                                rag_query=payload.get("rag_query"),
                            )
                            self._send_response(conn, {"success": True, **result})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    elif action == "llm_route":
                        try:
                            from web.services.llm_gateway import get_intelligence_hub
                            hub = get_intelligence_hub()
                            decision = hub.router.route(
                                prompt=payload.get("prompt", ""),
                                task_type=payload.get("task_type"),
                                privacy_level=payload.get("privacy_level"),
                                latency_requirement=payload.get("latency_requirement"),
                            )
                            self._send_response(conn, {"success": True, **decision})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    elif action == "llm_stats":
                        try:
                            from web.services.llm_gateway import get_intelligence_hub
                            hub = get_intelligence_hub()
                            stats = hub.cost_tracker.get_stats()
                            self._send_response(conn, {"success": True, **stats})
                        except Exception as e:
                            self._send_response(conn, {"success": False, "error": str(e)})

                    else:
                        self._send_response(conn, {"success": True, "message": "OK"})

                except json.JSONDecodeError:
                    self._send_response(
                        conn, {"success": False, "message": "Invalid JSON"}
                    )
        except Exception as e:
            try:
                self._send_response(conn, {"success": False, "message": str(e)})
            except (socket.error, BrokenPipeError, ConnectionResetError):
                log.debug(f"Could not send error response: connection closed")
        finally:
            try:
                conn.shutdown(socket.SHUT_WR)
                time.sleep(0.5)  # Allow time for data flush
            except (socket.error, OSError):
                pass
            conn.close()

    def _send_response(self, conn, response):
        """Send length-prefixed JSON response."""
        try:
            data = json.dumps(response).encode("utf-8")
            conn.sendall(len(data).to_bytes(4, "big"))
            conn.sendall(data)
        except (socket.error, BrokenPipeError, ConnectionResetError, OSError):
            log.debug("Failed to send response: connection closed")

    def _is_key_installed(self, public_key):
        """Check if the specific public key is already in any authorized_keys file."""
        if not public_key:
            return False

        try:
            key_parts = public_key.strip().split()
            if len(key_parts) < 2:
                return False
            # Fingerprint is "type data" (e.g., "ssh-rsa AAA...")
            key_fingerprint = f"{key_parts[0]} {key_parts[1]}"

            # List of files to check (same as _install_ssh_key)
            files_to_check = []

            # User's ~/.ssh/authorized_keys
            home_dir = os.path.expanduser("~")
            ssh_dir = os.path.join(home_dir, ".ssh")
            files_to_check.append(os.path.join(ssh_dir, "authorized_keys"))

            # On Windows, also check administrators_authorized_keys
            if platform.system() == "Windows":
                files_to_check.append(
                    r"C:\ProgramData\ssh\administrators_authorized_keys"
                )

            for fpath in files_to_check:
                if os.path.exists(fpath):
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            content = f.read()
                            if key_fingerprint in content:
                                return True
                    except Exception:
                        pass

            return False
        except Exception as e:
            log.error(f"Error checking if key installed: {e}")
            return False

    def _validate_ssh_key(self, public_key):
        """Validate SSH public key format and structure.

        Checks:
        - Key starts with valid type (ssh-rsa, ssh-ed25519, etc.)
        - Has at least 2 parts (type + base64 data)
        - Base64 data is valid
        """
        if not public_key:
            return False, "Empty key"

        # Valid SSH key types
        valid_types = (
            "ssh-rsa",
            "ssh-ed25519",
            "ssh-dss",
            "ecdsa-sha2-nistp256",
            "ecdsa-sha2-nistp384",
            "ecdsa-sha2-nistp521",
            "sk-ssh-ed25519@openssh.com",
            "sk-ecdsa-sha2-nistp256@openssh.com",
        )

        parts = public_key.strip().split()
        if len(parts) < 2:
            return False, "Key must have at least type and data parts"

        key_type = parts[0]
        key_data = parts[1]

        if key_type not in valid_types:
            return (
                False,
                f"Unknown key type: {key_type}. Expected one of: {', '.join(valid_types[:3])}...",
            )

        # Validate base64 encoding
        try:
            decoded = base64.b64decode(key_data)
            if len(decoded) < 32:  # Minimum reasonable key size
                return False, "Key data too short"
        except Exception as e:
            return False, f"Invalid base64 encoding: {e}"

        return True, "Valid"

    def _install_ssh_key(self, public_key, device_name):
        """Install SSH public key to authorized_keys (both user and admin on Windows)."""
        try:
            # Validate key format and structure
            is_valid, error_msg = self._validate_ssh_key(public_key)
            if not is_valid:
                log.warning(f"SSH key validation failed: {error_msg}")
                return {"success": False, "message": f"Invalid SSH key: {error_msg}"}

            # List of authorized_keys files to update
            auth_keys_files = []

            # User's ~/.ssh/authorized_keys
            home_dir = os.path.expanduser("~")
            ssh_dir = os.path.join(home_dir, ".ssh")
            user_auth_keys = os.path.join(ssh_dir, "authorized_keys")
            auth_keys_files.append(user_auth_keys)

            # On Windows, also update administrators_authorized_keys for admin users
            if platform.system() == "Windows":
                admin_auth_keys = r"C:\ProgramData\ssh\administrators_authorized_keys"
                if os.path.exists(r"C:\ProgramData\ssh"):
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
                        with open(auth_keys_file, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith("#"):
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
                    with open(auth_keys_file, "a", encoding="utf-8") as f:
                        # Add newline if file doesn't end with one
                        if (
                            os.path.exists(auth_keys_file)
                            and os.path.getsize(auth_keys_file) > 0
                        ):
                            with open(auth_keys_file, "rb") as rf:
                                rf.seek(-1, 2)
                                if rf.read(1) != b"\n":
                                    f.write("\n")
                        f.write(
                            f"{SSH_KEY_PREFIX} {device_name} - added {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        )
                        f.write(f"{public_key}\n")

                    log.info(f"Key installed to {auth_keys_file}")
                    installed_count += 1

                    # Set correct permissions (non-Windows)
                    if platform.system() != "Windows":
                        try:
                            os.chmod(auth_keys_file, 0o600)
                        except OSError as e:
                            log.warning(
                                f"Could not set permissions on {auth_keys_file}: {e}"
                            )

                except Exception as e:
                    log.warning(f"Failed to update {auth_keys_file}: {e}")

            if installed_count > 0:
                return {
                    "success": True,
                    "message": f"SSH key installed for {device_name} ({installed_count} files)",
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to install key to any authorized_keys file",
                }

        except Exception as e:
            return {"success": False, "message": f"Failed to install key: {str(e)}"}

    def _save_projects(
        self,
        device_id,
        device_name,
        ip,
        projects,
        ip_candidates=None,
        note_content_port=None,
    ):
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
                name = project.get("name") or project.get("title") or "Unnamed"
                path = (
                    project.get("path")
                    or project.get("workingDirectory")
                    or project.get("working_directory")
                    or project.get("dir")
                    or ""
                )
                cleaned.append(
                    {
                        "id": project.get("id")
                        or project.get("projectId")
                        or project.get("project_id"),
                        "name": str(name),
                        "path": str(path) if path is not None else "",
                        "description": project.get("description", ""),
                        "updated_at": project.get("updated_at", 0),
                    }
                )

            with self._storage_lock:
                state = load_projects_state()
                devices = state.get("devices", {})

                # Extract base fingerprint for cleanup
                base_fingerprint = (
                    device_id.split(":com.")[0] if ":com." in device_id else None
                )

                # Clean up stale device entries with same base fingerprint
                if base_fingerprint:
                    stale_ids = [
                        did
                        for did in devices.keys()
                        if did != device_id
                        and (
                            did == base_fingerprint  # Old format (no package suffix)
                            or did.startswith(
                                base_fingerprint + ":com."
                            )  # Different package
                        )
                    ]
                    for stale_id in stale_ids:
                        log.info(
                            f"Removing stale device entry from projects: {stale_id}"
                        )
                        del devices[stale_id]

                device = devices.get(device_id, {})
                device.update(
                    {
                        "id": device_id,
                        "name": device_name,
                        "ip": ip,
                        "last_seen": time.time(),
                        "projects": cleaned,
                    }
                )
                if (
                    isinstance(note_content_port, int)
                    and 1 <= note_content_port <= 65535
                ):
                    device["note_content_port"] = note_content_port
                merged_candidates = []
                existing_candidates = device.get("ip_candidates", [])
                if isinstance(ip_candidates, list):
                    for candidate in ip_candidates:
                        if isinstance(candidate, str) and candidate:
                            merged_candidates.append(candidate)
                if isinstance(existing_candidates, list):
                    for candidate in existing_candidates:
                        if isinstance(candidate, str) and candidate:
                            merged_candidates.append(candidate)
                merged_candidates = [
                    candidate
                    for candidate in merged_candidates
                    if candidate and candidate != ip
                ]
                deduped = []
                for candidate in merged_candidates:
                    if candidate not in deduped:
                        deduped.append(candidate)
                if deduped:
                    device["ip_candidates"] = deduped
                else:
                    device.pop("ip_candidates", None)
                devices[device_id] = device

                save_projects_state({"devices": devices})
                log.info(
                    f"Saved {len(cleaned)} projects for {device_name} ({device_id})"
                )

                # Broadcast update to web dashboard
                try:
                    from web.routes.websocket import broadcast_projects_updated

                    broadcast_projects_updated(device_id)
                except Exception:
                    pass  # Web server may not be running
        except Exception:
            log.exception("Failed to save projects")

    def _save_notes(
        self,
        device_id,
        device_name,
        ip,
        notes,
        ip_candidates=None,
        note_content_port=None,
    ):
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
                    "id": note.get("id", ""),
                    "title": note.get("title", "Untitled"),
                    "updatedAt": note.get("updatedAt", 0),
                }
                # Include content if provided (for offline access)
                if note.get("content"):
                    note_data["content"] = note.get("content")
                cleaned.append(note_data)

            with self._storage_lock:
                state = load_notes_state()
                devices = state.get("devices", {})

                # Extract base fingerprint for cleanup
                base_fingerprint = (
                    device_id.split(":com.")[0] if ":com." in device_id else None
                )

                # Clean up stale device entries with same base fingerprint
                if base_fingerprint:
                    stale_ids = [
                        did
                        for did in devices.keys()
                        if did != device_id
                        and (
                            did == base_fingerprint  # Old format (no package suffix)
                            or did.startswith(
                                base_fingerprint + ":com."
                            )  # Different package
                        )
                    ]
                    for stale_id in stale_ids:
                        log.info(f"Removing stale device entry: {stale_id}")
                        del devices[stale_id]

                device = devices.get(device_id, {})
                device.update(
                    {
                        "id": device_id,
                        "name": device_name,
                        "ip": ip,
                        "last_seen": time.time(),
                        "notes": cleaned,
                    }
                )
                if (
                    isinstance(note_content_port, int)
                    and 1 <= note_content_port <= 65535
                ):
                    device["note_content_port"] = note_content_port
                merged_candidates = []
                existing_candidates = device.get("ip_candidates", [])
                if isinstance(ip_candidates, list):
                    for candidate in ip_candidates:
                        if isinstance(candidate, str) and candidate:
                            merged_candidates.append(candidate)
                if isinstance(existing_candidates, list):
                    for candidate in existing_candidates:
                        if isinstance(candidate, str) and candidate:
                            merged_candidates.append(candidate)
                merged_candidates = [
                    candidate
                    for candidate in merged_candidates
                    if candidate and candidate != ip
                ]
                deduped = []
                for candidate in merged_candidates:
                    if candidate not in deduped:
                        deduped.append(candidate)
                if deduped:
                    device["ip_candidates"] = deduped
                else:
                    device.pop("ip_candidates", None)
                devices[device_id] = device

                save_notes_state({"devices": devices})
                log.info(f"Saved {len(cleaned)} notes for {device_name} ({device_id})")

                # Broadcast update to web dashboard
                try:
                    from web.routes.websocket import broadcast_notes_updated

                    broadcast_notes_updated(device_id)
                except Exception:
                    pass  # Web server may not be running
        except Exception:
            log.exception("Failed to save notes")

    def _save_agents(self, device_id, device_name, ip, agents):
        """Save per-device agents to local file."""
        try:
            if not isinstance(agents, list):
                return
            with self._storage_lock:
                state = load_agents_state()
                devices = state.get("devices", {})
                device = devices.get(device_id, {})
                device.update(
                    {
                        "id": device_id,
                        "name": device_name,
                        "ip": ip,
                        "last_seen": time.time(),
                        "agents": agents,
                    }
                )
                devices[device_id] = device
                save_agents_state({"devices": devices})
                log.info(f"Saved {len(agents)} agents for {device_name}")
        except Exception:
            log.exception("Failed to save agents")

    def _save_tasks(self, device_id, device_name, ip, tasks):
        """Save per-device tasks to local file."""
        try:
            if not isinstance(tasks, list):
                return
            with self._storage_lock:
                state = load_tasks_state()
                devices = state.get("devices", {})
                device = devices.get(device_id, {})
                device.update(
                    {
                        "id": device_id,
                        "name": device_name,
                        "ip": ip,
                        "last_seen": time.time(),
                        "tasks": tasks,
                    }
                )
                devices[device_id] = device
                save_tasks_state({"devices": devices})
                log.info(f"Saved {len(tasks)} tasks for {device_name}")
        except Exception:
            log.exception("Failed to save tasks")

    def _save_automations(self, device_id, device_name, ip, automations):
        """Save per-device automations to local file."""
        try:
            if not isinstance(automations, list):
                return
            with self._storage_lock:
                state = load_automations_state()
                devices = state.get("devices", {})
                device = devices.get(device_id, {})
                device.update(
                    {
                        "id": device_id,
                        "name": device_name,
                        "ip": ip,
                        "last_seen": time.time(),
                        "automations": automations,
                    }
                )
                devices[device_id] = device
                save_automations_state({"devices": devices})
                log.info(f"Saved {len(automations)} automations for {device_name}")
        except Exception:
            log.exception("Failed to save automations")

    def _migrate_saved_device(self, old_device_id, new_device_id, device_name, ip):
        """Migrate persisted projects from an IP-based id to a stable device id."""
        try:
            with self._storage_lock:
                state = load_projects_state()
                devices = state.get("devices", {})

                old = devices.get(old_device_id)
                new = devices.get(new_device_id)

                if isinstance(old, dict) and not isinstance(new, dict):
                    devices[new_device_id] = dict(old)
                    devices.pop(old_device_id, None)
                elif isinstance(old, dict) and isinstance(new, dict):
                    # Merge projects if needed
                    if not new.get("projects") and old.get("projects"):
                        new["projects"] = old.get("projects")
                    devices.pop(old_device_id, None)
                    devices[new_device_id] = new

                if isinstance(devices.get(new_device_id), dict):
                    devices[new_device_id].update(
                        {
                            "id": new_device_id,
                            "name": device_name,
                            "ip": ip,
                            "last_seen": time.time(),
                        }
                    )

                save_projects_state({"devices": devices})
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
                if now - info["last_seen"] < 300:  # 5 minutes
                    active[did] = info
        return active


def load_projects_state():
    """Load per-device projects state from local file."""
    try:
        if os.path.exists(PROJECTS_FILE):
            with open(PROJECTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

                # v2 schema (multi-device)
                if isinstance(data, dict) and isinstance(data.get("devices"), dict):
                    devices = {
                        str(k): v
                        for k, v in data.get("devices", {}).items()
                        if isinstance(v, dict)
                    }
                    return {
                        "version": int(data.get("version", 2)),
                        "updated": float(data.get("updated", 0)),
                        "devices": devices,
                    }

                # v1 schema (single list of projects)
                legacy_projects = (
                    data.get("projects", []) if isinstance(data, dict) else []
                )
                return {
                    "version": 1,
                    "updated": float(data.get("updated", 0))
                    if isinstance(data, dict)
                    else 0.0,
                    "devices": {
                        "legacy": {
                            "id": "legacy",
                            "name": "Legacy",
                            "ip": None,
                            "last_seen": float(data.get("updated", 0))
                            if isinstance(data, dict)
                            else 0.0,
                            "projects": legacy_projects
                            if isinstance(legacy_projects, list)
                            else [],
                        }
                    },
                }
    except Exception:
        pass

    return {"version": 2, "updated": 0.0, "devices": {}}


def save_projects_state(state):
    """Persist per-device projects state to local file."""
    try:
        os.makedirs(os.path.dirname(PROJECTS_FILE), exist_ok=True)
        payload = {
            "version": 2,
            "updated": time.time(),
            "devices": state.get("devices", {}),
        }
        with open(PROJECTS_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def load_notes_state():
    """Load per-device notes state from local file."""
    try:
        if os.path.exists(NOTES_FILE):
            with open(NOTES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("devices"), dict):
                    devices = {
                        str(k): v
                        for k, v in data.get("devices", {}).items()
                        if isinstance(v, dict)
                    }
                    return {
                        "version": int(data.get("version", 1)),
                        "updated": float(data.get("updated", 0)),
                        "devices": devices,
                    }
    except Exception:
        pass
    return {"version": 1, "updated": 0.0, "devices": {}}


def save_notes_state(state):
    """Persist per-device notes state to local file."""
    try:
        os.makedirs(os.path.dirname(NOTES_FILE), exist_ok=True)
        payload = {
            "version": 1,
            "updated": time.time(),
            "devices": state.get("devices", {}),
        }
        with open(NOTES_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def load_agents_state():
    """Load per-device agents state from local file."""
    try:
        if os.path.exists(AGENTS_FILE):
            with open(AGENTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("devices"), dict):
                    return data
    except Exception:
        pass
    return {"version": 1, "updated": 0.0, "devices": {}}


def save_agents_state(state):
    """Persist per-device agents state to local file."""
    try:
        os.makedirs(os.path.dirname(AGENTS_FILE), exist_ok=True)
        payload = {
            "version": 1,
            "updated": time.time(),
            "devices": state.get("devices", {}),
        }
        with open(AGENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def load_tasks_state():
    """Load per-device tasks state from local file."""
    try:
        if os.path.exists(TASKS_FILE):
            with open(TASKS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("devices"), dict):
                    return data
    except Exception:
        pass
    return {"version": 1, "updated": 0.0, "devices": {}}


def save_tasks_state(state):
    """Persist per-device tasks state to local file."""
    try:
        os.makedirs(os.path.dirname(TASKS_FILE), exist_ok=True)
        payload = {
            "version": 1,
            "updated": time.time(),
            "devices": state.get("devices", {}),
        }
        with open(TASKS_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def load_automations_state():
    """Load per-device automations state from local file."""
    try:
        if os.path.exists(AUTOMATIONS_FILE):
            with open(AUTOMATIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("devices"), dict):
                    return data
    except Exception:
        pass
    return {"version": 1, "updated": 0.0, "devices": {}}


def save_automations_state(state):
    """Persist per-device automations state to local file."""
    try:
        os.makedirs(os.path.dirname(AUTOMATIONS_FILE), exist_ok=True)
        payload = {
            "version": 1,
            "updated": time.time(),
            "devices": state.get("devices", {}),
        }
        with open(AUTOMATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def get_note_content_from_cache(note_id):
    """Get note content from local cache if available."""
    try:
        state = load_notes_state()
        devices = state.get("devices", {})
        for device_id, device_info in devices.items():
            for note in device_info.get("notes", []):
                if note.get("id") == note_id:
                    content = note.get("content")
                    if content:
                        return {
                            "id": note_id,
                            "title": note.get("title", "Untitled"),
                            "content": content,
                            "updatedAt": note.get("updatedAt", 0),
                        }
    except Exception:
        pass
    return None


def is_pc_path(path):
    """Check if path is a PC path (starts with drive letter)."""
    if not path:
        return False
    # Windows drive letter pattern (C:\, D:\, etc.)
    if len(path) >= 2 and path[1] == ":":
        return True
    # Unix absolute path
    if path.startswith("/") and not path.startswith("/storage/"):
        return True
    return False


def open_folder(path):
    """Open folder in file explorer."""
    try:
        if IS_WINDOWS:
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception:
        pass


class CompanionRelayServer(threading.Thread):
    """TCP server that relays messages between Claude Code plugin and Android app.

    This enables the Claude Code Companion feature:
    - Plugin sends approval requests, session events to Android
    - Android sends approval responses, user replies back to plugin
    - Clipboard sync for reply injection
    - Session handoff: Android can request PC session context
    """

    # SECURITY: Limit concurrent connections to prevent resource exhaustion
    MAX_WORKERS = 10

    def __init__(self, on_status_change=None, approved_devices=None):
        super().__init__(daemon=True)
        self.running = True
        self.sock = None
        self.on_status_change = on_status_change
        self.bind_failed = False  # Track if port binding failed
        self.bind_error = None  # Store error message

        # SECURITY: Reference to approved device set from DataReceiver
        self._approved_devices = approved_devices

        # Connected clients
        self._plugin_conn = None  # Connection from Claude Code plugin
        self._device_conns = {}  # device_id -> connection
        self._conns_lock = threading.Lock()

        # Pending messages for offline devices
        self._pending_messages = {}  # device_id -> [messages]
        self._pending_lock = threading.Lock()

        # PC Session state for handoff (enables Android to continue PC sessions)
        self._pc_session = None  # Current active session
        self._pc_session_lock = threading.Lock()
        self._session_history = []  # Recent interactions for context (max 20)

        # Transcript watcher for PC session streaming
        self._transcript_watcher_thread = None
        self._transcript_path = None
        self._transcript_last_pos = 0
        self._transcript_stop_event = threading.Event()

        # SECURITY: Use thread pool instead of unbounded daemon threads
        self._executor = ThreadPoolExecutor(
            max_workers=self.MAX_WORKERS, thread_name_prefix="CompanionRelay"
        )

    def run(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(("0.0.0.0", COMPANION_PORT))
            self.sock.listen(5)
            self.sock.settimeout(1.0)

            log.info(f"[OK] Companion relay server listening on port {COMPANION_PORT}")
            self.bind_failed = False
            self.bind_error = None
            if self.on_status_change:
                self.on_status_change("listening", COMPANION_PORT)

            while self.running:
                try:
                    conn, addr = self.sock.accept()
                    # SECURITY: Use thread pool with bounded size instead of unbounded threads
                    self._executor.submit(self._handle_connection, conn, addr)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        log.error(f"Companion accept error: {e}")
        except OSError as e:
            log.error(
                f"[ERR] FATAL: CompanionRelay cannot bind to port {COMPANION_PORT}: {e}"
            )
            log.error(f"Another application may be using port {COMPANION_PORT}")
            log.error(
                "Claude Code companion relay will NOT work until this is resolved"
            )
            self.bind_failed = True
            self.bind_error = str(e)
            if self.on_status_change:
                self.on_status_change("bind_failed", COMPANION_PORT)
        except Exception as e:
            log.error(f"[ERR] FATAL: CompanionRelay unexpected error: {e}")
            self.bind_failed = True
            self.bind_error = str(e)
            if self.on_status_change:
                self.on_status_change("error", COMPANION_PORT)
        finally:
            # Gracefully shutdown thread pool with timeout to prevent hanging
            try:
                self._executor.shutdown(wait=True, timeout=5.0)
            except Exception:
                self._executor.shutdown(wait=False, cancel_futures=True)
            if self.sock:
                self.sock.close()

    def stop(self):
        self.running = False
        with self._conns_lock:
            if self._plugin_conn:
                try:
                    self._plugin_conn.close()
                except (socket.error, OSError):
                    pass  # Connection already closed
            for conn in self._device_conns.values():
                try:
                    conn.close()
                except (socket.error, OSError):
                    pass  # Connection already closed

    def _handle_connection(self, conn, addr):
        """Handle incoming connection (either plugin or device)."""
        # Categorize connection source for debugging multi-IP strategy
        ip = addr[0]
        if ip.startswith("192.168.") or ip.startswith("10.") or ip.startswith("172."):
            conn_type = "LAN"
        elif ip.startswith("100."):
            conn_type = "Tailscale"
        elif ip.startswith("127."):
            conn_type = "Localhost"
        else:
            conn_type = "Other"

        log.info(f"Companion connection from {addr} ({conn_type})")
        client_type = None
        device_id = None

        try:
            # Set reasonable timeout for reads (30 seconds)
            conn.settimeout(30.0)

            # Enable TCP keepalive to detect stale connections
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

            # Platform-specific keepalive settings
            if platform.system() == "Windows":
                # Windows: (enabled, idle_time_ms, interval_ms)
                # Start probing after 10 seconds idle, probe every 3 seconds
                conn.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 10000, 3000))
            else:
                # Linux/Mac: Set keepalive parameters
                try:
                    conn.setsockopt(
                        socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 10
                    )  # 10 seconds idle
                    conn.setsockopt(
                        socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 3
                    )  # 3 second intervals
                    conn.setsockopt(
                        socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3
                    )  # 3 probes before giving up
                except (AttributeError, OSError):
                    # Some platforms don't support these options
                    pass

            log.debug(f"TCP keepalive enabled for {addr}")

            while self.running:
                # Read length-prefixed message
                length_bytes = conn.recv(4)
                if not length_bytes or len(length_bytes) < 4:
                    break

                length = int.from_bytes(length_bytes, "big")
                if length <= 0 or length > 1_000_000:
                    log.warning(f"Invalid message length from {addr}: {length}")
                    break

                data = b""
                while len(data) < length:
                    chunk = conn.recv(min(length - len(data), 8192))
                    if not chunk:
                        break
                    data += chunk

                if len(data) < length:
                    break

                try:
                    message = json.loads(data.decode("utf-8"))
                    msg_type = message.get("type", "")

                    # Handle handshake to identify client type
                    if msg_type == "handshake":
                        device_id = message.get("deviceId")
                        if device_id:
                            # This is an Android device
                            client_type = "device"
                            with self._conns_lock:
                                self._device_conns[device_id] = conn
                            log.info(f"Device connected: {device_id}")

                            # Send any pending messages
                            self._send_pending_messages(device_id, conn)

                            # Send handshake ack
                            self._send_to_conn(
                                conn, {"type": "handshake_ack", "success": True}
                            )
                        else:
                            # This is the plugin
                            client_type = "plugin"
                            with self._conns_lock:
                                self._plugin_conn = conn
                            log.info("Plugin connected")
                            self._send_to_conn(
                                conn, {"type": "handshake_ack", "success": True}
                            )

                    # Handle approval response from device
                    elif msg_type == "approval_response":
                        self._relay_to_plugin(message)

                    # Handle user input/reply from device
                    elif msg_type == "user_input":
                        payload = message.get("payload", {})
                        text = payload.get("text", "")
                        action = payload.get("action", "")

                        # SECURITY: Check device is approved before allowing sensitive actions
                        device_is_approved = (
                            self._approved_devices is not None
                            and device_id
                            and device_id in self._approved_devices
                        )

                        if action == "clipboard_sync" and text and device_is_approved:
                            # Copy to Windows clipboard
                            self._copy_to_clipboard(text)
                            log.info(f"Reply copied to clipboard: {text[:50]}...")
                        elif action == "clipboard_sync" and not device_is_approved:
                            log.warning(f"Blocked clipboard_sync from unapproved device: {device_id}")

                        elif action == "terminal_inject" and text:
                            # SECURITY: Only approved devices can inject terminal commands
                            if not device_is_approved:
                                log.warning(
                                    f"BLOCKED terminal_inject from unapproved device: {device_id}"
                                )
                                result_msg = {
                                    "type": "terminal_inject_result",
                                    "success": False,
                                    "error": "Device not approved for terminal injection",
                                }
                                self._send_to_conn(conn, result_msg)
                            else:
                                auto_submit = payload.get("autoSubmit", True)
                                success = self._inject_terminal_input(
                                    text, auto_submit=auto_submit
                                )
                                # Send result back to device
                                result_msg = {
                                    "type": "terminal_inject_result",
                                    "success": success,
                                    "text": text[:50],
                                }
                                self._send_to_conn(conn, result_msg)
                                log.info(
                                    f"Terminal inject {'succeeded' if success else 'failed'}: {text[:50]}..."
                                )

                        # Also relay to plugin
                        self._relay_to_plugin(message)

                    # Handle messages from plugin to device
                    elif msg_type in [
                        "approval_request",
                        "session_start",
                        "session_complete",
                        "session_end",
                        "notification",
                        "approval_dismiss",
                        "automation_trigger",
                        "agent_command",
                    ]:
                        target_device = message.get("deviceId")

                        # Track PC session state for handoff
                        self._track_session_event(msg_type, message)

                        relayed = self._relay_to_device(target_device, message)

                        # Send immediate ack for fire-and-forget messages
                        if msg_type in [
                            "session_start",
                            "session_end",
                            "session_complete",
                            "notification",
                            "approval_dismiss",
                        ]:
                            ack = {
                                "type": "ack",
                                "success": relayed,
                                "originalType": msg_type,
                            }
                            self._send_to_conn(conn, ack)
                        # For approval_request, response comes async from device
                        # The plugin connection is tracked via _plugin_conn from handshake

                    # Handle Android requesting PC session for handoff
                    elif msg_type == "get_pc_session":
                        session_data = self._get_pc_session_for_handoff()
                        response = {
                            "type": "pc_session_response",
                            "id": f"msg_{int(time.time() * 1000)}",
                            "timestamp": int(time.time() * 1000),
                            "session": session_data,
                        }
                        self._send_to_conn(conn, response)
                        log.info(f"Sent PC session data to device: {device_id}")

                    # Internal relay action for Task completion notifications
                    elif msg_type == "relay_message":
                        inner_message = message.get("message")
                        if inner_message:
                            self._relay_to_all_devices(inner_message)
                            log.info(
                                f"Internal relay successful for: {inner_message.get('type')}"
                            )

                    else:
                        log.debug(f"Unknown message type: {msg_type}")

                except json.JSONDecodeError:
                    log.warning(f"Invalid JSON from {addr}")
                except Exception as e:
                    log.error(f"Error processing message: {e}")

        except socket.timeout:
            log.warning(f"Connection timeout from {addr} - stale connection detected")
            if device_id:
                log.info(f"Removing stale device connection: {device_id}")
        except Exception as e:
            log.debug(f"Companion connection closed: {e}")
        finally:
            # Clean up connection
            try:
                conn.shutdown(socket.SHUT_WR)
                time.sleep(0.5)  # Allow time for data flush
            except (socket.error, OSError):
                pass
            with self._conns_lock:
                if client_type == "plugin" and self._plugin_conn == conn:
                    self._plugin_conn = None
                    log.info("Plugin disconnected")
                elif client_type == "device" and device_id:
                    self._device_conns.pop(device_id, None)
                    log.info(f"Device disconnected: {device_id}")
            try:
                conn.close()
            except (socket.error, OSError):
                pass  # Connection already closed

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
                    except (
                        socket.error,
                        BrokenPipeError,
                        ConnectionResetError,
                        OSError,
                    ):
                        log.debug(f"Failed to relay to device {did}: connection closed")

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
                self._pending_messages[device_id] = self._pending_messages[device_id][
                    -50:
                ]

        log.info(f"Message queued for offline device: {device_id or 'any'}")
        return False

    def _send_pending_messages(self, device_id, conn):
        """Send any pending messages to a newly connected device."""
        with self._pending_lock:
            messages = self._pending_messages.pop(device_id, [])
        for msg in messages:
            try:
                self._send_to_conn(conn, msg)
            except (socket.error, BrokenPipeError, ConnectionResetError, OSError):
                log.debug(
                    f"Failed to send pending message to {device_id}: connection closed"
                )
                break

    def _send_to_conn(self, conn, message):
        """Send length-prefixed JSON message to connection."""
        data = json.dumps(message).encode("utf-8")
        conn.sendall(len(data).to_bytes(4, "big"))
        conn.sendall(data)

    def _notify_web_dashboard_sessions_updated(self, device_id):
        """Notify web dashboard that sessions have been updated (triggers WebSocket broadcast)."""

        def do_notify():
            try:
                import urllib.request
                import json as json_module

                data = json_module.dumps({"device_id": device_id}).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:6767/api/sessions/sync",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=2)
            except Exception as e:
                log.debug(f"Could not notify web dashboard: {e}")

        threading.Thread(target=do_notify, daemon=True).start()

    def _notify_web_dashboard_cards_updated(self, device_id):
        """Notify web dashboard that cards have been updated (triggers WebSocket broadcast)."""

        def do_notify():
            try:
                import urllib.request
                import json as json_module

                data = json_module.dumps({"device_id": device_id}).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:6767/api/cards/sync",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=2)
            except Exception as e:
                log.debug(f"Could not notify web dashboard: {e}")

        threading.Thread(target=do_notify, daemon=True).start()

    def _notify_web_dashboard_collections_updated(self, device_id):
        """Notify web dashboard that collections have been updated (triggers WebSocket broadcast)."""

        def do_notify():
            try:
                import urllib.request
                import json as json_module

                data = json_module.dumps({"device_id": device_id}).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:6767/api/collections/sync",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=2)
            except Exception as e:
                log.debug(f"Could not notify web dashboard: {e}")

                threading.Thread(target=do_notify, daemon=True).start()

        def _notify_web_dashboard_session_message(
            self, session_id, message, is_update=False
        ):
            """Send a session message update to the web dashboard."""

            def do_notify():
                try:
                    import urllib.request

                    import json as json_module

                    payload = json_module.dumps(
                        {
                            "session_id": session_id,
                            "message": message,
                            "is_update": bool(is_update),
                        }
                    ).encode("utf-8")

                    req = urllib.request.Request(
                        "http://127.0.0.1:6767/api/sessions/message",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )

                    urllib.request.urlopen(req, timeout=2)

                except Exception as e:
                    log.debug(f"Could not notify web dashboard: {e}")

            # Run in background thread to not block the relay

            threading.Thread(target=do_notify, daemon=True).start()

        def _start_transcript_watcher(self, transcript_path):
            """Start watching a transcript file for new content."""

            if not transcript_path or not os.path.exists(transcript_path):
                log.warning(
                    f"Transcript path invalid or doesn't exist: {transcript_path}"
                )

                return

            self._stop_transcript_watcher()

            self._transcript_path = transcript_path

            self._transcript_last_pos = (
                os.path.getsize(transcript_path)
                if os.path.exists(transcript_path)
                else 0
            )

            self._transcript_stop_event.clear()

            self._transcript_watcher_thread = threading.Thread(
                target=self._watch_transcript, daemon=True
            )

        self._transcript_watcher_thread.start()
        log.info(f"Started transcript watcher for: {transcript_path}")

    def _stop_transcript_watcher(self):
        """Stop the transcript watcher."""
        self._transcript_stop_event.set()
        if (
            self._transcript_watcher_thread
            and self._transcript_watcher_thread.is_alive()
        ):
            self._transcript_watcher_thread.join(timeout=2)
        self._transcript_watcher_thread = None
        self._transcript_path = None

    def _watch_transcript(self):
        """Watch transcript file for new content and sync to web."""
        while not self._transcript_stop_event.is_set():
            try:
                if not self._transcript_path or not os.path.exists(
                    self._transcript_path
                ):
                    time.sleep(1)
                    continue

                current_size = os.path.getsize(self._transcript_path)
                if current_size > self._transcript_last_pos:
                    with open(
                        self._transcript_path, "r", encoding="utf-8", errors="ignore"
                    ) as f:
                        f.seek(self._transcript_last_pos)
                        new_content = f.read()
                        self._transcript_last_pos = f.tell()

                    if new_content.strip():
                        self._parse_and_store_transcript(new_content)

                time.sleep(0.5)
            except Exception as e:
                log.error(f"Transcript watcher error: {e}")
                time.sleep(1)

    def _parse_and_store_transcript(self, content):
        """Parse transcript content and store messages for web sync."""
        session_id = self._pc_session.get("sessionId") if self._pc_session else None
        if not session_id:
            return

        provider = self._pc_session.get("provider") if self._pc_session else None
        model = self._pc_session.get("model") if self._pc_session else None
        device_id = (
            self._pc_session.get("deviceId")
            if self._pc_session
            else f"pc:{socket.gethostname()}"
        )
        device_name = (
            self._pc_session.get("deviceName")
            if self._pc_session
            else f"PC-{socket.gethostname()[:8]}"
        )

        lines = content.strip().split("\n")
        for line in lines:
            if not line.strip():
                continue

            role = "assistant"
            message_content = ""
            try:
                msg_data = json.loads(line)
                role = (
                    msg_data.get("role", msg_data.get("sender", "assistant"))
                    or "assistant"
                )
                message_content = msg_data.get("content", msg_data.get("message", ""))

                if isinstance(message_content, list):
                    text_parts = []
                    for block in message_content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "tool_use":
                                text_parts.append(
                                    f"[Tool: {block.get('name', 'unknown')}]"
                                )
                        elif isinstance(block, str):
                            text_parts.append(block)
                    message_content = "\n".join(text_parts)
            except json.JSONDecodeError:
                message_content = line.strip()

            if not message_content:
                continue

            message_payload = {
                "id": f"msg_{int(time.time() * 1000)}",
                "role": role or "assistant",
                "content": str(message_content)[:5000],
                "timestamp": int(time.time() * 1000),
                "device_id": device_id,
                "device_name": device_name,
                "backend_type": "SSH",
                "provider": provider,
                "model": model,
            }

            if SYNC_SERVICE_AVAILABLE:
                append_session_message(
                    session_id=session_id,
                    message=message_payload,
                    session_meta={
                        "device_id": device_id,
                        "device_name": device_name,
                        "backend_type": "SSH",
                        "provider": provider,
                        "model": model,
                    },
                )

            self._notify_web_dashboard_session_message(session_id, message_payload)

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
                text_bytes = text.encode("utf-16le") + b"\x00\x00"
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

                if platform.system() == "Darwin":
                    subprocess.run(["pbcopy"], input=text.encode(), check=True)
                else:
                    subprocess.run(
                        ["xclip", "-selection", "clipboard"],
                        input=text.encode(),
                        check=True,
                    )
            except Exception as e:
                log.warning(f"Clipboard copy not available: {e}")

    def _get_clipboard(self):
        """Get current clipboard contents (to restore later)."""
        if IS_WINDOWS:
            try:
                import ctypes
                from ctypes import wintypes

                CF_UNICODETEXT = 13

                ctypes.windll.user32.OpenClipboard(0)
                try:
                    if ctypes.windll.user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
                        handle = ctypes.windll.user32.GetClipboardData(CF_UNICODETEXT)
                        if handle:
                            ptr = ctypes.windll.kernel32.GlobalLock(handle)
                            if ptr:
                                # Read as wide string
                                text = ctypes.wstring_at(ptr)
                                ctypes.windll.kernel32.GlobalUnlock(handle)
                                return text
                finally:
                    ctypes.windll.user32.CloseClipboard()
            except Exception as e:
                log.debug(f"Failed to get clipboard: {e}")
        return None

    def _find_terminal_window(self):
        """Find a terminal window likely running Claude Code.

        Searches for windows with titles containing 'claude', 'Claude Code',
        'Windows Terminal', 'cmd.exe', or 'PowerShell'.
        Returns window handle (HWND) or None.
        """
        if not IS_WINDOWS:
            return None

        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32

            # Window patterns to look for (in priority order)
            patterns = [
                "claude",  # Most specific - Claude Code session
                "Claude Code",
                "Windows Terminal",  # Common modern terminal
                "Command Prompt",
                "PowerShell",
                "cmd.exe",
            ]

            found_windows = []

            # Callback for EnumWindows
            EnumWindowsProc = ctypes.WINFUNCTYPE(
                wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
            )

            def enum_callback(hwnd, lparam):
                if user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buffer = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buffer, length + 1)
                        title = buffer.value.lower()

                        for i, pattern in enumerate(patterns):
                            if pattern.lower() in title:
                                found_windows.append((i, hwnd, buffer.value))
                                break
                return True

            user32.EnumWindows(EnumWindowsProc(enum_callback), 0)

            if found_windows:
                # Sort by priority (lower index = higher priority)
                found_windows.sort(key=lambda x: x[0])
                best_match = found_windows[0]
                log.info(
                    f"Found terminal window: '{best_match[2]}' (hwnd={best_match[1]})"
                )
                return best_match[1]

            log.warning("No terminal window found")
            return None

        except Exception as e:
            log.error(f"Error finding terminal window: {e}")
            return None

    def _send_keystrokes(self, hwnd, ctrl_v=True, enter=True):
        """Send Ctrl+V and Enter keystrokes to a window.

        Uses SendInput for reliable keystroke injection.
        """
        if not IS_WINDOWS:
            return False

        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32

            # Bring window to foreground
            user32.SetForegroundWindow(hwnd)
            time.sleep(0.1)  # Brief delay for window to focus

            # Define input structures
            INPUT_KEYBOARD = 1
            KEYEVENTF_KEYUP = 0x0002

            VK_CONTROL = 0x11
            VK_V = 0x56
            VK_RETURN = 0x0D

            class KEYBDINPUT(ctypes.Structure):
                _fields_ = [
                    ("wVk", wintypes.WORD),
                    ("wScan", wintypes.WORD),
                    ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
                ]

            class INPUT(ctypes.Structure):
                _fields_ = [
                    ("type", wintypes.DWORD),
                    ("ki", KEYBDINPUT),
                    ("padding", ctypes.c_ubyte * 8),
                ]

            def make_key_input(vk, flags=0):
                inp = INPUT()
                inp.type = INPUT_KEYBOARD
                inp.ki.wVk = vk
                inp.ki.dwFlags = flags
                return inp

            inputs = []

            if ctrl_v:
                # Ctrl down, V down, V up, Ctrl up
                inputs.append(make_key_input(VK_CONTROL))
                inputs.append(make_key_input(VK_V))
                inputs.append(make_key_input(VK_V, KEYEVENTF_KEYUP))
                inputs.append(make_key_input(VK_CONTROL, KEYEVENTF_KEYUP))

            if enter:
                time.sleep(0.05)  # Brief delay before Enter
                inputs.append(make_key_input(VK_RETURN))
                inputs.append(make_key_input(VK_RETURN, KEYEVENTF_KEYUP))

            # Send all inputs
            if inputs:
                input_array = (INPUT * len(inputs))(*inputs)
                user32.SendInput(len(inputs), input_array, ctypes.sizeof(INPUT))

            log.info("Sent keystrokes to terminal")
            return True

        except Exception as e:
            log.error(f"Error sending keystrokes: {e}")
            return False

    def _inject_terminal_input(self, text, auto_submit=True):
        """Inject text into terminal by clipboard paste.

        Preserves user's clipboard contents by saving and restoring.

        Args:
            text: The text to inject
            auto_submit: If True, press Enter after pasting

        Returns:
            bool: True if successful
        """
        if not IS_WINDOWS:
            log.warning("Terminal injection only supported on Windows")
            return False

        # Find terminal window first
        hwnd = self._find_terminal_window()
        if not hwnd:
            log.error("No terminal window found for injection")
            return False

        # Save current clipboard
        original_clipboard = self._get_clipboard()
        log.debug(
            f"Saved clipboard: {original_clipboard[:50] if original_clipboard else 'None'}..."
        )

        try:
            # Copy our text to clipboard
            self._copy_to_clipboard(text)
            time.sleep(0.05)  # Brief delay for clipboard

            # Send Ctrl+V (and Enter if auto_submit)
            success = self._send_keystrokes(hwnd, ctrl_v=True, enter=auto_submit)

            if success:
                log.info(f"Injected text into terminal: {text[:50]}...")

            return success

        finally:
            # Restore original clipboard after a delay
            # (need to wait for paste to complete)
            def restore_clipboard():
                time.sleep(0.3)  # Wait for paste to complete
                if original_clipboard is not None:
                    self._copy_to_clipboard(original_clipboard)
                    log.debug("Restored original clipboard")

            threading.Thread(target=restore_clipboard, daemon=True).start()

    def _track_session_event(self, msg_type, message):
        """Track session events for handoff capability.

        Stores PC session state so Android can request context for continuing
        conversations across devices (PC -> Phone -> Car -> Watch).
        """
        with self._pc_session_lock:
            payload = message.get("payload", {})
            session_id = message.get("sessionId")
            timestamp = message.get("timestamp", int(time.time() * 1000))

            if msg_type == "session_start":
                # New Claude Code session started
                transcript_path = payload.get("transcriptPath", "")
                provider = (
                    payload.get("cliProvider")
                    or payload.get("provider")
                    or payload.get("clientName")
                    or "claude-code"
                )
                model = payload.get("model") or payload.get("modelName")
                device_id = f"pc:{socket.gethostname()}"
                self._pc_session = {
                    "sessionId": session_id,
                    "hostname": payload.get("hostname", socket.gethostname()),
                    "cwd": payload.get("cwd", ""),
                    "username": payload.get(
                        "username", os.environ.get("USERNAME", "user")
                    ),
                    "transcriptPath": transcript_path,
                    "startedAt": timestamp,
                    "lastActivityAt": timestamp,
                    "isActive": True,
                    "deviceType": "PC",
                    "deviceName": f"PC-{socket.gethostname()[:8]}",
                    "provider": provider,
                    "model": model,
                    "deviceId": device_id,
                }
                self._session_history = []  # Reset history for new session
                log.info(f"PC session started: {session_id}")

                # Start transcript watcher for bidirectional sync
                if transcript_path:
                    self._start_transcript_watcher(transcript_path)

                # Broadcast to all connected devices that PC session is available
                self._broadcast_session_availability()

                if SYNC_SERVICE_AVAILABLE:
                    upsert_session(
                        {
                            "id": session_id,
                            "device_id": device_id,
                            "device_name": self._pc_session.get("deviceName"),
                            "backend_type": "SSH",
                            "provider": provider,
                            "model": model,
                            "projectId": payload.get("projectId"),
                            "lastActivityAt": timestamp,
                            "startedAt": timestamp,
                            "isActive": True,
                        }
                    )
                    self._notify_web_dashboard_sessions_updated(device_id)

            elif msg_type in ["session_end", "session_complete"]:
                # Session ended
                if self._pc_session:
                    self._pc_session["isActive"] = False
                    self._pc_session["endedAt"] = timestamp
                    log.info(f"PC session ended: {session_id}")

                # Stop transcript watcher
                self._stop_transcript_watcher()

                if SYNC_SERVICE_AVAILABLE:
                    device_id = (
                        self._pc_session.get("deviceId")
                        if self._pc_session
                        else f"pc:{socket.gethostname()}"
                    )
                    upsert_session(
                        {
                            "id": session_id,
                            "lastActivityAt": timestamp,
                            "endedAt": timestamp,
                            "isActive": False,
                        }
                    )
                    self._notify_web_dashboard_sessions_updated(device_id)

            elif msg_type in ["approval_request", "notification"]:
                # Track activity for context
                if self._pc_session:
                    self._pc_session["lastActivityAt"] = timestamp

                    # Add to history for context (last 20 items)
                    history_item = {
                        "type": msg_type,
                        "timestamp": timestamp,
                        "summary": payload.get("prompt", payload.get("message", ""))[
                            :200
                        ],
                    }
                    if msg_type == "approval_request":
                        history_item["toolName"] = payload.get("toolName", "")

                    self._session_history.append(history_item)
                    if len(self._session_history) > 20:
                        self._session_history = self._session_history[-20:]

    def _broadcast_session_availability(self):
        """Notify all connected devices that a PC session is available for handoff."""
        with self._pc_session_lock:
            if not self._pc_session:
                return

            message = {
                "type": "pc_session_available",
                "id": f"msg_{int(time.time() * 1000)}",
                "timestamp": int(time.time() * 1000),
                "payload": {
                    "sessionId": self._pc_session.get("sessionId"),
                    "hostname": self._pc_session.get("hostname"),
                    "cwd": self._pc_session.get("cwd"),
                    "deviceName": self._pc_session.get("deviceName"),
                    "startedAt": self._pc_session.get("startedAt"),
                },
            }

        # Send to all connected devices
        with self._conns_lock:
            for device_id, conn in self._device_conns.items():
                try:
                    self._send_to_conn(conn, message)
                    log.info(f"Broadcast PC session availability to {device_id}")
                except Exception as e:
                    log.debug(f"Failed to broadcast to {device_id}: {e}")

    def _get_pc_session_for_handoff(self):
        """Get current PC session data for handoff to Android.

        Returns session context that Android can use to continue the conversation.
        """
        with self._pc_session_lock:
            if not self._pc_session or not self._pc_session.get("isActive"):
                return None

            return {
                "version": 1,
                "session": {
                    "sessionId": self._pc_session.get("sessionId"),
                    "projectId": "pc_claude_code",  # Virtual project for PC sessions
                    "hostname": self._pc_session.get("hostname"),
                    "cwd": self._pc_session.get("cwd"),
                    "username": self._pc_session.get("username"),
                    "deviceType": "PC",
                    "deviceName": self._pc_session.get("deviceName"),
                    "startedAt": self._pc_session.get("startedAt"),
                    "lastActivityAt": self._pc_session.get("lastActivityAt"),
                    "isActive": True,
                },
                "recentActivity": list(self._session_history),
                "handoffHint": f"Continue your Claude Code session from {self._pc_session.get('hostname')}",
            }

    def get_active_pc_session(self):
        """Public method to check if there's an active PC session (for UI display)."""
        with self._pc_session_lock:
            if self._pc_session and self._pc_session.get("isActive"):
                return self._pc_session.copy()
            return None


class DiscoveryServer(threading.Thread):
    """UDP server that responds to discovery requests and registers mDNS service."""

    def __init__(self, connection_info, app_instance=None):
        super().__init__(daemon=True)
        self.connection_info = connection_info
        self.app_instance = app_instance
        self.running = True
        self.sock = None
        self.bind_failed = False  # Track if port binding failed
        self.bind_error = None  # Store error message
        self.zeroconf = None
        self.service_infos = []  # Store multiple service infos

    def run(self):
        # 1. Register mDNS services
        if HAS_ZEROCONF:
            try:
                self.zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
                desc = {
                    "username": self.connection_info.get("username", "root"),
                    "version": str(self.connection_info.get("version", 1)),
                    "mode": self.connection_info.get("mode", "local"),
                }

                hostname = socket.gethostname()
                # Use non-blocking IP detection - returns cached value or starts background detection
                # This prevents DiscoveryServer thread from blocking during startup
                local_ip = get_local_ip(wait=False)
                tailscale_ip = get_tailscale_ip()
                ssh_port = self.connection_info.get("port", 22)

                # If IPs are still detecting, schedule a delayed retry to register services later
                if (
                    not local_ip
                    or local_ip == "Detecting..."
                    or (
                        tailscale_ip == "Detecting..."
                        and local_ip not in ("Detecting...", "127.0.0.1", "0.0.0.0")
                    )
                ):
                    # Defer service registration to background thread when IPs are detected
                    def register_when_ready():
                        time.sleep(2)  # Give IP detection time to complete
                        local_ip = get_local_ip(wait=False)
                        tailscale_ip = get_tailscale_ip()
                        # Retry registration with updated IPs
                        if self.zeroconf and self.running:
                            try:
                                if local_ip and local_ip not in (
                                    "Detecting...",
                                    "127.0.0.1",
                                    "0.0.0.0",
                                ):
                                    info_local = ServiceInfo(
                                        "_shadowai._tcp.local.",
                                        f"{hostname} (Local)._shadowai._tcp.local.",
                                        addresses=[socket.inet_aton(local_ip)],
                                        port=ssh_port,
                                        properties=desc,
                                        server=f"{hostname}.local.",
                                    )
                                    self.zeroconf.register_service(info_local)
                                    self.service_infos.append(info_local)
                                    log.info(
                                        f"[OK] mDNS service registered (Local): {local_ip}"
                                    )
                            except Exception as e:
                                log.debug(f"Failed to register local mDNS: {e}")

                    threading.Thread(target=register_when_ready, daemon=True).start()

                # 1. Register Local IP
                if local_ip and local_ip not in (
                    "Detecting...",
                    "127.0.0.1",
                    "0.0.0.0",
                ):
                    try:
                        info_local = ServiceInfo(
                            "_shadowai._tcp.local.",
                            f"{hostname} (Local)._shadowai._tcp.local.",
                            addresses=[socket.inet_aton(local_ip)],
                            port=ssh_port,
                            properties=desc,
                            server=f"{hostname}.local.",
                        )
                        self.zeroconf.register_service(info_local)
                        self.service_infos.append(info_local)
                        log.info(f"[OK] mDNS service registered (Local): {local_ip}")
                    except Exception as e:
                        log.debug(f"Failed to register local mDNS: {e}")

                # Register Tailscale IP if available
                if tailscale_ip and tailscale_ip not in (
                    "Detecting...",
                    "127.0.0.1",
                    "0.0.0.0",
                ):
                    try:
                        info_ts = ServiceInfo(
                            "_shadowai._tcp.local.",
                            f"{hostname} (Tailscale)._shadowai._tcp.local.",
                            addresses=[socket.inet_aton(tailscale_ip)],
                            port=ssh_port,
                            properties=desc,
                            server=f"{hostname}-ts.local.",
                        )
                        self.zeroconf.register_service(info_ts)
                        self.service_infos.append(info_ts)
                        log.info(
                            f"[OK] mDNS service registered (Tailscale): {tailscale_ip}"
                        )
                    except Exception as e:
                        log.debug(f"Failed to register tailscale mDNS: {e}")

            except Exception as e:
                log.warning(f"Failed to register mDNS service: {e}")
        else:
            log.warning("Zeroconf module not found - mDNS discovery disabled")

        # 2. Start UDP discovery responder
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind(("", DISCOVERY_PORT))
            self.sock.settimeout(1.0)
            log.info(f"[OK] Discovery server listening on UDP {DISCOVERY_PORT}")
        except OSError as e:
            log.error(
                f"[ERR] FATAL: DiscoveryServer cannot bind to port {DISCOVERY_PORT}: {e}"
            )
            log.error(f"Another application may be using UDP port {DISCOVERY_PORT}")
            log.error("Network discovery will NOT work until this is resolved")
            self.bind_failed = True
            self.bind_error = str(e)
            return  # Exit thread - cannot continue without port
        except Exception as e:
            log.error(f"[ERR] FATAL: DiscoveryServer unexpected error during bind: {e}")
            self.bind_failed = True
            self.bind_error = str(e)
            return

        # Port binding successful
        self.bind_failed = False
        self.bind_error = None

        # Start active broadcast thread
        def active_broadcast():
            broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            broadcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            while self.running:
                try:
                    # Refresh info before broadcasting
                    info = self.connection_info
                    if self.app_instance and hasattr(
                        self.app_instance, "get_connection_info"
                    ):
                        info = self.app_instance.get_connection_info()

                    # Broadcast to LAN
                    msg = DISCOVERY_MAGIC + json.dumps(info).encode()
                    broadcast_sock.sendto(msg, ("255.255.255.255", DISCOVERY_PORT))
                except Exception:
                    pass
                time.sleep(5.0)  # Broadcast every 5 seconds
            broadcast_sock.close()

        threading.Thread(target=active_broadcast, daemon=True).start()

        try:
            while self.running:
                try:
                    data, addr = self.sock.recvfrom(1024)
                    if data.startswith(DISCOVERY_MAGIC):
                        # Refresh info before responding to ensure IPs are current with LAN priority
                        fresh_info = self.connection_info
                        if self.app_instance and hasattr(
                            self.app_instance, "get_connection_info"
                        ):
                            fresh_info = self.app_instance.get_connection_info()

                        # Log discovery response with IP count for debugging
                        num_hosts = len(fresh_info.get("hosts", []))
                        primary = fresh_info.get("host", "N/A")
                        log.debug(
                            f"Discovery response to {addr[0]}: {num_hosts} IPs (primary: {primary})"
                        )

                        response = DISCOVERY_MAGIC + json.dumps(fresh_info).encode()
                        self.sock.sendto(response, addr)
                except socket.timeout:
                    continue
                except Exception as e:
                    # Only log non-timeout errors to reduce spam
                    if self.running and not isinstance(e, socket.timeout):
                        log.debug(f"Discovery error: {e}")
        finally:
            if self.sock:
                self.sock.close()
            if self.zeroconf:
                try:
                    for info in self.service_infos:
                        self.zeroconf.unregister_service(info)
                    self.zeroconf.close()
                except Exception:
                    pass

    def stop(self):
        self.running = False

    def update_info(self, info):
        self.connection_info = info


class CloudSyncService(threading.Thread):
    """

    Background service that pushes local ShadowBridge data (~/.shadowai/*.json)

    to the Central Cloud Backend (api.driver.ai).

    """

    SYNC_INTERVAL = 300  # 5 minutes

    def __init__(self):
        super().__init__(daemon=True)

        self.running = True

        self.api_token = None

        self.base_url = "https://api.driver.ai/v1/content"

        self._load_token()

    def _load_token(self):
        """Try to find the API token from connected devices or settings."""

        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    settings = json.load(f)

                    self.api_token = settings.get("cloud_api_token")

        except Exception:
            pass

    def run(self):
        log.info("Cloud Sync Service starting...")

        while self.running:
            if self.api_token:
                try:
                    self._perform_sync()

                except Exception as e:
                    log.error(f"Cloud sync error: {e}")

            time.sleep(self.SYNC_INTERVAL)

    def _perform_sync(self):
        """Push local JSON data to cloud."""

        # Sync Projects

        if os.path.exists(PROJECTS_FILE):
            with open(PROJECTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

                # Projects are under devices in this file

                for device_id, device_info in data.get("devices", {}).items():
                    projects = device_info.get("projects", [])

                    for proj in projects:
                        self._push_to_cloud("projects", proj)

        # Sync Notes

        if os.path.exists(NOTES_FILE):
            with open(NOTES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

                for device_id, device_info in data.get("devices", {}).items():
                    notes = device_info.get("notes", [])

                    for note in notes:
                        self._push_to_cloud("notes", note)

    def _push_to_cloud(self, endpoint, payload):
        """HTTP POST to cloud backend."""

        try:
            import requests

            headers = {"Authorization": f"Bearer {self.api_token}"}

            url = f"{self.base_url}/{endpoint}"

            # Re-map local fields to backend fields if necessary

            # Backend expects project: {name, description, path, metadata}

            # Backend expects note: {title, content, project_id, category, tags}

            res = requests.post(url, json=payload, headers=headers, timeout=10)

            if res.status_code == 201:
                # Reduced logging - only log errors, not every successful sync
                pass

        except Exception as e:
            # Only log errors, suppress common transient failures
            log.debug(f"Cloud sync item failed: {e}")


class ShadowBridgeApp:
    """Main application - compact popup style."""

    def __init__(self):
        # Reduced logging at startup to prevent log spam
        log.info(
            f"ShadowBridge v{APP_VERSION} initializing (frozen={getattr(sys, 'frozen', False)})"
        )

        try:
            set_app_user_model_id("ShadowBridge")
            self.root = tk.Tk()
            # Set background IMMEDIATELY to prevent white flash
            self.root.configure(bg=COLORS["bg_surface"])
            self.root.title(APP_NAME)
        except Exception as e:
            log.critical(f"FATAL: Failed to create root window: {e}", exc_info=True)
            raise

        # Apply modern Windows theme if available
        theme_applied = False
        if HAS_SV_TTK:
            try:
                sv_ttk.set_theme("dark")
                theme_applied = True
            except Exception as e:
                log.error(f"Failed to set sv_ttk theme: {e}")

        # Set window icon from logo.png
        icon_path = get_app_icon_path()
        if icon_path and HAS_PIL:
            try:
                icon_img = Image.open(icon_path)
                icon_img = icon_img.resize(
                    (64, 64),
                    Image.Resampling.LANCZOS
                    if hasattr(Image, "Resampling")
                    else Image.LANCZOS,
                )
                self.app_icon = ImageTk.PhotoImage(icon_img)
                self.root.iconphoto(True, self.app_icon)
            except Exception as e:
                log.debug(f"Could not load app icon: {e}")

        # Window sizing - compact size, ALLOW MOVABLE AND RESIZABLE
        self.root.update_idletasks()
        try:
            dpi = self.root.winfo_fpixels("1i")
            scale = dpi / 96.0  # 96 is standard DPI
            # Skip verbose logging to reduce startup log spam
        except Exception:
            scale = 1.0

        # Fixed compact size for quick connect utility
        self.window_width = int(480 * scale)
        self.window_height = int(780 * scale)

        # Set minimum size and enable resizing
        self.root.minsize(320, 600)
        self.root.resizable(True, True)

        # State - auto-detect SSH port (now non-blocking)
        self.ssh_port = find_ssh_port() or 22
        log.info(f"Initial SSH port: {self.ssh_port}")

        # Start a background thread to refresh UI when SSH port is detected
        def delayed_ssh_refresh():
            # Shorter, more efficient retry loop - checks 5 times with 2s intervals
            max_attempts = 5
            for attempt in range(max_attempts):
                detected = find_ssh_port()
                if detected and detected != self.ssh_port:
                    self.ssh_port = detected
                    log.info(f"Updated SSH port: {self.ssh_port}")
                    self.root.after(0, self.update_status)
                    self.root.after(0, self.update_qr_code)
                    return
                # Only sleep if we need to retry (not last attempt)
                if attempt < max_attempts - 1:
                    time.sleep(2)

        threading.Thread(target=delayed_ssh_refresh, daemon=True).start()

        self.is_broadcasting = False
        self.discovery_server = None
        self.data_receiver = None
        self.companion_relay = None
        self.web_process = None
        self.web_server_thread = None
        self.tray_icon = None
        self.selected_device_id = "__ALL__"  # '__ALL__' or a device_id
        self.devices = load_projects_state().get(
            "devices", {}
        )  # device_id -> {name, ip, last_seen, projects}
        self.notes_devices = load_notes_state().get(
            "devices", {}
        )  # device_id -> {name, ip, last_seen, notes}
        self.sessions_devices = {}  # device_id -> {name, ip, last_seen, sessions}
        self.cards_devices = {}  # device_id -> {name, ip, last_seen, cards}
        self.collections_devices = {}  # device_id -> {name, ip, last_seen, collections}
        self.selected_notes_device_id = "__ALL__"
        self._device_menu_updating = False

        self._auto_web_dashboard_attempts = 0

        # Initialize services but defer heavy startup operations to background
        # This improves startup time and prevents UI freezing
        self.signaling_service = None
        self.sentinel = None
        self.watchdog = None
        self.upnp_manager = None

        # Start background services in a thread to avoid blocking UI initialization
        def start_background_services():
            # Start Signaling Service for instant discovery/roaming
            try:
                from zeroconf import IPVersion, ServiceInfo, Zeroconf

                self.signaling_service = SignalingService()
                self.signaling_service.start()
                # Skip verbose logging to reduce startup log spam
            except Exception as e:
                # Only log actual errors, not startup info
                pass

            # Start UPnP Port Mapping
            try:
                self.upnp_manager = UPnPManager()
                self.upnp_manager.discover_and_map()
                # Skip verbose logging to reduce startup log spam
            except Exception as e:
                # Only log actual errors
                pass

            # Start Ouroboros Sentinel
            if HAS_SENTINEL:
                try:
                    self.sentinel = Sentinel()
                    self.sentinel.start()
                    # Skip verbose logging to reduce startup log spam
                except Exception as e:
                    # Only log actual errors
                    pass

            # Start Thread Watchdog for server health monitoring
            try:
                from utils.watchdog import ThreadWatchdog

                self.watchdog = ThreadWatchdog(check_interval=5)
                self.watchdog.start()
                # Skip verbose logging to reduce startup log spam
            except Exception as e:
                # Only log actual errors
                pass

        threading.Thread(target=start_background_services, daemon=True).start()

        # Setup modern styles
        # Skip verbose logging to reduce startup log spam
        try:
            self.setup_styles()
            log.info("[OK] Styles setup completed successfully")
        except Exception as e:
            log.error(f"\u2717 Failed to setup styles: {e}", exc_info=True)

        # Initialize Threading Primitives
        self._web_dashboard_monitor_stop_event = threading.Event()
        self._web_dashboard_start_lock = threading.Lock()
        self._tool_install_lock = threading.Lock()
        self._tool_install_procs = {}
        self._web_dashboard_monitor_thread = None

        # Build UI
        log.info("Creating widgets...")
        try:
            self.create_widgets()
            log.info("[OK] Widgets created successfully")
        except Exception as e:
            log.critical(f"\u2717 FATAL: Failed to create widgets: {e}", exc_info=True)
            raise

        # Set window size and position (load saved state or default to bottom right)
        if not self._load_window_state():
            # Fallback if loading fails
            self.root.geometry(f"{self.window_width}x{self.window_height}")

        # Apply theme once after a short delay to ensure window handle is ready
        self.root.after(200, lambda: apply_windows_11_theme(self.root))

        # Start data receiver for project sync
        self.start_data_receiver()

        # Handle close and minimize
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Unmap>", self.on_minimize)

        # Initial updates - consolidated to reduce startup delays
        self.root.after(300, self.update_qr_code)
        self.root.after(500, self.update_status)
        self.root.after(800, self.auto_start_broadcast)
        self.root.after(1000, self.auto_start_web_dashboard)
        self.root.after(1200, self._start_web_dashboard_monitor)
        self.root.after(1500, self.check_for_updates_on_startup)
        self.root.after(1800, self.check_web_server_status)
        self.root.after(2000, self.check_tools)  # Check tool status once at startup

        if AUTO_INSTALL:
            self.root.after(3000, self.check_and_auto_install_tools)

    def check_and_auto_install_tools(self):
        """Check for missing essential tools and auto-install them."""
        log.info("Checking for essential tools to auto-install...")
        essential_tools = ["claude", "gemini"]

        for tool_id in essential_tools:
            if not check_tool_installed(tool_id):
                log.info(f"Auto-installing missing essential tool: {tool_id}")
                spec = self._tool_specs.get(tool_id)
                if spec:
                    self.install_tool(tool_id, tool_id.capitalize(), spec)
                else:
                    log.warning(f"No spec found for essential tool: {tool_id}")

    def setup_styles(self):
        """Configure ttk styles for modern M3-inspired look."""
        style = ttk.Style()

        # Apply theme
        theme_set = False
        if HAS_SV_TTK:
            try:
                sv_ttk.set_theme("dark")
                theme_set = True
            except Exception as e:
                log.debug(f"sv_ttk theme failed: {e}")

        if not theme_set:
            try:
                style.theme_use("clam")
            except Exception as e:
                log.debug(f"Failed to set 'clam' theme: {e}")

        # Configure colors for themed widgets
        try:
            # Root and basic elements
            style.configure(
                ".", background=COLORS["bg_surface"], foreground=COLORS["text"]
            )
            style.configure("TFrame", background=COLORS["bg_surface"])
            style.configure("Card.TFrame", background=COLORS["bg_card"], relief="flat")

            # Form elements (fallbacks for when sv_ttk is not present)
            style.configure(
                "TCheckbutton",
                background=COLORS["bg_surface"],
                foreground=COLORS["text"],
            )
            style.configure(
                "TRadiobutton",
                background=COLORS["bg_surface"],
                foreground=COLORS["text"],
            )
            style.configure(
                "TEntry",
                fieldbackground=COLORS["bg_elevated"],
                foreground=COLORS["text"],
                insertcolor=COLORS["text"],
            )
            style.configure(
                "TCombobox",
                fieldbackground=COLORS["bg_elevated"],
                foreground=COLORS["text"],
            )
            style.configure(
                "Horizontal.TProgressbar",
                background=COLORS["accent"],
                troughcolor=COLORS["bg_elevated"],
                borderwidth=0,
            )
        except Exception as e:
            log.debug(f"Failed to configure base styles: {e}")

        # Label styles
        try:
            style.configure(
                "TLabel",
                background=COLORS["bg_surface"],
                foreground=COLORS["text"],
                font=("Segoe UI", 10),
            )
            style.configure(
                "Header.TLabel",
                font=("Segoe UI", 20, "bold"),
                foreground=COLORS["text"],
            )
            style.configure(
                "Subheader.TLabel",
                foreground=COLORS["text_secondary"],
                font=("Segoe UI", 10),
            )
            style.configure(
                "Caption.TLabel",
                foreground=COLORS["text_muted"],
                font=("Segoe UI", 8, "bold"),
            )
            style.configure(
                "Badge.TLabel",
                background=COLORS["accent_container"],
                foreground=COLORS["accent_light"],
                font=("Segoe UI", 8, "bold"),
            )
        except Exception as e:
            log.debug(f"Failed to configure label styles: {e}")

        # Button styles - Accent (Terracotta)
        try:
            style.configure(
                "Accent.TButton",
                font=("Segoe UI", 10, "bold"),
                background=COLORS["accent"],
                foreground="white",
            )
            style.map(
                "Accent.TButton",
                background=[
                    ("active", COLORS["accent_hover"]),
                    ("pressed", COLORS["accent"]),
                ],
                foreground=[("active", "white")],
            )
        except Exception as e:
            log.debug(f"Failed to configure accent button styles: {e}")

        # Button styles - Secondary
        try:
            style.configure(
                "Secondary.TButton",
                font=("Segoe UI", 9),
                background=COLORS["bg_elevated"],
            )
            style.map(
                "Secondary.TButton",
                background=[
                    ("active", COLORS["accent"]),
                    ("pressed", COLORS["bg_elevated"]),
                ],
                foreground=[("active", "white")],
            )
            log.info("[OK] Secondary button styles configured")
        except Exception as e:
            log.error(f"\u2717 Failed to configure secondary button styles: {e}")

        # Scrollbar styling
        try:
            style.configure(
                "Vertical.TScrollbar",
                background=COLORS["bg_elevated"],
                troughcolor=COLORS["bg_surface"],
                bordercolor=COLORS["bg_surface"],
                arrowcolor=COLORS["text_dim"],
                relief="flat",
                borderwidth=0,
                width=12,
            )
            style.map(
                "Vertical.TScrollbar",
                background=[
                    ("active", COLORS["accent"]),
                    ("pressed", COLORS["accent_hover"]),
                ],
            )
            log.info("[OK] Scrollbar styles configured")
        except Exception as e:
            log.error(f"\u2717 Failed to configure scrollbar styles: {e}")

        log.info("[OK] setup_styles() completed")

    def create_widgets(self):
        """Create compact single-column layout using tk for reliable dark theming."""
        log.info("Creating main container...")
        # Main container with scrolling support
        try:
            self.canvas = tk.Canvas(
                self.root, bg=COLORS["bg_surface"], highlightthickness=0, borderwidth=0
            )
            self.scrollbar = ttk.Scrollbar(
                self.root, orient="vertical", command=self.canvas.yview
            )
            # self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)  # Hidden - scrollbar disabled
            self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            self.canvas.configure(yscrollcommand=self.scrollbar.set)

            # Inner padding frame (the scrollable content)
            left_inner = tk.Frame(
                self.canvas,
                bg=COLORS["bg_surface"],
                padx=20,
                pady=16,
                relief="flat",
                borderwidth=0,
                highlightthickness=0,
            )
            self.canvas_window = self.canvas.create_window(
                (0, 0), window=left_inner, anchor="nw", width=self.window_width
            )

            # Update scrollregion when content changes
            def _on_frame_configure(event):
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))

            left_inner.bind("<Configure>", _on_frame_configure)

            # Update window width when canvas is resized
            def _on_canvas_configure(event):
                self.canvas.itemconfig(self.canvas_window, width=event.width)

            self.canvas.bind("<Configure>", _on_canvas_configure)

            # Enable mouse wheel scrolling
            def _on_mousewheel(event):
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

            self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

            log.info("[OK] Scrollable main container created")
        except Exception as e:
            log.error(f"\u2717 Failed to create scrollable container: {e}", exc_info=True)
            raise

        # Header with title
        header = tk.Frame(
            left_inner,
            bg=COLORS["bg_surface"],
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        header.pack(fill=tk.X, pady=(0, 12))

        title_label = tk.Label(
            header,
            text="ShadowBridge",
            bg=COLORS["bg_surface"],
            fg=COLORS["text"],
            font=("Segoe UI", 20, "bold"),
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        title_label.pack(side=tk.LEFT)

        # Version badge
        version_label = tk.Label(
            header,
            text=f"v{APP_VERSION}",
            bg=COLORS["accent_container"],
            fg=COLORS["accent_light"],
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=2,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        version_label.pack(side=tk.LEFT, padx=(10, 0))

        # Host info row
        device_row = tk.Frame(
            left_inner,
            bg=COLORS["bg_surface"],
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        device_row.pack(fill=tk.X, pady=(0, 15))

        # Host stack
        host_stack = tk.Frame(
            device_row,
            bg=COLORS["bg_surface"],
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        host_stack.pack(side=tk.LEFT)
        host_label = tk.Label(
            host_stack,
            text="HOST",
            bg=COLORS["bg_surface"],
            fg=COLORS["text_muted"],
            font=("Segoe UI", 8, "bold"),
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        host_label.pack(anchor="w")
        hostname_label = tk.Label(
            host_stack,
            text=f"{socket.gethostname()}",
            bg=COLORS["bg_surface"],
            fg=COLORS["text_secondary"],
            font=("Segoe UI", 10),
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        hostname_label.pack(anchor="w")

        # Status stack
        status_stack = tk.Frame(
            device_row,
            bg=COLORS["bg_surface"],
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        status_stack.pack(side=tk.RIGHT)

        self.connected_device_label = tk.Label(
            status_stack,
            text="No device",
            bg=COLORS["bg_surface"],
            fg=COLORS["text_muted"],
            font=("Segoe UI", 8),
            anchor="e",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        self.connected_device_label.pack(anchor="e")

        self.broadcast_status_label = tk.Label(
            status_stack,
            text="â— Broadcasting",
            bg=COLORS["bg_surface"],
            fg=COLORS["success"],
            font=("Segoe UI", 8, "bold"),
            anchor="e",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        self.broadcast_status_label.pack(anchor="e")

        # QR Code card
        qr_card = tk.Frame(
            left_inner,
            bg=COLORS["bg_card"],
            padx=20,
            pady=15,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        qr_card.pack(fill=tk.X, pady=(0, 12))

        tk.Label(
            qr_card,
            text="QUICK CONNECT",
            bg=COLORS["bg_card"],
            fg=COLORS["text_muted"],
            font=("Segoe UI", 8, "bold"),
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        ).pack(anchor="w", pady=(0, 10))

        self.qr_label = tk.Label(
            qr_card,
            bg=COLORS["bg_card"],
            text="Loading...",
            fg=COLORS["text_dim"],
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        self.qr_label.pack(pady=(0, 15))
        self._last_qr_data = None

        # Connection info
        info_row = tk.Frame(
            qr_card,
            bg=COLORS["bg_elevated"],
            padx=15,
            pady=10,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        info_row.pack(fill=tk.X)

        self.ip_label = tk.Label(
            info_row,
            text=f"{get_local_ip()}",
            bg=COLORS["bg_elevated"],
            fg=COLORS["accent_light"],
            font=("Consolas", 12, "bold"),
            justify=tk.CENTER,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        self.ip_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        self.ssh_label = tk.Label(
            info_row,
            text="SSH: ...",
            bg=COLORS["bg_elevated"],
            fg=COLORS["text_dim"],
            font=("Consolas", 10),
            justify=tk.CENTER,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        self.ssh_label.pack(side=tk.TOP, fill=tk.X)

        # Tools section
        tools_card = tk.Frame(
            left_inner,
            bg=COLORS["bg_card"],
            padx=20,
            pady=15,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        tools_card.pack(fill=tk.X, pady=(0, 10))

        tools_label = tk.Label(
            tools_card,
            text="TOOLS",
            bg=COLORS["bg_card"],
            fg=COLORS["text_muted"],
            font=("Segoe UI", 8, "bold"),
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        tools_label.pack(anchor="w", pady=(0, 12))

        # Tool buttons grid
        self.tool_buttons = {}
        self.tool_status = {}
        self._tool_specs = {}
        self._tool_install_procs = {}
        self._tool_install_lock = threading.Lock()

        tools = [
            (
                "Claude Code",
                "claude",
                {
                    "type": "npm",
                    "commands": ["npm install -g @anthropic-ai/claude-code"],
                    "uninstall_commands": [
                        "npm uninstall -g @anthropic-ai/claude-code"
                    ],
                },
            ),
            (
                "Codex",
                "codex",
                {
                    "type": "npm",
                    "commands": ["npm install -g @openai/codex"],
                    "uninstall_commands": ["npm uninstall -g @openai/codex"],
                },
            ),
            (
                "Gemini",
                "gemini",
                {
                    "type": "npm",
                    "commands": ["npm install -g @google/gemini-cli"],
                    "uninstall_commands": ["npm uninstall -g @google/gemini-cli"],
                },
            ),
            (
                "OpenCode",
                "opencode",
                {
                    "type": "npm",
                    "commands": ["npm install -g @opencode-ai/plugin"],
                    "uninstall_commands": ["npm uninstall -g @opencode-ai/plugin"],
                },
            ),
            (
                "Aider",
                "aider",
                {
                    "type": "pip",
                    "commands": [
                        "py -m pip install -U aider-chat",
                        "python -m pip install -U aider-chat",
                        "pip install -U aider-chat",
                    ],
                    "uninstall_commands": [
                        "py -m pip uninstall -y aider-chat",
                        "python -m pip uninstall -y aider-chat",
                        "pip uninstall -y aider-chat",
                    ],
                    "fallback_url": "https://aider.chat/",
                },
            ),
            (
                "Ollama",
                "ollama",
                {
                    "type": "winget" if IS_WINDOWS else "brew",
                    "commands": [
                        "winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements"
                    ]
                    if IS_WINDOWS
                    else ["brew install ollama"],
                    "uninstall_commands": [
                        "winget uninstall -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements"
                    ]
                    if IS_WINDOWS
                    else ["brew uninstall ollama"],
                    "fallback_url": "https://ollama.com/download/windows"
                    if IS_WINDOWS
                    else "https://ollama.com/download/mac",
                },
            ),
            (
                "Tailscale",
                "tailscale",
                {
                    "type": "winget" if IS_WINDOWS else "brew",
                    "commands": [
                        "winget install -e --id Tailscale.Tailscale --accept-source-agreements --accept-package-agreements"
                    ]
                    if IS_WINDOWS
                    else ["brew install tailscale"],
                    "uninstall_commands": [
                        "winget uninstall -e --id Tailscale.Tailscale --accept-source-agreements --accept-package-agreements"
                    ]
                    if IS_WINDOWS
                    else ["brew uninstall tailscale"],
                    "fallback_url": "https://tailscale.com/download",
                },
            ),
        ]

        # Function to create modern tk buttons
        def create_modern_button(
            container, text, command, width=12, primary=False, pady=5, font_size=9
        ):
            bg_color = COLORS["accent"] if primary else COLORS["bg_elevated"]
            fg_color = "white" if primary else COLORS["text"]

            btn = tk.Button(
                container,
                text=text,
                command=command,
                width=width,
                bg=bg_color,
                fg=fg_color,
                font=("Segoe UI", font_size, "bold")
                if primary
                else ("Segoe UI", font_size),
                relief="flat",
                bd=0,
                padx=10,
                pady=pady,
                cursor="hand2",
                activebackground=COLORS["accent_hover"],
                activeforeground="white",
                highlightthickness=0,
                overrelief="flat",
            )

            def on_enter(e):
                btn.configure(bg=COLORS["accent_hover"], fg="white")

            def on_leave(e):
                btn.configure(bg=bg_color, fg=fg_color)

            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
            return btn

        for name, tool_id, spec in tools:
            self._tool_specs[tool_id] = spec
            row = tk.Frame(
                tools_card,
                bg=COLORS["bg_card"],
                relief="flat",
                borderwidth=0,
                highlightthickness=0,
            )
            row.pack(fill=tk.X, pady=4)

            status_frame = tk.Frame(
                row,
                bg=COLORS["bg_card"],
                relief="flat",
                borderwidth=0,
                highlightthickness=0,
            )
            status_frame.pack(side=tk.LEFT, padx=(0, 12))

            status_canvas = tk.Canvas(
                status_frame,
                width=12,
                height=12,
                bg=COLORS["bg_card"],
                highlightthickness=0,
                borderwidth=0,
                relief="flat",
                takefocus=0,
            )
            status_canvas.pack(pady=6)
            status_canvas.create_oval(
                2, 2, 10, 10, fill=COLORS["text_muted"], outline="", tags="dot"
            )
            self.tool_status[tool_id] = status_canvas

            tool_name_label = tk.Label(
                row,
                text=name,
                bg=COLORS["bg_card"],
                fg=COLORS["text_secondary"],
                font=("Segoe UI", 10),
                width=12,
                anchor="w",
                relief="flat",
                borderwidth=0,
                highlightthickness=0,
            )
            tool_name_label.pack(side=tk.LEFT)

            # Create install button with hover effects
            btn = tk.Button(
                row,
                text="Install",
                width=12,
                bg=COLORS["success"],
                fg="white",
                font=("Segoe UI", 9, "bold"),
                relief="flat",
                bd=0,
                padx=10,
                pady=5,
                cursor="hand2",
                activebackground="#a5d6a7",
                activeforeground="white",
                highlightthickness=0,
                overrelief="flat",
                command=lambda s=spec, n=name, t=tool_id: self.install_tool(t, n, s),
            )
            btn.pack(side=tk.RIGHT)

            def on_enter_tool(e, b=btn):
                if b.cget("state") != tk.DISABLED:
                    b.configure(bg=COLORS["accent_hover"])

            def on_leave_tool(e, b=btn, original_bg=COLORS["success"]):
                if b.cget("state") != tk.DISABLED:
                    current_text = b.cget("text")
                    if current_text == "Install":
                        b.configure(bg=COLORS["success"])
                    elif current_text == "Uninstall":
                        b.configure(
                            bg=COLORS["error"] if "error" in COLORS else "#ff6b6b"
                        )
                    elif current_text == "Stop":
                        b.configure(bg=COLORS["warning"])
                    else:
                        b.configure(bg=original_bg)

            btn.bind("<Enter>", on_enter_tool)
            btn.bind("<Leave>", on_leave_tool)

            self.tool_buttons[tool_id] = btn

        # Bottom section
        bottom = tk.Frame(
            left_inner,
            bg=COLORS["bg_surface"],
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        bottom.pack(fill=tk.X, pady=(20, 0))

        self.web_dashboard_btn = create_modern_button(
            bottom,
            "Open Web Dashboard",
            self.launch_web_dashboard,
            width=30,
            primary=True,
            pady=12,
            font_size=11,
        )
        self.web_dashboard_btn.pack(fill=tk.X, pady=(0, 8))

        exit_btn = create_modern_button(
            bottom, "Exit", self.force_exit, width=30, pady=12, font_size=11
        )
        exit_btn.pack(fill=tk.X, pady=(0, 15))

        # Options row
        opts = tk.Frame(
            bottom,
            bg=COLORS["bg_surface"],
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        opts.pack(fill=tk.X)

        if IS_WINDOWS:
            self.startup_var = tk.BooleanVar(value=is_startup_enabled())
            startup_cb = tk.Checkbutton(
                opts,
                text="Auto-start",
                variable=self.startup_var,
                command=self.toggle_startup,
                bg=COLORS["bg_surface"],
                fg=COLORS["text_secondary"],
                selectcolor=COLORS[
                    "bg_card"
                ],  # Box background when selected (fixes white box)
                activebackground=COLORS["bg_surface"],
                activeforeground=COLORS["text"],
                font=("Segoe UI", 9),
                highlightthickness=0,
                bd=0,
                relief="flat",
                overrelief="flat",
                borderwidth=0,
                takefocus=0,
            )
            startup_cb.pack(side=tk.LEFT)

            self.auto_update_var = tk.BooleanVar(value=is_auto_update_enabled())
            auto_update_cb = tk.Checkbutton(
                opts,
                text="Updates",
                variable=self.auto_update_var,
                command=self.toggle_auto_update,
                bg=COLORS["bg_surface"],
                fg=COLORS["text_secondary"],
                selectcolor=COLORS["bg_card"],  # Fixes white box
                activebackground=COLORS["bg_surface"],
                activeforeground=COLORS["text"],
                font=("Segoe UI", 9),
                highlightthickness=0,
                bd=0,
                relief="flat",
                overrelief="flat",
                borderwidth=0,
                takefocus=0,
            )
            auto_update_cb.pack(side=tk.LEFT, padx=(12, 0))

        # Revoke Keys link
        revoke_link = tk.Label(
            opts,
            text="Revoke Keys",
            bg=COLORS["bg_surface"],
            fg=COLORS["error"],
            font=("Segoe UI", 9, "underline"),
            cursor="hand2",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        revoke_link.pack(side=tk.RIGHT, padx=(0, 12))
        revoke_link.bind("<Button-1>", lambda e: self.revoke_all_keys_ui())

        help_link = tk.Label(
            opts,
            text="Help",
            bg=COLORS["bg_surface"],
            fg=COLORS["accent_light"],
            font=("Segoe UI", 9, "underline"),
            cursor="hand2",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        help_link.pack(side=tk.RIGHT)
        help_link.bind("<Button-1>", lambda e: self.show_ssh_help())

        # Start animations
        self._pulse_alpha = 0
        self._pulse_dir = 1
        self._animate_pulse()

    def _animate_pulse(self):
        """Animate status dots with a subtle color pulse."""
        self._pulse_alpha += 0.08 * self._pulse_dir
        if self._pulse_alpha >= 1.0:
            self._pulse_alpha = 1.0
            self._pulse_dir = -1
        elif self._pulse_alpha <= 0.2:
            self._pulse_alpha = 0.2
            self._pulse_dir = 1

        # Interpolate color between text_muted and accent
        def interpolate_color(c1, c2, factor):
            def to_rgb(hex_color):
                hex_color = hex_color.lstrip("#")
                return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

            rgb1 = to_rgb(c1)
            rgb2 = to_rgb(c2)
            res = tuple(int(rgb1[i] + (rgb2[i] - rgb1[i]) * factor) for i in range(3))
            return f"#{res[0]:02x}{res[1]:02x}{res[2]:02x}"

        pulse_color = interpolate_color(
            COLORS["text_muted"], COLORS["accent"], self._pulse_alpha
        )

        # Apply to all status dots
        for canvas in self.tool_status.values():
            canvas.itemconfig("dot", fill=pulse_color)

        if hasattr(self, "tailscale_status"):
            self.tailscale_status.itemconfig("dot", fill=pulse_color)

        self.root.after(250, self._animate_pulse)

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
            tooltip.configure(bg=COLORS["bg_elevated"])
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{x}+{y}")
            label = tk.Label(
                tooltip,
                text=text,
                bg=COLORS["bg_elevated"],
                fg=COLORS["text_secondary"],
                font=("Segoe UI", 8),
                padx=6,
                pady=3,
                relief="flat",
                borderwidth=0,
                highlightthickness=0,
            )
            label.pack()

        def hide_tooltip(event):
            nonlocal tooltip
            if tooltip:
                tooltip.destroy()
                tooltip = None

        widget.bind(
            "<Enter>",
            lambda e: widget.after(
                500,
                lambda: show_tooltip(e)
                if widget.winfo_containing(
                    widget.winfo_pointerx(), widget.winfo_pointery()
                )
                == widget
                else None,
            ),
            add="+",
        )
        widget.bind("<Leave>", hide_tooltip, add="+")

    def update_status(self):
        """Update status indicators."""
        try:
            # Early return optimization: skip if nothing changed since last update
            # This reduces unnecessary UI updates and CPU usage
            if hasattr(self, "_last_update_state"):
                all_ips = get_all_ips()
                local_ips = all_ips.get("local", [])
                tailscale_ip = get_tailscale_ip()
                ssh_ok = check_ssh_running(
                    self.ssh_port if isinstance(self.ssh_port, int) else 22
                )
                now = time.time()
                active_devices = [
                    d
                    for d in (self.devices or {}).values()
                    if now - float(d.get("last_seen", 0) or 0) < 300
                ]
                ip_display = "\n".join([f"LAN: {local_ips[0]}"] if local_ips else [])
                if tailscale_ip:
                    ip_display += f"\nTailscale: {tailscale_ip}"
                elif globals().get("_cached_tailscale_ip") == "Detecting...":
                    ip_display += "\nTailscale: Detecting..."
                else:
                    ip_display = "Scanning..."

                # Check if state matches last update
                current_state = {
                    "ssh_ok": ssh_ok,
                    "ip_display": ip_display,
                    "active_device_count": len(active_devices),
                    "is_broadcasting": self.is_broadcasting,
                }
                if current_state == self._last_update_state:
                    # Nothing changed, skip this update cycle
                    self.root.after(10000, self.update_status)
                    return

            # Handle port as string or integer safely
            try:
                test_port = int(self.ssh_port) if str(self.ssh_port).isdigit() else 22
            except (ValueError, TypeError):
                test_port = 22

            ssh_ok = check_ssh_running(test_port)

            # Update IP label - show LAN and Tailscale IPs
            all_ips = get_all_ips()
            local_ips = all_ips.get("local", [])
            tailscale_ip = get_tailscale_ip()

            display_parts = []
            if local_ips:
                display_parts.append(f"LAN: {local_ips[0]}")

            if tailscale_ip:
                display_parts.append(f"Tailscale: {tailscale_ip}")
            elif globals().get("_cached_tailscale_ip") == "Detecting...":
                display_parts.append("Tailscale: Detecting...")

            current_display = (
                "\n".join(display_parts) if display_parts else "Scanning..."
            )

            if hasattr(self, "ip_label"):
                ip_text = self.ip_label.cget("text")
                if ip_text != current_display:
                    self.ip_label.configure(text=current_display)

            # Update SSH label with port info
            if ssh_ok:
                self.ssh_label.configure(
                    text=f"SSH: {self.ssh_port} [OK]", fg=COLORS["success"]
                )
            else:
                self.ssh_label.configure(
                    text=f"SSH: {self.ssh_port} [X]", fg=COLORS["error"]
                )

            # Merge active devices from Signaling Service (UDP)
            if hasattr(self, "signaling_service") and self.signaling_service:
                try:
                    # Use copy to avoid thread contention
                    signaling_devices = self.signaling_service.active_devices.copy()
                    for dev_id, info in signaling_devices.items():
                        # Update if we have new info or it's a new device
                        current_dev = self.devices.get(dev_id, {})
                        last_seen_gui = float(current_dev.get("last_seen", 0) or 0)
                        last_seen_udp = float(info.get("last_seen", 0) or 0)

                        if last_seen_udp > last_seen_gui:
                            # Extract name from status json if available
                            name = dev_id
                            status = info.get("status", {})
                            if status and isinstance(status, dict):
                                name = status.get(
                                    "device_name", status.get("name", dev_id)
                                )

                            current_dev.update(
                                {
                                    "id": dev_id,
                                    "name": name,
                                    "ip": info.get("ip"),
                                    "last_seen": last_seen_udp,
                                }
                            )
                            self.devices[dev_id] = current_dev
                except Exception as e:
                    log.debug(f"Error merging signaling devices: {e}")

            now = time.time()
            active_devices = [
                d
                for d in (self.devices or {}).values()
                if now - float(d.get("last_seen", 0) or 0) < 300
            ]

            # Update broadcast status label based on connection state
            if hasattr(self, "broadcast_status_label"):
                if active_devices:
                    self.broadcast_status_label.configure(
                        text=f"â— {len(active_devices)} Connected", fg=COLORS["success"]
                    )
                elif self.is_broadcasting and ssh_ok:
                    self.broadcast_status_label.configure(
                        text="â— Broadcasting", fg=COLORS["success"]
                    )
                elif ssh_ok:
                    self.broadcast_status_label.configure(
                        text="â— Ready", fg=COLORS["warning"]
                    )
                else:
                    self.broadcast_status_label.configure(
                        text="â— No SSH", fg=COLORS["error"]
                    )

            # Update connected devices display - only refresh periodically to reduce CPU usage
            self.refresh_connected_devices_ui()

            # Save state for early return optimization in next update cycle
            all_ips = get_all_ips()
            local_ips = all_ips.get("local", [])
            tailscale_ip = get_tailscale_ip()
            ip_display = "\n".join([f"LAN: {local_ips[0]}"] if local_ips else [])
            if tailscale_ip:
                ip_display += f"\nTailscale: {tailscale_ip}"
            elif globals().get("_cached_tailscale_ip") == "Detecting...":
                ip_display += "\nTailscale: Detecting..."
            else:
                ip_display = "Scanning..."

            self._last_update_state = {
                "ssh_ok": ssh_ok,
                "ip_display": ip_display,
                "active_device_count": len(active_devices),
                "is_broadcasting": self.is_broadcasting,
            }
        except Exception as e:
            log.error(f"Error in update_status loop: {e}", exc_info=True)

        # Reduced from 30000ms to 10000ms to improve responsiveness
        self.root.after(10000, self.update_status)

    def start_data_receiver(self):
        """Start TCP server to receive project and notes data from Android app."""
        try:
            self.data_receiver = DataReceiver(
                on_data_received=self.on_projects_received,
                on_device_connected=self.on_device_connected,
                on_notes_received=self.on_notes_received,
                on_sessions_received=self.on_sessions_received,
                on_cards_received=self.on_cards_received,
                on_collections_received=self.on_collections_received,
                on_key_approval_needed=self.on_key_approval_needed,
            )
            self.data_receiver.start()
            log.info(f"Data receiver started on port {DATA_PORT}")

            # Register with watchdog for auto-restart
            if self.watchdog:
                self.watchdog.register(
                    "DataReceiver",
                    self.data_receiver,
                    self.start_data_receiver_thread,
                    max_restarts=5,
                    restart_window=300,
                )
        except Exception as e:
            log.error(f"Failed to start data receiver: {e}")

        # Also start companion relay for Claude Code plugin
        self.start_companion_relay()

        # Start Cloud Sync
        self.start_cloud_sync()

    def start_data_receiver_thread(self):
        """Internal method to create new DataReceiver thread for watchdog restart."""
        try:
            self.data_receiver = DataReceiver(
                on_data_received=self.on_projects_received,
                on_device_connected=self.on_device_connected,
                on_notes_received=self.on_notes_received,
                on_sessions_received=self.on_sessions_received,
                on_cards_received=self.on_cards_received,
                on_collections_received=self.on_collections_received,
                on_key_approval_needed=self.on_key_approval_needed,
            )
            self.data_receiver.start()
            log.info(f"DataReceiver restarted by watchdog")
            return self.data_receiver
        except Exception as e:
            log.error(f"Failed to restart DataReceiver: {e}")
            return None

    def start_companion_relay(self):
        """Start TCP relay server for Claude Code Companion feature."""
        try:
            # Pass approved devices from DataReceiver for security checks
            approved = getattr(self.data_receiver, '_approved_devices', None) if self.data_receiver else None
            self.companion_relay = CompanionRelayServer(
                on_status_change=self.on_companion_status_change,
                approved_devices=approved,
            )
            self.companion_relay.start()
            log.info(f"Companion relay started on port {COMPANION_PORT}")

            # Wire agent orchestrator to relay for command forwarding
            try:
                from web.services.agent_orchestrator import get_orchestrator
                get_orchestrator().set_relay(self.companion_relay)
            except Exception:
                pass

            # Register with watchdog for auto-restart
            if self.watchdog:
                self.watchdog.register(
                    "CompanionRelay",
                    self.companion_relay,
                    self.start_companion_relay_thread,
                    max_restarts=5,
                    restart_window=300,
                )
        except Exception as e:
            log.error(f"Failed to start companion relay: {e}")

    def start_companion_relay_thread(self):
        """Internal method to create new CompanionRelay thread for watchdog restart."""
        try:
            approved = getattr(self.data_receiver, '_approved_devices', None) if self.data_receiver else None
            self.companion_relay = CompanionRelayServer(
                on_status_change=self.on_companion_status_change,
                approved_devices=approved,
            )
            self.companion_relay.start()
            log.info(f"CompanionRelay restarted by watchdog")
            return self.companion_relay
        except Exception as e:
            log.error(f"Failed to restart CompanionRelay: {e}")
            return None

    def start_cloud_sync(self):
        """Start background cloud sync service."""
        try:
            self.cloud_sync = CloudSyncService()
            self.cloud_sync.start()
            log.info("Cloud sync service started")
        except Exception as e:
            log.error(f"Failed to start cloud sync: {e}")

    def on_companion_status_change(self, status, port):
        """Called when companion relay status changes."""
        log.info(f"Companion relay: {status} on port {port}")

    def on_projects_received(self, device_id, projects):
        """Called when projects data is received from Android app."""
        if not isinstance(projects, list):
            return
        self.projects = projects
        # Schedule UI update on main thread
        self.root.after(0, self.refresh_projects_ui)

    def check_tools(self):
        """Check which tools are installed."""
        tool_names = {
            "claude": "Claude Code",
            "codex": "Codex",
            "gemini": "Gemini",
            "opencode": "OpenCode",
            "aider": "Aider",
            "ollama": "Ollama",
            "tailscale": "Tailscale",
        }

        def check():
            # Check tools
            for tool_id, canvas in self.tool_status.items():
                with self._tool_install_lock:
                    # Don't stomp the UI if an install/uninstall is currently in progress.
                    if tool_id in self._tool_install_procs:
                        continue
                installed = check_tool_installed(tool_id)
                color = COLORS["success"] if installed else COLORS["error"]
                name = tool_names.get(tool_id, tool_id)

                def update_ui(c=canvas, col=color, t=tool_id, n=name, inst=installed):
                    self._update_tool_dot(c, col)
                    # Button state: Install vs Uninstall
                    btn = self.tool_buttons[t]
                    if inst:
                        btn.configure(
                            text="Uninstall",
                            bg=COLORS["error"],
                            fg="white",
                            activebackground="#f87171",
                            command=lambda tid=t, nm=n: self.uninstall_tool(tid, nm),
                        )
                    else:
                        btn.configure(
                            text="Install",
                            bg=COLORS["success"],
                            fg="white",
                            activebackground="#a5d6a7",
                            command=lambda tid=t, nm=n: self.install_tool(
                                tid, nm, self._tool_specs.get(tid, {})
                            ),
                        )

                self.root.after(0, update_ui)

        threading.Thread(target=check, daemon=True).start()
        # NOTE: Removed auto-polling - only check at startup and after install/uninstall clicks

    def _update_tool_dot(self, canvas, color):
        """Update dot color without deleting the tag for pulse animation."""
        canvas.itemconfig("dot", fill=color)

    def get_connection_info(self):
        """Get connection info with multi-IP fallback strategy.

        Returns ALL available IPs in priority order:
        1. Private LAN IPs (192.168.x.x, 10.x.x.x, 172.x.x.x) - FASTEST
        2. Tailscale IPs (100.x.x.x) - Cross-network VPN
        3. Other public/VPN IPs - Last resort

        Android app should try IPs in order until one succeeds.
        """
        tailscale_ip = get_tailscale_ip()
        local_ip = get_local_ip()
        hostname_local = get_hostname_local()
        all_ips = get_all_ips()

        # Get the dynamic encryption salt for note sync (Phase 5.3 requirement)
        try:
            from web.services.data_service import DYNAMIC_SALT

            salt_b64 = base64.b64encode(DYNAMIC_SALT).decode("ascii")
        except Exception:
            salt_b64 = None

        # Build comprehensive list of hosts with priority ordering
        # CRITICAL: Android app must try these IN ORDER
        hosts_to_try = []

        # Priority 1: Private LAN IPs (FASTEST - same Wi-Fi network)
        for ip in all_ips.get("local", []):
            if ip and not ip.startswith("127."):
                hosts_to_try.append(
                    {
                        "host": ip,
                        "type": "local",
                        "stable": False,
                        "label": "LAN",
                        "priority": 1,
                    }
                )

        # Priority 2: Tailscale IPs (Cross-network VPN - works anywhere but slower)
        for ip in all_ips.get("tailscale", []):
            hosts_to_try.append(
                {
                    "host": ip,
                    "type": "tailscale",
                    "stable": True,
                    "label": "Tailscale",
                    "priority": 2,
                }
            )

        # Priority 3: VPN IPs (OpenVPN, WireGuard, etc.)
        for ip in all_ips.get("vpn", []):
            hosts_to_try.append(
                {
                    "host": ip,
                    "type": "vpn",
                    "stable": False,
                    "label": "VPN",
                    "priority": 3,
                }
            )

        # Priority 4: Other IPs (Public or unknown)
        for ip in all_ips.get("other", []):
            hosts_to_try.append(
                {
                    "host": ip,
                    "type": "other",
                    "stable": False,
                    "label": "Other",
                    "priority": 4,
                }
            )

        # Primary host (best available - PREFER LAN, Tailscale as fallback)
        primary_host = local_ip or tailscale_ip
        mode = "local" if local_ip else "tailscale"

        # Log multi-IP advertising for debugging
        if hosts_to_try:
            ip_summary = ", ".join(
                [f"{h['label']}:{h['host']}" for h in hosts_to_try[:4]]
            )
            log.debug(
                f"Advertising {len(hosts_to_try)} IPs (primary: {primary_host}): {ip_summary}"
            )

        return {
            "type": "shadowai_connect",
            "version": 7,  # Bumped version for WebSocket backend support
            "mode": mode,
            "host": primary_host,  # Primary for QR display (LAN preferred)
            "port": self.ssh_port,
            "data_port": DATA_PORT,  # Port for key exchange and data sync
            "username": get_username(),
            "hostname": socket.gethostname(),
            "hostname_local": hostname_local,
            "hosts": hosts_to_try,  # Multi-IP array with priority ordering
            "local_ip": local_ip,
            "tailscale_ip": tailscale_ip,
            "encryption_salt": salt_b64,
            "websocket_enabled": True,  # NEW: WebSocket backend available
            "websocket_port": WEB_PORT,  # Same as web dashboard (6767)
            "websocket_path": "/ws/cli",  # WebSocket endpoint path
            "timestamp": int(time.time()),
        }

    def update_qr_code(self):
        """Generate and display QR code - only if changed."""
        if not HAS_QRCODE or not HAS_PIL:
            if hasattr(self, "qr_label"):
                self.qr_label.configure(text="QR unavailable")
            return

        try:
            info = self.get_connection_info()
            info_json = json.dumps(info)
            encoded = base64.urlsafe_b64encode(info_json.encode()).decode()
            qr_data = f"shadowai://connect?data={encoded}"

            # Check if QR data changed
            if hasattr(self, "_last_qr_data") and self._last_qr_data == qr_data:
                return  # No change, skip regeneration
            self._last_qr_data = qr_data

            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,  # Increased box size
                border=2,
            )
            qr.add_data(qr_data)
            qr.make(fit=True)

            # Create PIL image with WHITE modules on dark card background
            img = qr.make_image(fill_color="white", back_color=COLORS["bg_card"])
            # Resize to be much larger
            img = img.resize(
                (380, 380),
                Image.Resampling.LANCZOS
                if hasattr(Image, "Resampling")
                else Image.LANCZOS,
            )

            # Convert to PhotoImage for Tkinter
            self.qr_photo = ImageTk.PhotoImage(img)
            if hasattr(self, "qr_label"):
                self.qr_label.configure(image=self.qr_photo, text="")
        except Exception as e:
            log.error(f"Error updating QR code: {e}", exc_info=True)
            if hasattr(self, "qr_label"):
                self.qr_label.configure(text="QR error")

    def auto_start_broadcast(self):
        """Auto-start broadcasting if SSH is running."""
        if check_ssh_running(self.ssh_port) or platform.system() == "Darwin":
            self.start_broadcast()

    def _get_tools_dir(self):
        if getattr(sys, "frozen", False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))

    def _get_web_server_command(self, open_browser: bool):
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "--web-server"]
        else:
            cmd = [sys.executable, os.path.abspath(__file__), "--web-server"]
        if not open_browser:
            cmd.append("--no-browser")
        return cmd

    def _launch_web_server_process(self, tools_dir: str):
        """Start the web dashboard subprocess and keep it detached."""
        cmd = self._get_web_server_command(open_browser=False)
        web_log = open(WEB_LOG_FILE, "a", encoding="utf-8", errors="replace")

        if IS_WINDOWS:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0
            creation_flags = (
                subprocess.CREATE_NO_WINDOW
                | subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP
            )
            process = subprocess.Popen(
                cmd,
                cwd=tools_dir,
                startupinfo=startupinfo,
                creationflags=creation_flags,
                stdin=subprocess.DEVNULL,
                stdout=web_log,
                stderr=web_log,
                close_fds=True,
            )
        else:
            process = subprocess.Popen(
                cmd, cwd=tools_dir, stdout=web_log, stderr=web_log
            )

        web_log.close()
        self.web_process = process
        log.info(f"Web dashboard started (PID: {process.pid})")
        return process

    def _is_web_server_process_alive(self) -> bool:
        proc = getattr(self, "web_process", None)
        return proc is not None and proc.poll() is None

    def _is_web_server_thread_alive(self) -> bool:
        thread = getattr(self, "web_server_thread", None)
        return thread is not None and thread.is_alive()

    def _start_web_server_thread(self, open_browser: bool, show_errors: bool) -> bool:
        if self._is_web_server_process_alive():
            return True
        if self._is_web_server_thread_alive():
            return True

        # Launch web server as subprocess for proper shutdown
        try:
            python_exe = sys.executable
            script_path = os.path.abspath(__file__)
            args = [python_exe, script_path, "--web-server"]
            if not open_browser:
                args.append("--no-browser")

            process = subprocess.Popen(
                args,
                creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0,
            )
            self.web_process = process
            log.info(f"Web dashboard started as subprocess (PID: {process.pid})")

            # Open browser in separate thread if requested
            if open_browser:
                url = "http://127.0.0.1:6767"
                monitor_thread = threading.Thread(
                    target=lambda: self._monitor_web_server_and_open(url, show_errors),
                    daemon=True,
                )
                monitor_thread.start()

            return True
        except Exception as exc:
            log.error(f"Failed to start web server subprocess: {exc}")
            if show_errors:
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Web Dashboard Error",
                        f"Failed to launch web dashboard:\n{exc}",
                    ),
                )
            return False

    def _monitor_web_server_and_open(self, url: str, show_errors: bool):
        """Poll the web server until available and optionally open browser."""

        def monitor():
            time.sleep(1.5)
            attempts = 0
            while attempts < 4:
                try:
                    import urllib.request

                    urllib.request.urlopen(url, timeout=3)
                    webbrowser.open(url)
                    return
                except Exception as exc:
                    log.warning(f"Waiting for web dashboard: {exc}")
                    attempts += 1
                    time.sleep(1.5)

            log.error("Web dashboard failed to start")
            if show_errors:
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Web Dashboard Error",
                        "Web server failed to start. Check if port 6767 is available.",
                    ),
                )

        threading.Thread(target=monitor, daemon=True).start()

    def _start_web_dashboard_process(
        self, open_browser: bool, show_errors: bool
    ) -> bool:
        """Ensure the web dashboard server is running; optionally open browser."""
        web_port = 6767
        web_url = f"http://127.0.0.1:{web_port}"

        try:
            import urllib.request
            import urllib.error

            urllib.request.urlopen(web_url, timeout=1)
            if open_browser:
                webbrowser.open(web_url)
            return True
        except Exception:
            pass

        if self._is_web_server_process_alive():
            if open_browser:
                self._monitor_web_server_and_open(web_url, show_errors)
            return True

        try:
            tools_dir = self._get_tools_dir()
            if not getattr(sys, "frozen", False):
                web_folder = os.path.join(tools_dir, "web")
                if not os.path.exists(web_folder):
                    msg = (
                        f"Web dashboard folder not found.\n\nExpected at:\n{web_folder}"
                    )
                    log.warning(msg)
                    if show_errors:
                        messagebox.showerror("Web Dashboard Missing", msg)
                    return False

            self._launch_web_server_process(tools_dir)
            if open_browser:
                self._monitor_web_server_and_open(web_url, show_errors)
            return True
        except Exception as exc:
            log.error(f"Failed to start web dashboard: {exc}")
            if show_errors:
                messagebox.showerror("Error", f"Failed to launch web dashboard:\n{exc}")
            if self._start_web_server_thread(
                open_browser=open_browser, show_errors=show_errors
            ):
                if open_browser:
                    self._monitor_web_server_and_open(web_url, show_errors)
                return True
            return False

    def auto_start_web_dashboard(self):
        """Auto-start web dashboard server (without opening browser)."""
        self._auto_web_dashboard_attempts = 0
        self._attempt_auto_start_web_dashboard()

    def _attempt_auto_start_web_dashboard(self):
        if self._start_web_dashboard_process(open_browser=False, show_errors=False):
            return

        self._auto_web_dashboard_attempts += 1
        if self._auto_web_dashboard_attempts < 3:
            self.root.after(5000, self._attempt_auto_start_web_dashboard)
        else:
            log.warning("Auto-start web dashboard exceeded retry limit.")

    def _start_web_dashboard_monitor(self):
        if (
            getattr(self, "_web_dashboard_monitor_thread", None)
            and self._web_dashboard_monitor_thread.is_alive()
        ):
            return

        self._web_dashboard_monitor_stop_event.clear()

        def monitor_loop():
            log.info("Web dashboard monitor loop started")
            while not self._web_dashboard_monitor_stop_event.is_set():
                try:
                    # Increased from 4s to 6s to reduce CPU load while still providing reasonable response time
                    time.sleep(6)

                    if self._web_dashboard_monitor_stop_event.is_set():
                        break

                    if not self._is_web_server_process_alive():
                        # Only log restart attempts once every 60 seconds to reduce log spam
                        if not hasattr(self, "_last_restart_log_time"):
                            self._last_restart_log_time = 0
                        now = time.time()
                        if now - self._last_restart_log_time > 60:
                            log.debug(
                                "Web dashboard monitor detected downtime, restarting"
                            )
                            self._last_restart_log_time = now
                        self._ensure_web_dashboard_running()
                except Exception as exc:
                    # Only log errors once every 60 seconds to reduce log spam
                    if not hasattr(self, "_last_monitor_error_time"):
                        self._last_monitor_error_time = 0
                    now = time.time()
                    if now - self._last_monitor_error_time > 60:
                        log.debug(f"Web dashboard monitor error: {exc}")
                        self._last_monitor_error_time = now
                    time.sleep(1)  # Prevent tight loop on error

        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        self._web_dashboard_monitor_thread = thread

    def _stop_web_dashboard_monitor(self):
        if not getattr(self, "_web_dashboard_monitor_stop_event", None):
            return

        self._web_dashboard_monitor_stop_event.set()
        thread = getattr(self, "_web_dashboard_monitor_thread", None)
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        self._web_dashboard_monitor_thread = None

    def _ensure_web_dashboard_running(self):
        if self._web_dashboard_start_lock.locked():
            return False

        with self._web_dashboard_start_lock:
            if self._is_web_server_process_alive():
                return True
            return self._start_web_dashboard_process(
                open_browser=False, show_errors=False
            )

    def toggle_broadcast(self):
        """Toggle broadcasting."""
        if self.is_broadcasting:
            self.stop_broadcast()
        else:
            self.start_broadcast()

    def launch_web_dashboard(self):
        """Launch the web dashboard in browser."""
        self._start_web_dashboard_process(open_browser=True, show_errors=True)

    def check_web_server_status(self):
        """Check if web dashboard server is running and update status indicator - non-blocking."""
        if not hasattr(self, "web_dashboard_btn"):
            self.root.after(3000, self.check_web_server_status)
            return

        def perform_check():
            try:
                import urllib.request

                # Create a request to check local server
                req = urllib.request.Request("http://127.0.0.1:6767")
                with urllib.request.urlopen(req, timeout=1) as response:
                    pass
                status = True
            except Exception:
                status = False

            def update_ui(is_running=status):
                if is_running:
                    # Server is running
                    if hasattr(self, "web_status_dot") and self.web_status_dot:
                        self.web_status_dot.itemconfig("dot", fill=COLORS["success"])
                    self.web_dashboard_btn.configure(text="Open Web Dashboard")
                else:
                    # Server not running
                    if hasattr(self, "web_status_dot") and self.web_status_dot:
                        self.web_status_dot.itemconfig("dot", fill=COLORS["text_muted"])
                    self.web_dashboard_btn.configure(text="Launch Web Dashboard")
                    log.debug("Web dashboard down; triggering monitor to restart it")
                    self._ensure_web_dashboard_running()

                # Schedule next check - increased from 5000ms to 15000ms to reduce CPU load
                self.root.after(15000, self.check_web_server_status)

            self.root.after(0, update_ui)

        threading.Thread(target=perform_check, daemon=True).start()

    def start_broadcast(self):
        """Start discovery server."""

        def do_start():
            try:
                # Ensure firewall rule exists for discovery (blocking subprocess)
                setup_firewall_rule()

                self.discovery_server = DiscoveryServer(
                    self.get_connection_info(), app_instance=self
                )
                self.discovery_server.start()
                self.is_broadcasting = True

                # Update status label on main thread
                self.root.after(
                    0,
                    lambda: self.broadcast_status_label.configure(
                        text="â— Broadcasting", fg=COLORS["success"]
                    )
                    if hasattr(self, "broadcast_status_label")
                    else None,
                )

                log.info("Discovery server started successfully")

                # Register with watchdog for auto-restart
                if self.watchdog:
                    self.watchdog.register(
                        "DiscoveryServer",
                        self.discovery_server,
                        self.start_broadcast_thread,
                        max_restarts=5,
                        restart_window=300,
                    )
            except Exception as e:
                log.error(f"Failed to start discovery server: {e}")
                self.is_broadcasting = False

                self.root.after(
                    0,
                    lambda: self.broadcast_status_label.configure(
                        text="â— Broadcast Failed", fg=COLORS["error"]
                    )
                    if hasattr(self, "broadcast_status_label")
                    else None,
                )

                # Show error to user on main thread
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Network Discovery Failed",
                        f"Could not start network discovery service.\n\n"
                        f"Error: {e}\n\n"
                        f"Check that port {DISCOVERY_PORT} is not in use by another application.",
                    ),
                )

        threading.Thread(target=do_start, daemon=True).start()

    def start_broadcast_thread(self):
        """Internal method to create new DiscoveryServer thread for watchdog restart."""
        try:
            setup_firewall_rule()
            self.discovery_server = DiscoveryServer(
                self.get_connection_info(), app_instance=self
            )
            self.discovery_server.start()
            self.is_broadcasting = True
            if hasattr(self, "broadcast_status_label"):
                self.broadcast_status_label.configure(
                    text="â— Broadcasting", fg=COLORS["success"]
                )
            log.info(f"DiscoveryServer restarted by watchdog")
            return self.discovery_server
        except Exception as e:
            log.error(f"Failed to restart DiscoveryServer: {e}")
            self.is_broadcasting = False
            if hasattr(self, "broadcast_status_label"):
                self.broadcast_status_label.configure(
                    text="â— Broadcast Failed", fg=COLORS["error"]
                )
            return None

    def stop_broadcast(self):
        """Stop discovery server."""
        if self.discovery_server:
            self.discovery_server.stop()
            self.discovery_server = None
        self.is_broadcasting = False
        # Update status label
        if hasattr(self, "broadcast_status_label"):
            self.broadcast_status_label.configure(
                text="â— Stopped", fg=COLORS["text_muted"]
            )

    def install_tool(self, tool_id, name, spec):
        """Install a tool (npm/pip/winget)."""
        with self._tool_install_lock:
            if tool_id in self._tool_install_procs:
                return
        btn = self.tool_buttons[tool_id]
        original_text = btn.cget("text")
        btn.configure(text="Stop", state="normal", bg=COLORS["warning"])

        def do_install():
            install_type = (spec or {}).get("type", "npm")
            commands = (spec or {}).get("commands", [])
            fallback_url = (spec or {}).get("fallback_url", None)

            if install_type == "npm":
                if not check_npm_installed():
                    self.root.after(
                        0,
                        lambda: self._prompt_nodejs_install(
                            tool_id, name, original_text
                        ),
                    )
                    return
            elif install_type == "pip":
                if not check_python_pip_installed():
                    self.root.after(
                        0,
                        lambda: self._prompt_python_install(
                            tool_id, name, original_text
                        ),
                    )
                    return
            elif install_type == "winget":
                if not check_winget_installed():
                    self.root.after(
                        0,
                        lambda: self._prompt_direct_download(
                            tool_id, name, original_text, fallback_url
                        ),
                    )
                    return

            try:
                flags = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
                timeout_s = 900 if install_type == "winget" else 300

                last_error = "Unknown error"
                for command in commands:
                    # Use shlex.split and shell=False for security
                    args = shlex.split(command)
                    proc = subprocess.Popen(
                        args,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        creationflags=flags,
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
                self.root.after(
                    0,
                    lambda: self._mark_failed(tool_id, name, "Installation timed out"),
                )
            except Exception as e:
                self.root.after(
                    0, lambda: self._mark_failed(tool_id, name, str(e)[:100])
                )

        threading.Thread(target=do_install, daemon=True).start()
        btn.configure(
            command=lambda tid=tool_id, nm=name: self.stop_tool_install(tid, nm)
        )

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
                    # SECURITY: Use array form instead of shell=True
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        creationflags=subprocess.CREATE_NO_WINDOW,
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
        self._update_tool_dot(self.tool_status[tool_id], COLORS["success"])
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
        btn.configure(text="Install", state="normal", bg=COLORS["accent"])
        btn.configure(
            command=lambda tid=tool_id, nm=name: self.install_tool(
                tid, nm, self._tool_specs.get(tid, {})
            )
        )
        if error and error != "Cancelled":
            messagebox.showerror(
                "Install Failed", f"Failed to install {name}:\n\n{error}"
            )
        self.check_tools()

    def _prompt_nodejs_install(self, tool_id, name, original_text):
        """Prompt user to install Node.js."""
        btn = self.tool_buttons[tool_id]
        btn.configure(text=original_text, state="normal", bg=COLORS["accent"])

        if messagebox.askyesno(
            "Node.js Required",
            f"npm is required to install {name}.\n\n"
            "Would you like to download Node.js now?\n\n"
            "(After installing, restart ShadowBridge and try again)",
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

        btn.configure(text="Uninstalling...", state="disabled", bg=COLORS["warning"])

        def do_uninstall():
            if not commands:
                self.root.after(
                    0,
                    lambda: self._mark_failed(
                        tool_id, name, "No uninstall command configured"
                    ),
                )
                return

            if install_type == "npm" and not check_npm_installed():
                self.root.after(
                    0, lambda: self._prompt_nodejs_install(tool_id, name, "Uninstall")
                )
                return
            if install_type == "pip" and not check_python_pip_installed():
                self.root.after(
                    0, lambda: self._prompt_python_install(tool_id, name, "Uninstall")
                )
                return
            if install_type == "winget" and not check_winget_installed():
                self.root.after(
                    0,
                    lambda: self._prompt_direct_download(
                        tool_id, name, "Uninstall", fallback_url
                    ),
                )
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
                    creationflags=flags,
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

            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Uninstall Failed", f"Failed to uninstall {name}:\n\n{last_error}"
                ),
            )
            self.root.after(0, lambda: self.check_tools())

        threading.Thread(target=do_uninstall, daemon=True).start()

    def _prompt_python_install(self, tool_id, name, original_text):
        """Prompt user to install Python (for pip installs)."""
        btn = self.tool_buttons[tool_id]
        btn.configure(text=original_text, state="normal", bg=COLORS["accent"])

        if messagebox.askyesno(
            "Python Required",
            f"Python + pip is required to install {name}.\n\n"
            "Would you like to download Python now?\n\n"
            "(After installing, restart ShadowBridge and try again)",
        ):
            webbrowser.open("https://www.python.org/downloads/")

    def _prompt_direct_download(self, tool_id, name, original_text, url):
        """Prompt user to download an installer directly."""
        btn = self.tool_buttons[tool_id]
        btn.configure(text=original_text, state="normal", bg=COLORS["accent"])

        if not url:
            messagebox.showinfo("Install", f"Please install {name} manually.")
            return

        if messagebox.askyesno(
            "Download Required",
            f"{name} couldn't be installed automatically.\n\nOpen the download page?",
        ):
            webbrowser.open(url)

    def toggle_startup(self):
        """Toggle Windows startup."""
        enabled = self.startup_var.get()
        if not set_startup_enabled(enabled):
            self.startup_var.set(not enabled)

    def toggle_auto_update(self):
        """Toggle auto-update setting."""
        enabled = self.auto_update_var.get()
        set_auto_update_enabled(enabled)

    def revoke_all_keys_ui(self):
        """Prompt user and revoke all SSH keys."""
        if not messagebox.askyesno(
            "Revoke All SSH Keys?",
            "This will remove all Shadow-added keys from your PC's authorized_keys files and clear all pairing data.\n\n"
            "Your Android app will need to re-pair before it can connect again.\n\n"
            "Proceed?",
            icon="warning",
        ):
            return

        if hasattr(self, "data_receiver") and self.data_receiver:
            result = self.data_receiver.revoke_all_keys()
            if result.get("success"):
                messagebox.showinfo("Keys Revoked", result.get("message"))
                self.refresh_connected_devices_ui()
                self.update_status()
            else:
                messagebox.showerror("Error", result.get("message"))

    def check_for_updates_on_startup(self):
        """Check for updates on startup (runs in background thread)."""
        if not is_auto_update_enabled():
            return

        def check_update_thread():
            try:
                has_update, latest_version, download_url = check_for_updates()
                if has_update and download_url:
                    # Schedule UI update on main thread
                    self.root.after(
                        0, lambda: self.prompt_for_update(latest_version, download_url)
                    )
            except Exception as e:
                log.warning(f"Update check failed: {e}")

        thread = threading.Thread(target=check_update_thread, daemon=True)
        thread.start()

    def prompt_for_update(self, latest_version, download_url):
        """Show update prompt dialog."""
        import tkinter.messagebox as messagebox

        result = messagebox.askyesno(
            "Update Available",
            f"ShadowBridge v{latest_version} is available.\n"
            f"You have v{APP_VERSION}.\n\n"
            "Would you like to download and install the update?\n"
            "(App will restart after update)",
            parent=self.root,
        )

        if result:
            self.download_and_apply_update(download_url)

    def download_and_apply_update(self, download_url):
        """Download and apply update with progress dialog."""
        import tkinter.messagebox as messagebox

        # Create progress dialog
        progress_win = tk.Toplevel(self.root)
        progress_win.title("Updating...")
        progress_win.geometry("300x100")
        progress_win.configure(bg=COLORS["bg_dark"])
        progress_win.transient(self.root)
        progress_win.grab_set()

        label = tk.Label(
            progress_win,
            text="Downloading update...",
            bg=COLORS["bg_dark"],
            fg=COLORS["text"],
            font=("Segoe UI", 10),
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        label.pack(pady=20)

        progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(
            progress_win, variable=progress_var, maximum=100, length=250
        )
        progress_bar.pack(pady=10)

        def update_progress(fraction):
            self.root.after(0, lambda: progress_var.set(fraction * 100))

        def download_thread():
            try:
                update_path = download_update(download_url, update_progress)
                if update_path:
                    self.root.after(
                        0, lambda: label.configure(text="Applying update...")
                    )
                    if apply_update(update_path):
                        self.root.after(0, progress_win.destroy)
                        self.root.after(100, self.root.quit)
                    else:
                        self.root.after(0, progress_win.destroy)
                        self.root.after(
                            0,
                            lambda: messagebox.showerror(
                                "Update Failed",
                                "Failed to apply update. Please try again.",
                                parent=self.root,
                            ),
                        )
                else:
                    self.root.after(0, progress_win.destroy)
                    self.root.after(
                        0,
                        lambda: messagebox.showerror(
                            "Download Failed",
                            "Failed to download update. Please try again.",
                            parent=self.root,
                        ),
                    )
            except Exception as e:
                log.error(f"Update failed: {e}")
                self.root.after(0, progress_win.destroy)
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Update Failed", f"Update failed: {e}", parent=self.root
                    ),
                )

        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()

    def show_ssh_help(self):
        """Show SSH help dialog."""
        help_win = tk.Toplevel(self.root)
        help_win.title("SSH Setup")
        help_win.geometry("400x300")
        help_win.configure(bg=COLORS["bg_dark"])
        help_win.transient(self.root)

        text = """Windows SSH Setup:

1. Open Settings -> Apps -> Optional Features
2. Click "Add a feature"
3. Find and install "OpenSSH Server"
4. Open Services (Win+R -> services.msc)
5. Find "OpenSSH SSH Server"
6. Set to "Automatic" and click "Start"

Or run in PowerShell (Admin):
  Start-Service sshd
  Set-Service -Name sshd -StartupType Automatic"""

        tk.Label(
            help_win,
            text=text,
            bg=COLORS["bg_dark"],
            fg=COLORS["text"],
            font=("Consolas", 9),
            justify="left",
            padx=20,
            pady=20,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        ).pack(fill=tk.BOTH, expand=True)

    def _save_window_state(self):
        """Save window position and size to file."""
        try:
            geometry = self.root.geometry()
            # Parse geometry string like "1180x820+100+50"
            import re

            match = re.match(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", geometry)
            if match:
                state = {
                    "width": int(match.group(1)),
                    "height": int(match.group(2)),
                    "x": int(match.group(3)),
                    "y": int(match.group(4)),
                }
                with open(WINDOW_STATE_FILE, "w", encoding="utf-8") as f:
                    json.dump(state, f)
        except Exception as e:
            log.debug(f"Failed to save window state: {e}")

    def _load_window_state(self):
        """Load and apply saved window position and size."""
        try:
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()

            # Get DPI scale factor for padding calculation
            try:
                dpi = self.root.winfo_fpixels("1i")
                scale = dpi / 96.0  # 96 is standard DPI
            except Exception:
                scale = 1.0

            # Calculate 12dp padding in pixels (accounting for DPI scaling)
            padding = int(12 * scale)

            if os.path.exists(WINDOW_STATE_FILE):
                with open(WINDOW_STATE_FILE, "r", encoding="utf-8") as f:
                    state = json.load(f)
                w = state.get("width", self.window_width)
                h = state.get("height", self.window_height)
                x = state.get("x", 100)
                y = state.get("y", 100)
                # Validate position is on screen
                if x < 0 or x > screen_w - 100:
                    x = screen_w - w - padding
                if y < 0 or y > screen_h - 100:
                    y = screen_h - h - padding
                self.root.geometry(f"{w}x{h}+{x}+{y}")
                return True
            else:
                # First run: position at bottom right with 12dp padding
                x = screen_w - self.window_width - padding
                y = screen_h - self.window_height - padding
                self.root.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")
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
        if icon_image and icon_image.mode == "RGBA":
            background = Image.new("RGB", icon_image.size, COLORS["bg_dark"])
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
            pystray.MenuItem("Exit", on_exit),
        )

        self.tray_icon = pystray.Icon(APP_NAME, icon_image, APP_NAME, menu)
        self.tray_thread = threading.Thread(target=self.tray_icon.run)
        self.tray_thread.start()

    def show_window(self):
        """Make window visible and bring to front."""
        if hasattr(self, "tray_icon") and self.tray_icon:
            try:
                self.tray_icon.stop()
            except Exception:
                pass
            self.tray_icon = None

        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def show_ping_effects(self):
        """
        Show celebration effects when connection test succeeds.
        Brings window to focus, plays sound, and shows confetti.
        Called via IPC when Android app sends --ping command.
        """
        log.info("Showing ping celebration effects")
        try:
            # Bring window to foreground
            self.show_window()

            # Play sound and show confetti in GUI
            if HAS_EFFECTS:
                play_ping_sound()
                confetti = ConfettiEffect(self.root)
                confetti.start(duration_ms=4000)

            # Also trigger celebration in web dashboard
            try:
                from web.routes.websocket import broadcast_celebrate

                broadcast_celebrate("Connected from ShadowAI!")
            except Exception as web_err:
                log.debug(f"Web celebration not available: {web_err}")
        except Exception as e:
            log.error(f"Failed to show ping effects: {e}")

    def _restore_from_tray(self):
        """Restore from tray."""
        self.tray_icon = None
        self.root.deiconify()
        # Small delay to allow window to paint before lifting
        self.root.after(50, lambda: self.root.lift())
        self.root.after(50, lambda: self.root.focus_force())

    def quit_app(self):
        """Clean exit."""
        if self.tray_icon:
            self.tray_icon.stop()
        if self.discovery_server:
            self.discovery_server.stop()
        if self.data_receiver:
            self.data_receiver.stop()
        if self.companion_relay:
            self.companion_relay.stop()
        if self.signaling_service:
            self.signaling_service.stop()
        if self.sentinel:
            self.sentinel.stop()
        if self.web_process:
            self.web_process.terminate()
        if self.root:
            self.root.destroy()
        sys.exit(0)

    def force_exit(self):
        """Force exit of application immediately."""
        self._stop_web_dashboard_monitor()
        self.stop_broadcast()
        if hasattr(self, "signaling_service") and self.signaling_service:
            self.signaling_service.stop()
        if self.data_receiver:
            try:
                self.data_receiver.stop()
            except (RuntimeError, OSError):
                pass  # Already stopped
        if self.companion_relay:
            try:
                self.companion_relay.stop()
            except (RuntimeError, OSError):
                pass  # Already stopped
        if self.web_process:
            try:
                self.web_process.terminate()
                log.info("Web dashboard process terminated")
            except (OSError, ProcessLookupError):
                pass  # Process already terminated
        if self.tray_icon:
            try:
                self.tray_icon.stop()
            except (RuntimeError, OSError):
                pass  # Tray already stopped
        self.root.destroy()
        os._exit(0)  # Force exit

    def on_minimize(self, event=None):
        """Handle minimize - go to system tray instead of taskbar."""
        if event and self.root.state() == "iconic":
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
            key=lambda d: d.get("last_seen") or 0,
            reverse=True,
        )
        return [
            d for d in devices_sorted if now - float(d.get("last_seen", 0) or 0) < 300
        ]

    def refresh_connected_devices_ui(self):
        """Refresh connected device label in header - only if changed."""
        if not hasattr(self, "connected_device_label"):
            return

        active = self._get_active_devices()

        if not active:
            current_text = self.connected_device_label.cget("text")
            if current_text != "No device":
                self.connected_device_label.configure(
                    text="No device", foreground=COLORS["text_dim"]
                )
            return

        # Show first connected device name
        device = active[0]
        name = device.get("name") or device.get("id") or "Unknown"
        if len(active) > 1:
            new_text = f"â— {name} +{len(active) - 1}"
        else:
            new_text = f"â— {name}"

        current_text = self.connected_device_label.cget("text")
        if current_text != new_text:
            self.connected_device_label.configure(
                text=new_text, foreground=COLORS["success"]
            )

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
                menu.add_command(
                    label=label, command=lambda l=label: self.project_device_var.set(l)
                )

            add_option("All devices", "__ALL__")

            devices_sorted = sorted(
                (self.devices or {}).values(),
                key=lambda d: d.get("last_seen") or 0,
                reverse=True,
            )
            for device in devices_sorted:
                did = device.get("id")
                if not did:
                    continue
                name = device.get("name") or did
                ip = device.get("ip") or ""
                label = f"{name}{f' ({ip})' if ip else ''}"
                add_option(label, did)

            # Keep selection stable by device_id
            desired = (
                self.selected_device_id
                if getattr(self, "selected_device_id", "__ALL__")
                else "__ALL__"
            )
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
        self.selected_device_id = getattr(self, "_device_label_to_id", {}).get(
            label, "__ALL__"
        )
        self.refresh_projects_ui()

    def on_device_connected(self, device_id, device_name, ip):
        """Called when a device connects and sends data."""
        device = (self.devices or {}).get(device_id, {})
        device.update(
            {
                "id": device_id,
                "name": device_name,
                "ip": ip,
                "last_seen": time.time(),
                "projects": device.get("projects", []),
            }
        )
        self.devices[device_id] = device

        def refresh_ui():
            self.refresh_connected_devices_ui()
            self.refresh_project_device_menu()
            self.update_status()

        self.root.after(0, refresh_ui)

    def on_projects_received(self, device_id, projects):
        """Called when projects data is received from Android app."""
        if not isinstance(projects, list):
            return

        device = (self.devices or {}).get(device_id, {})
        device.update(
            {
                "id": device_id,
                "name": device.get("name", device_id),
                "ip": device.get("ip", None),
                "last_seen": time.time(),
                "projects": projects,
            }
        )
        self.devices[device_id] = device

        def refresh_ui():
            self.refresh_connected_devices_ui()
            self.refresh_project_device_menu()
            self.refresh_projects_ui()
            self.update_status()

        self.root.after(0, refresh_ui)

        # Notify web dashboard for real-time sync
        self._notify_web_dashboard_projects_updated(device_id)

    def _notify_web_dashboard_projects_updated(self, device_id):
        """Notify web dashboard that projects have been updated."""

        def do_notify():
            try:
                import urllib.request
                import json as json_module

                data = json_module.dumps({"device_id": device_id}).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:6767/api/projects/sync",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
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
        self.root.after(
            0, lambda: self._show_key_approval_dialog(device_id, device_name, ip)
        )

    def _show_key_approval_dialog(self, device_id, device_name, ip):
        """Show a dialog asking user to approve SSH key installation."""
        try:
            import tkinter.messagebox as messagebox

            # Ensure main window is visible so dialog has a parent and user sees it
            if self.root.state() == "withdrawn" or self.root.state() == "iconic":
                self.show_window()

            self.root.lift()
            self.root.focus_force()

            # Create approval dialog
            result = messagebox.askyesno(
                "SSH Key Approval Required",
                f"A new device wants to install an SSH key:\n\n"
                f"Device: {device_name}\n"
                f"IP Address: {ip}\n\n"
                f"This will allow the device to connect via SSH.\n\n"
                f"Do you want to approve this device?",
                icon="warning",
                parent=self.root,
            )

            if result:
                # User approved
                if hasattr(self, "data_receiver") and self.data_receiver:
                    response = self.data_receiver.approve_device(device_id)
                    if response.get("success"):
                        # Celebration! Play success sound and flash
                        self._show_success_celebration(device_name)
                    else:
                        messagebox.showwarning(
                            "Note",
                            f"Device approved but key installation pending.\n{response.get('message', '')}",
                            parent=self.root,
                        )
            else:
                # User rejected
                if hasattr(self, "data_receiver") and self.data_receiver:
                    self.data_receiver.reject_device(device_id)
                messagebox.showinfo(
                    "Rejected",
                    f"SSH key request from {device_name} was rejected.",
                    parent=self.root,
                )

        except Exception as e:
            log.error(f"Error showing key approval dialog: {e}")

    def _show_success_celebration(self, device_name):
        """Show an exciting success celebration with sound and visual feedback."""
        try:
            # Play success beep (Windows only)
            if IS_WINDOWS:
                # Play a cheerful ascending tone sequence
                def play_success_sound():
                    try:
                        winsound.Beep(523, 100)  # C5
                        winsound.Beep(659, 100)  # E5
                        winsound.Beep(784, 150)  # G5
                        winsound.Beep(1047, 200)  # C6 - triumphant finish
                    except Exception:
                        pass

                threading.Thread(target=play_success_sound, daemon=True).start()

            # Create celebration overlay window
            celebration = tk.Toplevel(self.root)
            celebration.overrideredirect(True)  # No window decorations
            celebration.attributes("-topmost", True)

            # Center on main window
            main_x = self.root.winfo_x()
            main_y = self.root.winfo_y()
            main_w = self.root.winfo_width()
            main_h = self.root.winfo_height()

            cel_w, cel_h = 300, 200
            cel_x = main_x + (main_w - cel_w) // 2
            cel_y = main_y + (main_h - cel_h) // 2
            celebration.geometry(f"{cel_w}x{cel_h}+{cel_x}+{cel_y}")

            # Green success background
            celebration.configure(bg="#10B981")

            # Big checkmark
            checkmark_label = tk.Label(
                celebration,
                text="\u2713",
                font=("Segoe UI", 72, "bold"),
                fg="white",
                bg="#10B981",
            )
            checkmark_label.pack(expand=True)

            # Device name
            name_label = tk.Label(
                celebration,
                text=f"{device_name} Connected!",
                font=("Segoe UI", 14, "bold"),
                fg="white",
                bg="#10B981",
            )
            name_label.pack(pady=(0, 20))

            # Animate: pulse the checkmark size
            def pulse_animation(step=0):
                if step < 6:
                    sizes = [72, 80, 72, 80, 72, 72]
                    checkmark_label.configure(font=("Segoe UI", sizes[step], "bold"))
                    celebration.after(100, lambda: pulse_animation(step + 1))
                else:
                    # Fade out and close
                    celebration.after(800, celebration.destroy)

            pulse_animation()

        except Exception as e:
            log.error(f"Error showing success celebration: {e}")
            # Fallback to simple messagebox
            messagebox.showinfo(
                "\u2713 Connected!",
                f"SSH key installed for {device_name}",
                parent=self.root,
            )

    def refresh_projects_ui(self):
        """Refresh the projects list UI."""
        if not hasattr(self, "projects_container"):
            return

        for widget in self.projects_container.winfo_children():
            widget.destroy()

        projects = []
        if getattr(self, "selected_device_id", "__ALL__") == "__ALL__":
            for device in (self.devices or {}).values():
                for project in device.get("projects", []) or []:
                    if isinstance(project, dict):
                        p = dict(project)
                        p["_device_name"] = device.get("name") or device.get("id")
                        projects.append(p)
        else:
            device = (self.devices or {}).get(self.selected_device_id) or {}
            projects = (
                device.get("projects", [])
                if isinstance(device.get("projects", []), list)
                else []
            )

        if hasattr(self, "projects_count_label"):
            self.projects_count_label.configure(text=f"{len(projects)}")

        if not projects:
            tk.Label(
                self.projects_container,
                text="No project folders synced yet.\nProjects will appear here after your phone syncs.",
                bg=COLORS["bg_card"],
                fg=COLORS["text_dim"],
                font=("Segoe UI", 9),
                justify="center",
                anchor="center",
                relief="flat",
                borderwidth=0,
                highlightthickness=0,
            ).pack(fill=tk.BOTH, expand=True, pady=20)
            return

        # Sort by last edited (most recent first)
        def get_project_timestamp(p):
            return (
                p.get("updated_at")
                or p.get("updatedAt")
                or p.get("lastAccessed")
                or p.get("last_accessed")
                or 0
            )

        projects.sort(key=get_project_timestamp, reverse=True)

        for project in projects:
            self._add_project_row(project)

    def _add_project_row(self, project):
        """Add a single project row to the UI. Double-click to open."""
        name = project.get("name", "Unnamed")
        path = (
            project.get("path")
            or project.get("workingDirectory")
            or project.get("working_directory")
            or project.get("dir")
            or ""
        )
        device_name = project.get("_device_name")

        # Check if this is an openable PC path
        is_openable = (
            is_pc_path(path) and path and (os.path.isdir(path) or os.path.exists(path))
        )

        row = tk.Frame(
            self.projects_container,
            bg=COLORS["bg_input"],
            padx=8,
            pady=6,
            cursor="hand2" if is_openable else "arrow",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        row.pack(fill=tk.X, pady=2)

        info_frame = tk.Frame(
            row,
            bg=COLORS["bg_input"],
            cursor="hand2" if is_openable else "arrow",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        name_label = tk.Label(
            info_frame,
            text=name,
            bg=COLORS["bg_input"],
            fg=COLORS["text"],
            font=("Segoe UI", 9, "bold"),
            anchor="w",
            cursor="hand2" if is_openable else "arrow",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        name_label.pack(anchor="w")

        if device_name and getattr(self, "selected_device_id", "__ALL__") == "__ALL__":
            tk.Label(
                info_frame,
                text=str(device_name),
                bg=COLORS["bg_input"],
                fg=COLORS["text_dim"],
                font=("Segoe UI", 8),
                anchor="w",
                cursor="hand2" if is_openable else "arrow",
                relief="flat",
                borderwidth=0,
                highlightthickness=0,
            ).pack(anchor="w")

        if is_pc_path(path) and path:
            path_display = path
            path_color = COLORS["text_dim"] if is_openable else COLORS["text_muted"]
        else:
            path_display = f"Phone: {path}" if path else "Phone only"
            path_color = COLORS["warning"]

        path_label = tk.Label(
            info_frame,
            text=path_display,
            bg=COLORS["bg_input"],
            fg=path_color,
            font=("Consolas", 8),
            anchor="w",
            wraplength=250,
            cursor="hand2" if is_openable else "arrow",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        path_label.pack(anchor="w")

        # Double-click to open folder
        if is_openable:

            def on_double_click(event, p=path):
                open_folder(p)

            for widget in [row, info_frame, name_label, path_label]:
                widget.bind("<Double-Button-1>", on_double_click)

            # Hover effect
            def on_enter(e):
                row.configure(bg=COLORS["bg_elevated"])
                info_frame.configure(bg=COLORS["bg_elevated"])
                for child in info_frame.winfo_children():
                    child.configure(bg=COLORS["bg_elevated"])

            def on_leave(e):
                row.configure(bg=COLORS["bg_input"])
                info_frame.configure(bg=COLORS["bg_input"])
                for child in info_frame.winfo_children():
                    child.configure(bg=COLORS["bg_input"])

            row.bind("<Enter>", on_enter)
            row.bind("<Leave>", on_leave)

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
        device.update(
            {
                "id": device_id,
                "name": device.get("name", device_id),
                "ip": device.get("ip", None),
                "last_seen": time.time(),
                "notes": notes,
            }
        )
        self.notes_devices[device_id] = device

        def refresh_ui():
            self.refresh_notes_device_menu()
            self.refresh_notes_ui()

        self.root.after(0, refresh_ui)

        # Notify web dashboard for real-time sync
        self._notify_web_dashboard_notes_updated(device_id)

    def _notify_web_dashboard_notes_updated(self, device_id):
        """Notify web dashboard that notes have been updated (triggers WebSocket broadcast)."""

        def do_notify():
            try:
                import urllib.request
                import json as json_module

                data = json_module.dumps({"device_id": device_id}).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:6767/api/notes/sync",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=2)
                log.debug(f"Notified web dashboard of notes update for {device_id}")
            except Exception as e:
                log.debug(f"Could not notify web dashboard: {e}")

        # Run in background thread to not block
        threading.Thread(target=do_notify, daemon=True).start()

    def on_sessions_received(self, device_id, sessions):
        """Called when sessions data is received from Android app."""
        if not isinstance(sessions, list):
            return

        device = (self.sessions_devices or {}).get(device_id, {})
        device.update(
            {
                "id": device_id,
                "name": device.get("name", device_id),
                "ip": device.get("ip", None),
                "last_seen": time.time(),
                "sessions": sessions,
            }
        )
        self.sessions_devices[device_id] = device

        # Notify web dashboard for real-time sync
        self._notify_web_dashboard_sessions_updated(device_id)

    def on_cards_received(self, device_id, cards):
        """Called when cards data is received from Android app."""
        if not isinstance(cards, list):
            return

        device = (self.cards_devices or {}).get(device_id, {})
        device.update(
            {
                "id": device_id,
                "name": device.get("name", device_id),
                "ip": device.get("ip", None),
                "last_seen": time.time(),
                "cards": cards,
            }
        )
        self.cards_devices[device_id] = device

        # Notify web dashboard for real-time sync
        self._notify_web_dashboard_cards_updated(device_id)

    def on_collections_received(self, device_id, collections):
        """Called when collections data is received from Android app."""
        if not isinstance(collections, list):
            return

        device = (self.collections_devices or {}).get(device_id, {})
        device.update(
            {
                "id": device_id,
                "name": device.get("name", device_id),
                "ip": device.get("ip", None),
                "last_seen": time.time(),
                "collections": collections,
            }
        )
        self.collections_devices[device_id] = device

        # Notify web dashboard for real-time sync
        self._notify_web_dashboard_collections_updated(device_id)

    def _notify_web_dashboard_sessions_updated(self, device_id):
        """Notify web dashboard that sessions have been updated."""

        def do_notify():
            try:
                import urllib.request
                import json as json_module

                data = json_module.dumps({"device_id": device_id}).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:6767/api/sessions/sync",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=2)
                log.debug(f"Notified web dashboard of sessions update for {device_id}")
            except Exception as e:
                log.debug(f"Could not notify web dashboard: {e}")

        threading.Thread(target=do_notify, daemon=True).start()

    def _notify_web_dashboard_cards_updated(self, device_id):
        """Notify web dashboard that cards have been updated (triggers WebSocket broadcast)."""

        def do_notify():
            try:
                import urllib.request
                import json as json_module

                data = json_module.dumps({"device_id": device_id}).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:6767/api/cards/sync",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=2)
                log.debug(f"Notified web dashboard of cards update for {device_id}")
            except Exception as e:
                log.debug(f"Could not notify web dashboard: {e}")

        threading.Thread(target=do_notify, daemon=True).start()

    def _notify_web_dashboard_collections_updated(self, device_id):
        """Notify web dashboard that collections have been updated (triggers WebSocket broadcast)."""

        def do_notify():
            try:
                import urllib.request
                import json as json_module

                data = json_module.dumps({"device_id": device_id}).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:6767/api/collections/sync",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=2)
                log.debug(
                    f"Notified web dashboard of collections update for {device_id}"
                )
            except Exception as e:
                log.debug(f"Could not notify web dashboard: {e}")

        threading.Thread(target=do_notify, daemon=True).start()

    def _notify_web_dashboard_session_message(
        self, session_id, message, is_update=False
    ):
        """Send a session message update to the web dashboard."""

        def do_notify():
            try:
                import urllib.request
                import json as json_module

                payload = json_module.dumps(
                    {
                        "session_id": session_id,
                        "message": message,
                        "is_update": bool(is_update),
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:6767/api/sessions/message",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=2)
            except Exception as e:
                log.debug(f"Could not notify web dashboard: {e}")

        threading.Thread(target=do_notify, daemon=True).start()

    def refresh_notes_ui(self):
        """Refresh the notes list UI."""
        if not hasattr(self, "notes_container"):
            return

        for widget in self.notes_container.winfo_children():
            widget.destroy()

        notes = []
        if getattr(self, "selected_notes_device_id", "__ALL__") == "__ALL__":
            for device in (self.notes_devices or {}).values():
                for note in device.get("notes", []) or []:
                    if isinstance(note, dict):
                        n = dict(note)
                        n["_device_name"] = device.get("name") or device.get("id")
                        n["_device_id"] = device.get("id")
                        n["_device_ip"] = device.get("ip")
                        n["_note_content_port"] = device.get("note_content_port")
                        notes.append(n)
        else:
            device = (self.notes_devices or {}).get(self.selected_notes_device_id) or {}
            for note in device.get("notes", []) or []:
                if isinstance(note, dict):
                    n = dict(note)
                    n["_device_name"] = device.get("name") or device.get("id")
                    n["_device_id"] = device.get("id")
                    n["_device_ip"] = device.get("ip")
                    n["_note_content_port"] = device.get("note_content_port")
                    notes.append(n)

        if hasattr(self, "notes_count_label"):
            self.notes_count_label.configure(text=f"{len(notes)}")

        if not notes:
            tk.Label(
                self.notes_container,
                text="No notes synced yet.\nNotes will appear here after your phone syncs.",
                bg=COLORS["bg_card"],
                fg=COLORS["text_dim"],
                font=("Segoe UI", 9),
                justify="center",
                anchor="center",
                relief="flat",
                borderwidth=0,
                highlightthickness=0,
            ).pack(fill=tk.BOTH, expand=True, pady=20)
            return

        # Sort by last edited (most recent first)
        def get_note_timestamp(n):
            return (
                n.get("updated_at")
                or n.get("updatedAt")
                or n.get("created_at")
                or n.get("createdAt")
                or 0
            )

        notes.sort(key=get_note_timestamp, reverse=True)

        for note in notes:
            self._add_note_row(note)

    def _add_note_row(self, note):
        """Add a single expandable note card to the UI."""
        title = note.get("title", "Untitled")
        note_id = note.get("id", "")
        device_name = note.get("_device_name")
        device_ip = note.get("_device_ip")
        device_id = note.get("_device_id")
        note_port = note.get("_note_content_port")

        # Main card container
        card = tk.Frame(
            self.notes_container,
            bg=COLORS["bg_input"],
            padx=8,
            pady=6,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        card.pack(fill=tk.X, pady=2)
        card._is_expanded = False
        card._content_loaded = False
        card._note_content = ""

        # Header row (clickable)
        header = tk.Frame(
            card,
            bg=COLORS["bg_input"],
            cursor="hand2",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        header.pack(fill=tk.X)

        # Expand indicator
        expand_label = tk.Label(
            header,
            text="â–¶",
            bg=COLORS["bg_input"],
            fg=COLORS["text_dim"],
            font=("Segoe UI", 8),
            width=2,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        expand_label.pack(side=tk.LEFT)

        info_frame = tk.Frame(
            header,
            bg=COLORS["bg_input"],
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        title_label = tk.Label(
            info_frame,
            text=title,
            bg=COLORS["bg_input"],
            fg=COLORS["text"],
            font=("Segoe UI", 9, "bold"),
            anchor="w",
            cursor="hand2",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        title_label.pack(anchor="w")

        if (
            device_name
            and getattr(self, "selected_notes_device_id", "__ALL__") == "__ALL__"
        ):
            tk.Label(
                info_frame,
                text=str(device_name),
                bg=COLORS["bg_input"],
                fg=COLORS["text_dim"],
                font=("Segoe UI", 8),
                anchor="w",
                relief="flat",
                borderwidth=0,
                highlightthickness=0,
            ).pack(anchor="w")

        # Content frame (initially hidden)
        content_frame = tk.Frame(
            card,
            bg=COLORS["bg_card"],
            padx=8,
            pady=8,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )

        # Loading label
        loading_label = tk.Label(
            content_frame,
            text="Loading...",
            bg=COLORS["bg_card"],
            fg=COLORS["text_dim"],
            font=("Segoe UI", 9, "italic"),
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )

        # Button bar for copy action (stored on card for access in _display_note_content)
        button_bar = tk.Frame(
            content_frame,
            bg=COLORS["bg_card"],
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
        )
        card._button_bar = button_bar

        def copy_note_content():
            """Copy note content to clipboard."""
            if card._note_content:
                self.root.clipboard_clear()
                self.root.clipboard_append(card._note_content)
                self.root.update()  # Required for clipboard to work
                # Show feedback
                copy_btn.configure(text="Copied!", fg=COLORS["success"])
                self.root.after(
                    1500, lambda: copy_btn.configure(text="Copy", fg="#ffffff")
                )

        copy_btn = tk.Button(
            button_bar,
            text="Copy",
            bg=COLORS["accent"],
            fg="#ffffff",
            font=("Segoe UI", 8),
            relief="flat",
            cursor="hand2",
            padx=8,
            pady=2,
            bd=0,
            highlightthickness=0,
            activebackground=COLORS["accent_hover"],
            activeforeground="white",
            command=copy_note_content,
        )
        copy_btn.pack(side=tk.RIGHT, padx=2)

        # Content text widget with scrollbar
        content_text = tk.Text(
            content_frame,
            bg=COLORS["bg_card"],
            fg=COLORS["text"],
            font=("Consolas", 9),
            wrap=tk.WORD,
            relief="flat",
            height=10,
            padx=4,
            pady=4,
            bd=0,
            highlightthickness=0,
            insertbackground=COLORS["text"],
        )
        content_text.configure(state="disabled")  # Read-only

        # Right-click context menu for copy
        context_menu = tk.Menu(
            content_text,
            tearoff=0,
            bg=COLORS["bg_elevated"],
            fg=COLORS["text"],
            activebackground=COLORS["accent"],
            activeforeground="white",
            bd=0,
        )
        context_menu.add_command(label="Copy All", command=copy_note_content)
        context_menu.add_command(
            label="Copy Selection", command=lambda: self._copy_selection(content_text)
        )

        def show_context_menu(event):
            try:
                context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                context_menu.grab_release()

        content_text.bind("<Button-3>", show_context_menu)

        # Also enable Ctrl+C for copying selection
        def on_ctrl_c(event):
            self._copy_selection(content_text)
            return "break"

        content_text.bind("<Control-c>", on_ctrl_c)

        def toggle_expand(event=None):
            if card._is_expanded:
                # Collapse
                content_frame.pack_forget()
                expand_label.configure(text="â–¶")
                card._is_expanded = False
            else:
                # Expand
                content_frame.pack(fill=tk.X, pady=(6, 0))
                expand_label.configure(text="â–¼")
                card._is_expanded = True

                # Load content if not already loaded
                if not card._content_loaded:
                    loading_label.pack(anchor="w")
                    self._fetch_note_content(
                        note_id,
                        device_ip,
                        note_port,
                        device_id,
                        card,
                        content_text,
                        loading_label,
                    )

        # Bind click events to header elements (single-click to expand)
        for widget in [header, expand_label, title_label, info_frame]:
            widget.bind("<Button-1>", toggle_expand)

        # Double-click to open in external editor
        def on_double_click(event):
            self.open_note(note_id, device_ip, title, note_port, device_id)

        for widget in [header, expand_label, title_label, info_frame]:
            widget.bind("<Double-Button-1>", on_double_click)

        # Tooltip
        self._create_tooltip(header, "Double-click to edit")

        # Hover effect on header
        def on_enter(e):
            header.configure(bg=COLORS["bg_elevated"])
            expand_label.configure(bg=COLORS["bg_elevated"])
            info_frame.configure(bg=COLORS["bg_elevated"])
            title_label.configure(bg=COLORS["bg_elevated"])
            for child in info_frame.winfo_children():
                child.configure(bg=COLORS["bg_elevated"])

        def on_leave(e):
            header.configure(bg=COLORS["bg_input"])
            expand_label.configure(bg=COLORS["bg_input"])
            info_frame.configure(bg=COLORS["bg_input"])
            title_label.configure(bg=COLORS["bg_input"])
            for child in info_frame.winfo_children():
                child.configure(bg=COLORS["bg_input"])

        header.bind("<Enter>", on_enter)
        header.bind("<Leave>", on_leave)

    def _fetch_note_content(
        self,
        note_id,
        device_ip,
        note_port,
        device_id,
        card,
        content_text,
        loading_label,
    ):
        """Fetch note content from local cache or device and display inline."""
        # Check local cache first (instant loading)
        cached = get_note_content_from_cache(note_id)
        if cached and cached.get("content"):
            self._display_note_content(
                card, content_text, loading_label, cached["content"]
            )
            return

        if not device_ip:
            self._show_note_error(
                content_text, loading_label, "Device IP not available"
            )
            return

        def fetch():
            sock = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                port = NOTE_CONTENT_PORT
                if isinstance(note_port, int) and 1 <= note_port <= 65535:
                    port = note_port
                elif isinstance(device_id, str):
                    package_name = device_id.split(":")[-1]
                    if package_name.endswith(".release6"):
                        port = NOTE_CONTENT_PORT + 1
                    elif "debug" in package_name:
                        port = NOTE_CONTENT_PORT + 2
                sock.connect((device_ip, port))

                request = json.dumps(
                    {"action": "fetch_note", "note_id": note_id}
                ).encode("utf-8")
                sock.sendall(len(request).to_bytes(4, "big"))
                sock.sendall(request)

                response_len = int.from_bytes(sock.recv(4), "big")
                if response_len <= 0 or response_len > 1000000:
                    raise Exception("Invalid response length")

                response_data = b""
                while len(response_data) < response_len:
                    chunk = sock.recv(min(4096, response_len - len(response_data)))
                    if not chunk:
                        break
                    response_data += chunk

                response = json.loads(response_data.decode("utf-8"))
                if not response.get("success"):
                    raise Exception(response.get("message", "Failed to fetch note"))

                content = response.get("content", "")

                # Update UI on main thread
                self.root.after(
                    0,
                    lambda: self._display_note_content(
                        card, content_text, loading_label, content
                    ),
                )

            except socket.timeout:
                self.root.after(
                    0,
                    lambda: self._show_note_error(
                        content_text,
                        loading_label,
                        "Connection timeout - is the app open?",
                    ),
                )
            except (ConnectionRefusedError, ConnectionResetError, OSError) as e:
                self.root.after(
                    0,
                    lambda: self._show_note_error(
                        content_text, loading_label, "Could not connect to device"
                    ),
                )
            except Exception as e:
                self.root.after(
                    0,
                    lambda: self._show_note_error(
                        content_text, loading_label, f"Error: {str(e)}"
                    ),
                )
            finally:
                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass

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
        if hasattr(card, "_button_bar"):
            card._button_bar.pack(fill=tk.X, pady=(0, 4))
        content_text.pack(fill=tk.BOTH, expand=True)
        content_text.configure(state="normal")
        content_text.delete("1.0", tk.END)
        content_text.insert("1.0", content)
        content_text.configure(state="disabled")
        card._content_loaded = True
        card._note_content = content

    def _show_note_error(self, content_text, loading_label, message):
        """Show error message in note content area."""
        loading_label.pack_forget()
        content_text.pack(fill=tk.BOTH, expand=True)
        content_text.configure(state="normal")
        content_text.delete("1.0", tk.END)
        content_text.insert("1.0", f"[Error] {message}")
        content_text.configure(
            state="disabled", fg=COLORS["error"] if "error" in COLORS else "#ff6b6b"
        )

    def open_note(self, note_id, device_ip, title, note_port=None, device_id=None):
        """Fetch note content from device and open in default editor."""
        if not device_ip:
            messagebox.showerror(
                "Error", "Device IP not available.\nReconnect your device."
            )
            return

        def fetch_and_open():
            sock = None
            try:
                # Connect to device's NoteContentServer
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                port = NOTE_CONTENT_PORT
                if isinstance(note_port, int) and 1 <= note_port <= 65535:
                    port = note_port
                elif isinstance(device_id, str):
                    package_name = device_id.split(":")[-1]
                    if package_name.endswith(".release6"):
                        port = NOTE_CONTENT_PORT + 1
                    elif "debug" in package_name:
                        port = NOTE_CONTENT_PORT + 2
                sock.connect((device_ip, port))

                # Send request
                request = json.dumps(
                    {"action": "fetch_note", "note_id": note_id}
                ).encode("utf-8")
                sock.sendall(len(request).to_bytes(4, "big"))
                sock.sendall(request)

                # Read response
                response_len = int.from_bytes(sock.recv(4), "big")
                if response_len <= 0 or response_len > 1000000:
                    raise Exception("Invalid response length")

                response_data = b""
                while len(response_data) < response_len:
                    chunk = sock.recv(min(4096, response_len - len(response_data)))
                    if not chunk:
                        break
                    response_data += chunk

                response = json.loads(response_data.decode("utf-8"))
                if not response.get("success"):
                    raise Exception(response.get("message", "Failed to fetch note"))

                content = response.get("content", "")
                note_title = response.get("title", title)

                # Write to temp file and open
                illegal_chars = '<>:"/\\|?*' if IS_WINDOWS else "/"
                safe_title = "".join(
                    c for c in note_title if c not in illegal_chars
                ).strip()[:100]
                temp_dir = os.path.join(tempfile.gettempdir(), "shadowai_notes")
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, f"{safe_title}_{note_id[:8]}.txt")

                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write(content)

                if IS_WINDOWS:
                    os.startfile(temp_path)
                elif platform.system() == "Darwin":
                    subprocess.run(["open", temp_path])
                else:
                    subprocess.run(["xdg-open", temp_path])

                log.info(f"Opened note: {note_title}")

            except socket.timeout:
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Connection Timeout",
                        "Could not connect to device.\nMake sure the Shadow app is open.",
                    ),
                )
            except (ConnectionRefusedError, ConnectionResetError, OSError) as e:
                log.error(f"Failed to connect to device: {e}")
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Device Not Connected",
                        "Could not connect to the device.\n\nMake sure:\n- Your phone is on the same network\n- The Shadow app is open\n- The device hasn't gone to sleep",
                    ),
                )
            except Exception as e:
                log.error(f"Failed to open note: {e}")
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Error", f"Failed to open note:\n{str(e)}"
                    ),
                )
            finally:
                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass

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
                menu.add_command(
                    label=label, command=lambda l=label: self.notes_device_var.set(l)
                )

            add_option("All devices", "__ALL__")

            for device_id, device in (self.notes_devices or {}).items():
                if isinstance(device, dict):
                    name = device.get("name") or device_id
                    label = f"{name}"
                    add_option(label, device_id)

            # Keep selection stable by device_id
            desired = (
                self.selected_notes_device_id
                if getattr(self, "selected_notes_device_id", "__ALL__")
                else "__ALL__"
            )
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
        self.selected_notes_device_id = getattr(
            self, "_notes_device_label_to_id", {}
        ).get(label, "__ALL__")
        self.refresh_notes_ui()


def _start_show_listener(lock_socket):
    """Listen for SHOW commands from other instances."""
    import threading

    def listener():
        while True:
            try:
                conn, _ = lock_socket.accept()
                data = conn.recv(16).decode("utf-8", errors="ignore").strip()
                conn.close()
                if data == "SHOW" and _app_instance:
                    # Schedule window restore on main thread
                    _app_instance.root.after(0, _app_instance.show_window)
            except socket.error:
                # Socket closed or connection error - expected during shutdown
                pass
            except Exception as e:
                log.debug(f"Show listener error: {e}")

    t = threading.Thread(target=listener, daemon=True)
    t.start()


def check_single_instance(port=19287):
    """Check if another instance is already running. Returns lock socket if successful."""
    try:
        # Try to bind to a specific port - if it fails, another instance is running
        lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lock_socket.bind(
            ("127.0.0.1", port)
        )  # Use a dedicated port for instance lock (not 19286 - companion uses that)
        lock_socket.listen(1)
        _start_show_listener(lock_socket)
        return lock_socket
    except socket.error:
        return None


def run_web_dashboard_server(open_browser: bool):
    """Run the web dashboard server (used by the --web-server mode)."""
    # Single-instance check using dynamic lock port
    web_lock_port = 6766
    if ENVIRONMENT == "DEBUG":
        web_lock_port = 6776
    elif ENVIRONMENT == "AIDEV":
        web_lock_port = 6786

    lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock_socket.bind(("127.0.0.1", web_lock_port))
        lock_socket.listen(1)
    except OSError:
        # Another instance is running - just open browser to existing
        log.info(f"{ENVIRONMENT} dashboard already running, opening browser...")
        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{WEB_PORT}")
        return

    try:
        from web.app import create_app, socketio

        host = "0.0.0.0"  # Allow external access
        port = WEB_PORT

        # Check if port is already in use (main GUI may already be running DataReceiver)
        def is_port_in_use(port):
            """Check if a port is already bound by another process."""
            import socket

            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 0
            )  # Don't reuse
            test_sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1
            )  # Windows: exclusive bind
            try:
                test_sock.bind(
                    ("0.0.0.0", port)
                )  # Use 0.0.0.0 to match main GUI's bind
                test_sock.close()
                return False  # Port is free
            except OSError:
                return True  # Port is in use
            finally:
                try:
                    test_sock.close()
                except Exception:
                    pass

        # Only start DataReceiver if the port is free (main GUI not running)
        data_receiver = None
        data_receiver_ref = [None]

        if is_port_in_use(DATA_PORT):
            log.info(
                f"Port {DATA_PORT} already in use - main GUI is handling key exchanges"
            )
            log.info("Skipping DataReceiver in web-server mode to avoid conflicts")
        else:
            # CRITICAL FIX: Start DataReceiver and Discovery server for SSH key exchange
            log.info(
                "Starting DataReceiver for key exchange in web-server mode (headless)..."
            )

            # SECURITY: Load or generate headless approval PIN
            headless_pin_path = os.path.join(HOME_DIR, ".shadowai", "headless_pin")
            headless_pin = None
            try:
                if os.path.exists(headless_pin_path):
                    with open(headless_pin_path, "r") as f:
                        headless_pin = f.read().strip()
                    if headless_pin:
                        log.info(
                            f"Headless approval PIN loaded from {headless_pin_path}"
                        )
                    else:
                        headless_pin = None
            except (IOError, OSError) as e:
                log.error(f"Failed to read headless PIN file: {e}")

            if not headless_pin:
                log.warning(
                    "[SECURITY] No headless approval PIN configured."
                )
                log.warning(
                    f"[SECURITY] SSH key requests will be REJECTED in headless mode."
                )
                log.warning(
                    f"[SECURITY] To enable headless approval, create {headless_pin_path} "
                    f"with a PIN/token value."
                )

            def headless_key_approval(device_id, device_name, ip):
                """Approve SSH keys in headless mode only if PIN is configured."""
                if not headless_pin:
                    log.warning(
                        f"HEADLESS MODE: REJECTING SSH key for {device_name} ({device_id}) from {ip} "
                        f"- no headless_pin configured"
                    )
                    log.warning(
                        f"[SECURITY] Create {headless_pin_path} with a PIN to enable headless approval"
                    )
                    return

                log.warning(
                    f"HEADLESS MODE: Approving SSH key for {device_name} ({device_id}) from {ip} "
                    f"(headless_pin configured)"
                )
                # Approve the device since PIN-based headless mode is opted in
                if data_receiver_ref[0]:
                    result = data_receiver_ref[0].approve_device(device_id)
                    log.info(f"Headless approval result: {result}")

            data_receiver = DataReceiver(
                on_data_received=lambda device_id, projects: log.info(
                    f"Projects received from {device_id}"
                ),
                on_device_connected=lambda device_id, ip: log.info(
                    f"Device connected: {device_id} from {ip}"
                ),
                on_notes_received=lambda device_id, notes: log.info(
                    f"Notes received from {device_id}"
                ),
                on_sessions_received=lambda device_id, sessions: log.info(
                    f"Sessions received from {device_id}"
                ),
                on_key_approval_needed=headless_key_approval,
            )
            data_receiver_ref[0] = data_receiver  # Store reference for callback
            data_receiver.start()
            log.info(f"DataReceiver started on port {DATA_PORT}")

        # Start discovery server
        log.info("Starting discovery server...")
        ssh_port = find_ssh_port() or 22

        # Build connection_info for DiscoveryServer
        all_ips = get_all_ips()
        tailscale_ip = get_tailscale_ip()
        local_ip = get_local_ip()
        hostname_local = get_hostname_local()

        # Get dynamic encryption salt
        try:
            from web.services.data_service import DYNAMIC_SALT

            salt_b64 = base64.b64encode(DYNAMIC_SALT).decode("ascii")
        except Exception:
            salt_b64 = None

        # Build hosts list
        hosts_to_try = []
        if tailscale_ip:
            hosts_to_try.append(
                {"host": tailscale_ip, "type": "tailscale", "stable": True}
            )
        if local_ip:
            hosts_to_try.append({"host": local_ip, "type": "local", "stable": False})

        connection_info = {
            "type": "shadowai_connect",
            "version": 3,
            "mode": "tailscale" if tailscale_ip else "local",
            "host": tailscale_ip or local_ip,
            "port": ssh_port,
            "username": get_username(),
            "hostname": socket.gethostname(),
            "hostname_local": hostname_local,
            "hosts": hosts_to_try,
            "local_ip": local_ip,
            "tailscale_ip": tailscale_ip,
            "encryption_salt": salt_b64,
            "timestamp": int(time.time()),
        }

        discovery_server = DiscoveryServer(connection_info)
        discovery_server.start()
        log.info(f"Discovery server started on port {DISCOVERY_PORT}")

        # Graceful shutdown handler
        shutdown_initiated = [False]  # Use list to allow mutation in nested function

        def shutdown_handler(signum=None, frame=None):
            """Gracefully shut down all servers."""
            if shutdown_initiated[0]:
                return  # Prevent double-shutdown
            shutdown_initiated[0] = True

            log.info("Shutting down web dashboard server...")

            # Stop DataReceiver
            if data_receiver:
                try:
                    data_receiver.stop()
                    log.info("DataReceiver stopped")
                except Exception as e:
                    log.debug(f"Error stopping DataReceiver: {e}")

            # Stop DiscoveryServer
            if discovery_server:
                try:
                    discovery_server.stop()
                    log.info("DiscoveryServer stopped")
                except Exception as e:
                    log.debug(f"Error stopping DiscoveryServer: {e}")

            # Close lock socket
            try:
                lock_socket.close()
                log.info("Lock socket closed")
            except Exception as e:
                log.debug(f"Error closing lock socket: {e}")

            log.info("Web dashboard shutdown complete")

        # Register signal handlers (Windows supports SIGINT, SIGTERM may not work)
        import signal
        import atexit

        signal.signal(signal.SIGINT, shutdown_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, shutdown_handler)

        # Register atexit handler for cleanup
        atexit.register(shutdown_handler)

        log.info("Graceful shutdown handlers registered")

        def open_browser_delayed():
            time.sleep(1.5)
            # Browser should still open to 127.0.0.1 for local user convenience
            webbrowser.open(f"http://127.0.0.1:{port}")

        if open_browser:
            thread = threading.Thread(target=open_browser_delayed, daemon=True)
            thread.start()

        app = create_app()
        try:
            from web.services.scheduler_service import get_scheduler

            get_scheduler().start()
        except Exception as e:
            log.error(f"Failed to start scheduler: {e}")

        # Use socketio.run if available, otherwise fall back to Flask's app.run
        if getattr(app, "socketio_enabled", False) and socketio is not None:
            socketio.run(
                app,
                host=host,
                port=port,
                debug=False,
                use_reloader=False,
                log_output=True,
            )
        else:
            # Fallback for PyInstaller frozen builds where socketio may be disabled
            app.run(host=host, port=port, debug=False, use_reloader=False)
    except Exception:
        try:
            import traceback

            with open(
                WEB_LOG_FILE, "a", encoding="utf-8", errors="replace"
            ) as log_file:
                log_file.write("\n=== Web dashboard crash ===\n")
                log_file.write(traceback.format_exc())
                log_file.write("\n")
        except Exception:
            pass
        raise


def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler to log tracebacks."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    log.critical(f"Unhandled exception:\n{error_msg}")

    # Save to desktop for easy retrieval
    try:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        with open(os.path.join(desktop, "shadow_error.log"), "a") as f:
            f.write(f"\n--- {datetime.now()} ---\n{error_msg}\n")
    except Exception:
        pass

    # Sentinel V3: Analyze exception and offer fix suggestion
    suggestion_id = None
    if HAS_SENTINEL:
        try:
            from ouroboros.sentinel import get_sentinel_analyzer
            analyzer = get_sentinel_analyzer()

            # Analyze the exception
            analysis = analyzer.analyze_exception(exc_type, exc_value, exc_traceback)

            if analysis:
                # Generate a fix suggestion
                suggestion_id = analyzer.suggest_fix(analysis)
                if suggestion_id:
                    log.info(f"[Sentinel] Created fix suggestion: {suggestion_id}")
        except Exception as sentinel_err:
            log.debug(f"Sentinel analysis skipped: {sentinel_err}")

    # Also show a non-blocking messagebox if possible
    try:
        import tkinter.messagebox as messagebox

        # Try to use the existing root if available
        parent = None
        if (
            "_app_instance" in globals()
            and _app_instance
            and hasattr(_app_instance, "root")
        ):
            parent = _app_instance.root

        # Include fix suggestion info if available
        extra_msg = ""
        if suggestion_id:
            extra_msg = "\n\n[*] Sentinel has a fix suggestion.\nView in dashboard: http://localhost:6767/ouroboros"

        messagebox.showerror(
            "ShadowBridge Error",
            f"An unhandled error occurred:\n\n{exc_value}\n\nSee shadow_error.log on your desktop.{extra_msg}",
            parent=parent,
        )
    except Exception:
        pass


sys.excepthook = handle_exception


def _launch_ouroboros_refiner():
    """Launch the Ouroboros Refiner in watch mode as a background subprocess."""
    refiner_script = Path("C:/shadow/scripts/ouroboros_refiner.py")
    if not refiner_script.exists():
        log.warning(f"Ouroboros Refiner not found at {refiner_script}")
        return None

    github_token = os.environ.get("GITHUB_TOKEN")
    # Fallback: read from config file if env var not set
    if not github_token:
        token_file = Path(HOME_DIR) / ".shadowai" / "github_token"
        if token_file.exists():
            github_token = token_file.read_text().strip()
            log.info("Loaded GITHUB_TOKEN from config file")
    if not github_token:
        log.warning("GITHUB_TOKEN not set - Ouroboros Refiner will not start")
        log.warning("Set env var or create ~/.shadowai/github_token file")
        return None

    # Find Python interpreter (sys.executable may be frozen ShadowBridge.exe)
    import shutil
    python_cmd = None
    for candidate in ["C:/Windows/py.exe", "python", "python3"]:
        if shutil.which(candidate):
            python_cmd = candidate
            break
    if not python_cmd:
        log.warning("Python interpreter not found - Ouroboros Refiner will not start")
        return None

    try:
        log.info(f"Launching Ouroboros Refiner (--watch --interval 120) via {python_cmd}...")
        env = os.environ.copy()
        env["GITHUB_TOKEN"] = github_token

        startupinfo = None
        if IS_WINDOWS:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE

        proc = subprocess.Popen(
            [python_cmd, str(refiner_script), "--watch", "--interval", "120"],
            env=env,
            startupinfo=startupinfo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info(f"Ouroboros Refiner started (PID {proc.pid})")
        return proc
    except Exception as e:
        log.error(f"Failed to launch Ouroboros Refiner: {e}")
        return None


def _launch_ouroboros_verifier():
    """Launch the Ouroboros Fix Verifier in watch mode as a background subprocess."""
    verifier_script = Path("C:/shadow/scripts/ouroboros_verifier.py")
    if not verifier_script.exists():
        log.warning(f"Ouroboros Verifier not found at {verifier_script}")
        return None

    # Find Python interpreter
    import shutil
    python_cmd = None
    for candidate in ["C:/Windows/py.exe", "python", "python3"]:
        if shutil.which(candidate):
            python_cmd = candidate
            break
    if not python_cmd:
        log.warning("Python interpreter not found - Ouroboros Verifier will not start")
        return None

    try:
        log.info(f"Launching Ouroboros Verifier (--watch --interval 21600) via {python_cmd}...")

        startupinfo = None
        if IS_WINDOWS:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE

        proc = subprocess.Popen(
            [python_cmd, str(verifier_script), "--watch", "--interval", "21600"],
            startupinfo=startupinfo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info(f"Ouroboros Verifier started (PID {proc.pid})")
        return proc
    except Exception as e:
        log.error(f"Failed to launch Ouroboros Verifier: {e}")
        return None


def main():
    """Main entry point."""
    global DEBUG_BUILD, AIDEV_MODE, AGENT_MODE, _single_instance

    # Refresh flags from current argv (in case they were modified or for clarity)
    DEBUG_BUILD = "--debug" in sys.argv
    AIDEV_MODE = "--aidev" in sys.argv
    AGENT_MODE = "--mode" in sys.argv and "agent" in sys.argv
    PING_MODE = "--ping" in sys.argv

    if PING_MODE:
        if HAS_SINGLETON:
            lock_name = f"ShadowBridge{ENVIRONMENT}"
            instance_port = 19287
            if ENVIRONMENT == "DEBUG":
                instance_port = 19297
            elif ENVIRONMENT == "AIDEV":
                instance_port = 19307

            inst = SingleInstance(lock_name, port=instance_port)
            if not inst.acquire():
                log.info("Sending ping to existing instance")
                inst.send_ping()
                sys.exit(0)
            else:
                log.info("No existing instance found to ping")
                sys.exit(0)
        else:
            sys.exit(0)

    if IMAGE_MODE:
        run_image_command()
        return
        run_video_command()
        return

    if AUDIO_MODE:
        run_audio_command()
        return

    if ASSEMBLY_MODE:
        # run_assembly_command()  # Not implemented yet in this version
        return

    if BROWSER_MODE:
        # run_browser_command() # Not implemented yet
        return

    if WEB_SERVER_MODE:
        open_browser = "--no-browser" not in sys.argv
        run_web_dashboard_server(open_browser=open_browser)
        return

    if AGENT_MODE:
        log.info("Running in ShadowAgent mode (Headless)")
        # Headless mode for AIDEV agents.
        # Starts web server without browser and without tray GUI.

        # Check for existing instance
        if HAS_SINGLETON:
            _single_instance = SingleInstance("ShadowBridgeAgent", port=19288)
            if not _single_instance.acquire():
                log.info("Another ShadowAgent instance is already running.")
                sys.exit(0)

        # Register install path
        try:
            register_install_path()
        except Exception:
            pass

        # Start web server (blocking)
        run_web_dashboard_server(open_browser=False)
        return

    start_minimized = "--minimized" in sys.argv

    # Check for existing instance using robust SingleInstance if available
    if HAS_SINGLETON:
        # Use different lock names and ports for side-by-side instances
        lock_name = f"ShadowBridge{ENVIRONMENT}"
        instance_port = 19287
        if ENVIRONMENT == "DEBUG":
            instance_port = 19297
        elif ENVIRONMENT == "AIDEV":
            instance_port = 19307

        _single_instance = SingleInstance(lock_name, port=instance_port)
        if not _single_instance.acquire():
            # Another instance is already running
            log.info(
                f"Another {ENVIRONMENT} instance is already running, requesting activation"
            )
            _single_instance.send_activate()
            sys.exit(0)
    else:
        # Fallback to simple socket check
        lock_port = 19287
        if ENVIRONMENT == "DEBUG":
            lock_port = 19297
        elif ENVIRONMENT == "AIDEV":
            lock_port = 19307

        lock_socket = check_single_instance(port=lock_port)
        if lock_socket is None:
            # Another instance is already running - try to show it
            if IS_WINDOWS:
                try:
                    show_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    show_sock.settimeout(2)
                    show_sock.connect(("127.0.0.1", lock_port))
                    show_sock.sendall(b"SHOW")
                    show_sock.close()
                except Exception:
                    pass
            sys.exit(0)

    if not HAS_PIL:
        log.critical("Missing pillow. Install with: pip install pillow")
        sys.exit(1)

    log.info(f"ShadowBridge {ENVIRONMENT} v{APP_VERSION} starting...")

    # Register install path so Android app can find us via SSH
    try:
        register_install_path()
    except Exception as e:
        log.error(f"Failed to register install path: {e}")

    app = ShadowBridgeApp()
    global _app_instance
    _app_instance = app
    log.info(f"ShadowBridge {ENVIRONMENT} initialized")

    if HAS_SINGLETON:

        def on_activate():
            """Called when another instance requests activation."""
            log.info(f"Received activation request for {ENVIRONMENT}")
            try:
                if "_app_instance" in globals() and _app_instance:
                    # Schedule on main thread (Tkinter is not thread-safe)
                    _app_instance.root.after(0, _app_instance.show_window)
            except Exception as e:
                log.error(f"Failed to process activation: {e}")

        _single_instance.set_activation_callback(on_activate)

        def on_ping():
            # Called when another instance requests ping
            log.info(f"Received ping request for {ENVIRONMENT}")
            try:
                if "_app_instance" in globals() and _app_instance:
                    _app_instance.root.after(0, _app_instance.show_ping_effects)
            except Exception as e:
                log.error(f"Failed to process ping: {e}")

        _single_instance.set_ping_callback(on_ping)

    if start_minimized and HAS_TRAY:
        app.root.after(500, app.minimize_to_tray)

    if HEADLESS_MODE:
        log.info("Running in Headless mode (window hidden)")
        app.root.withdraw()

    # Launch Ouroboros Refiner in AIDEV mode (auto-fix GitHub issues)
    refiner_proc = None
    verifier_proc = None
    if AIDEV_MODE:
        refiner_proc = _launch_ouroboros_refiner()
        verifier_proc = _launch_ouroboros_verifier()

    try:
        app.run()
    finally:
        if refiner_proc and refiner_proc.poll() is None:
            log.info("Stopping Ouroboros Refiner...")
            refiner_proc.terminate()
        if verifier_proc and verifier_proc.poll() is None:
            log.info("Stopping Ouroboros Verifier...")
            verifier_proc.terminate()


if __name__ == "__main__":
    main()
