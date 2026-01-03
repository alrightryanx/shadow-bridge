#!/usr/bin/env python3
"""
Universal Plugin Generator for ShadowAI Session Persistence
Generates plugins for Claude Code, Gemini CLI, Cursor, and Codex
"""
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent))
from shared.logger import create_logger, PluginLogger


# Configuration
PLATFORM_CONFIGS = {
    "claude": {
        "name": "Claude Code",
        "plugin_dir": "claude-shadow",
        "script_language": "PowerShell",
        "script_extension": ".ps1",
        "config_file": "hooks.json",
        "install_method": "local_directory"
    },
    "gemini": {
        "name": "Gemini CLI",
        "plugin_dir": "gemini-shadow",
        "script_language": "Bash",
        "script_extension": ".sh",
        "config_file": ".gemini/settings.json",
        "install_method": "hooks_install"
    },
    "cursor": {
        "name": "Cursor IDE",
        "plugin_dir": "cursor-shadow",
        "script_language": "Bash",
        "script_extension": ".sh",
        "config_file": ".cursor/settings.json",
        "install_method": "cursor_settings"
    },
    "codex": {
        "name": "OpenAI Codex",
        "plugin_dir": "codex-shadow",
        "script_language": "TypeScript",
        "script_extension": ".ts",
        "config_file": "wrapper-config.json",
        "install_method": "sdk_wrapper"
    }
}


def load_json_file(path: str) -> Dict[str, Any]:
    """Load JSON file with error handling"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing {path}: {e}")
        sys.exit(1)


def write_file(path: str, content: str) -> None:
    """Write file with directory creation"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def generate_plugin_manifest(platform: str, name: str, description: str, version: str = "1.0.0") -> str:
    """Generate plugin.json manifest"""
    config = PLATFORM_CONFIGS[platform]
    
    if platform == "claude":
        return json.dumps({
            "name": name,
            "version": version,
            "description": description,
            "author": {
                "name": "ShadowAI",
                "email": "contact@ryancartwright.com"
            },
            "homepage": "https://ryancartwright.com/shadowai",
            "repository": "https://github.com/alrightryanx/shadowai-plugins",
            "license": "MIT",
            "keywords": [
                "notifications",
                "mobile",
                "companion",
                "shadowai",
                "android",
                "remote"
            ],
            "hooks": os.path.join(".", "hooks", "hooks.json")
        }, indent=2)
    
    elif platform == "gemini":
        return json.dumps({
            "name": name,
            "version": version,
            "description": description,
            "author": "ShadowAI",
            "license": "MIT",
            "keywords": [
                "notifications",
                "mobile",
                "companion",
                "shadowai",
                "android",
                "gemini"
            ]
        }, indent=2)
    
    elif platform == "cursor":
        return json.dumps({
            "name": name,
            "version": version,
            "description": description
        }, indent=2)
    
    elif platform == "codex":
        return json.dumps({
            "name": name,
            "version": version,
            "description": description,
            "main": "wrapper.js",
            "author": "ShadowAI",
            "license": "MIT"
        }, indent=2)
    
    else:
        return ""


def generate_hooks_config(platform: str, name: str) -> str:
    """Generate hooks configuration file"""
    config = PLATFORM_CONFIGS[platform]
    base_dir = Path(__file__).parent.parent / "claude-shadow"  # Reference implementation
    
    # Load event mappings
    event_mappings = load_json_file(
        Path(__file__).parent / "shared" / "event_mappings.json"
    )
    
    platform_events = event_mappings.get(platform, {})
    
    if platform == "claude":
        # Full hooks.json (reference existing)
        hooks_json = load_json_file(base_dir / "hooks" / "hooks.json")
        return json.dumps(hooks_json, indent=2)
    
    elif platform == "gemini":
        # Generate .gemini/settings.json with hooks
        settings = {
            "hooks": {}
        }
        
        # Map events from Claude to Gemini
        for claude_event, gemini_event in platform_events.items():
            if gemini_event:
                settings["hooks"][gemini_event] = [{
                    "type": "command",
                    "command": f"scripts/hooks/{claude_event.lower()}.sh",
                    "timeout": 60000
                }]
        
        # Add session hooks
        settings["hooks"]["SessionStart"] = [{
            "type": "command",
            "command": "scripts/hooks/session-start.sh",
            "timeout": 30000
        }]
        settings["hooks"]["SessionEnd"] = [{
            "type": "command",
            "command": "scripts/hooks/session-end.sh",
            "timeout": 30000
        }]
        
        return json.dumps(settings, indent=2)
    
    elif platform == "cursor":
        # Generate .cursor/settings.json with supported hooks only
        settings = {
            "hooks": {}
        }
        
        # Map only supported events
        for claude_event, cursor_event in platform_events.items():
            if cursor_event:
                settings["hooks"][cursor_event] = [{
                    "type": "command",
                    "command": f"scripts/hooks/{claude_event.lower()}.sh"
                }]
        
        return json.dumps(settings, indent=2)
    
    elif platform == "codex":
        # Generate wrapper config (no hooks, just config)
        config = {
            "version": "1.0.0",
            "port": 19286,
            "logLevel": "debug",
            "provider": "codex"
        }
        return json.dumps(config, indent=2)
    
    return ""


def generate_script(platform: str, name: str) -> Dict[str, str]:
    """Generate script file based on platform"""
    config = PLATFORM_CONFIGS[platform]
    
    if platform in ["claude", "gemini", "cursor"]:
        # Generate Bash/PowerShell script
        return generate_bash_script(platform, name, config)
    
    elif platform == "codex":
        # Generate TypeScript SDK wrapper
        return generate_typescript_wrapper(name, config)
    
    return {}


def generate_bash_script(platform: str, name: str, config: Dict[str, str]) -> Dict[str, str]:
    """Generate Bash/PowerShell hook script"""
    script_type = "PowerShell" if platform == "claude" else "Bash"
    ext = config["script_extension"]
    
    scripts = {
        "session-start": f"""#!/usr/bin/env bash
# {config['name']} - Session Start Hook
# Sends session start notification to ShadowBridge

SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Load shared utilities (if available)
if [ -f "$SCRIPT_ROOT/scripts/shared.sh" ]; then
    source "$SCRIPT_ROOT/scripts/shared.sh"
else
    # Inline utilities
    log_message() {{
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    }}
fi

# Read hook input from stdin
HOOK_INPUT=$(cat)

# Extract session_id and cwd
SESSION_ID=$(echo "$HOOK_INPUT" | grep -oP '"session_id"[[:space:]]*" | grep -oP '"[^"]*"' | cut -d'"' -f2)
CWD=$(echo "$HOOK_INPUT" | grep -oP '"cwd"[[:space:]]*" | grep -oP '"[^"]*"' | cut -d'"' -f2)

log_message "Session start hook: sessionId=$SESSION_ID, cwd=$CWD"

# Build session start message
MESSAGE='{{"type":"session_start","id":"msg_$(date +%s%N)","sessionId":"$SESSION_ID","timestamp":$(date +%s)000,"payload":{{"hostname":"$(hostname)","cwd":"$CWD"}}}}'

# Send to ShadowBridge
echo "$MESSAGE" |
