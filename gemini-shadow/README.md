# Gemini Shadow

Mobile notifications and session persistence for Gemini CLI, part of the ShadowAI ecosystem.

## Overview

Gemini Shadow enables real-time synchronization between Gemini CLI and ShadowAI Android app through ShadowBridge. Get notified on your phone when Gemini needs input, approve tool executions remotely, and maintain session history across devices.

## Features

- **Session Tracking**: Automatic session start/end notifications
- **Permission Requests**: Remote approval for tool executions (shell commands, file writes)
- **Notifications**: Forward Gemini notifications to your phone
- **Context Caching**: Maintains context for richer notifications
- **Provider Identification**: All messages tagged with `provider: "gemini"` for routing

## Requirements

- Gemini CLI v1.0.0 or later
- ShadowBridge running on Windows PC
- ShadowAI Android app
- Dependencies:
  - `bash` (Git Bash on Windows, native on Linux/Mac)
  - `jq` for JSON parsing
  - `python3` for TCP communication

## Installation

### Option 1: Manual Installation

1. Copy the `gemini-shadow` directory to your Gemini CLI plugins location:
   ```bash
   # Linux/Mac
   cp -r gemini-shadow ~/.gemini/plugins/

   # Windows (Git Bash)
   cp -r gemini-shadow "$APPDATA/gemini/plugins/"
   ```

2. Make scripts executable:
   ```bash
   chmod +x ~/.gemini/plugins/gemini-shadow/scripts/*.sh
   ```

3. Configure Gemini CLI to load the plugin:
   ```bash
   gemini config set plugins.gemini-shadow.enabled true
   ```

### Option 2: Migration from Claude Shadow

If you already have `claude-shadow` configured, you can migrate:

```bash
# Use Gemini CLI's migration tool (if available)
gemini hooks migrate --from-claude

# Or manually copy and adapt configuration
```

### Option 3: Using gemini hooks install (if supported)

```bash
gemini hooks install ./gemini-shadow/
```

## Configuration

Create `~/.gemini-shadow-config.json`:

```json
{
  "bridgeHost": "127.0.0.1",
  "bridgePort": 19286,
  "enabled": true
}
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `bridgeHost` | `127.0.0.1` | ShadowBridge host IP |
| `bridgePort` | `19286` | ShadowBridge companion port |
| `enabled` | `true` | Enable/disable plugin |

## Hook Events

| Event | Script | Description |
|-------|--------|-------------|
| `SessionStart` | `session-start.sh` | Triggered when Gemini CLI session begins |
| `SessionEnd` | `session-end.sh` | Triggered when session ends |
| `BeforeAgent` | `before-agent.sh` | Triggered before processing user prompt |
| `AfterAgent` | `after-agent.sh` | Triggered after agent completes turn |
| `BeforeTool` | `before-tool.sh` | Triggered before tool execution (permission requests) |
| `AfterTool` | `after-tool.sh` | Triggered after tool execution |
| `Notification` | `notification.sh` | Triggered for Gemini notifications |

## Message Protocol

All messages use the ShadowBridge length-prefixed JSON protocol:

```
[4-byte big-endian length][JSON payload]
```

### Example Messages

**Session Start:**
```json
{
  "type": "session_start",
  "id": "msg_1234567890_abcd1234",
  "sessionId": "gemini_1234567890_efgh5678",
  "timestamp": 1704067200000,
  "provider": "gemini",
  "payload": {
    "hostname": "MY-PC",
    "cwd": "/home/user/project",
    "projectName": "project",
    "provider": "gemini"
  }
}
```

**Permission Request:**
```json
{
  "type": "permission_request",
  "id": "msg_1234567890_abcd1234",
  "sessionId": "gemini_1234567890_efgh5678",
  "timestamp": 1704067200000,
  "provider": "gemini",
  "payload": {
    "requestId": "req_1234567890_ijkl9012",
    "toolName": "run_shell_command",
    "description": "Run: npm install",
    "options": ["Approve", "Deny", "Always Allow"],
    "promptType": "PERMISSION",
    "provider": "gemini"
  }
}
```

## Troubleshooting

### Check Debug Logs

```bash
# View recent log entries
tail -f ~/.gemini-shadow-debug.log
```

### Test ShadowBridge Connection

```bash
# Check if ShadowBridge is listening
nc -zv 127.0.0.1 19286
```

### Common Issues

1. **Connection Refused**: Ensure ShadowBridge is running and listening on port 19286
2. **Permission Denied**: Run `chmod +x` on all script files
3. **jq not found**: Install jq: `apt install jq` (Linux) or `brew install jq` (Mac)
4. **Python not found**: Ensure python3 is in PATH

## Architecture

```
┌─────────────────┐
│   Gemini CLI    │◄──hooks (SessionStart, BeforeTool, etc.)
│   (gemini-      │
│    shadow)      │
└────────┬────────┘
         │ JSON via TCP (port 19286)
         ▼
┌─────────────────┐
│  ShadowBridge   │◄──WebSocket──► Android App
│   (Windows PC)  │    (Room Database)
└─────────────────┘
```

## Event Mapping (Claude → Gemini)

| Claude Code Event | Gemini CLI Event |
|-------------------|------------------|
| SessionStart | SessionStart |
| SessionEnd | SessionEnd |
| PermissionRequest | BeforeTool |
| UserPromptSubmit | BeforeAgent |
| Stop | AfterAgent |
| Notification | Notification |

## Tool Mapping (Claude → Gemini)

| Claude Tool | Gemini Tool |
|-------------|-------------|
| Bash | run_shell_command |
| Edit | replace |
| Read | read_file |
| Write | write_file |
| Glob | glob |
| Grep | search_file_content |

## License

MIT License - See LICENSE file for details.

## Contact

- Website: https://ryancartwright.com/shadowai
- Email: contact@ryancartwright.com
- Issues: https://github.com/alrightryanx/shadowai-plugins/issues
