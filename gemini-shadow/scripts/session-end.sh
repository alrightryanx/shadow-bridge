#!/usr/bin/env bash
# Gemini Shadow - Session End Hook
# Notifies ShadowBridge when a Gemini CLI session ends

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/companion-common.sh"

# Read hook input from stdin
HOOK_INPUT=$(read_hook_input)

log_debug "session-end: Hook invoked"
log_debug "session-end: Input: $HOOK_INPUT"

# Extract session info
SESSION_ID=$(echo "$HOOK_INPUT" | jq -r '.session_id // .sessionId // ""' 2>/dev/null)

# Build session end message
MESSAGE=$(cat << EOF
{
    "type": "session_end",
    "id": "$(generate_id)",
    "sessionId": "$SESSION_ID",
    "timestamp": $(timestamp_ms),
    "provider": "gemini",
    "payload": {
        "hostname": "$(hostname)",
        "provider": "gemini"
    }
}
EOF
)

# Send to bridge (fire and forget)
send_fire_and_forget "$MESSAGE"

log_debug "session-end: Message sent"

# Always allow session to end
exit 0
