#!/usr/bin/env bash
# Gemini Shadow - Session Start Hook
# Notifies ShadowBridge when a Gemini CLI session starts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/companion-common.sh"

# Read hook input from stdin
HOOK_INPUT=$(read_hook_input)

log_debug "session-start: Hook invoked"
log_debug "session-start: Input: $HOOK_INPUT"

# Extract session info from Gemini CLI hook data
SESSION_ID=$(echo "$HOOK_INPUT" | jq -r '.session_id // .sessionId // ""' 2>/dev/null)
CWD=$(echo "$HOOK_INPUT" | jq -r '.cwd // .working_directory // ""' 2>/dev/null)

# Generate session ID if not provided
if [ -z "$SESSION_ID" ]; then
    SESSION_ID="gemini_$(date +%s)_$(head -c 4 /dev/urandom | xxd -p 2>/dev/null || echo $RANDOM)"
fi

# Get current working directory if not provided
if [ -z "$CWD" ]; then
    CWD=$(pwd)
fi

PROJECT_NAME=$(get_project_name "$CWD")

log_debug "session-start: sessionId=$SESSION_ID, cwd=$CWD, project=$PROJECT_NAME"

# Build session start message
MESSAGE=$(cat << EOF
{
    "type": "session_start",
    "id": "$(generate_id)",
    "sessionId": "$SESSION_ID",
    "timestamp": $(timestamp_ms),
    "provider": "gemini",
    "payload": {
        "hostname": "$(hostname)",
        "cwd": "$CWD",
        "projectName": "$PROJECT_NAME",
        "provider": "gemini"
    }
}
EOF
)

# Send to bridge (fire and forget)
send_fire_and_forget "$MESSAGE"

log_debug "session-start: Message sent"

# Always allow session to start
exit 0
