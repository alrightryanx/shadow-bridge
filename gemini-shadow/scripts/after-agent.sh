#!/usr/bin/env bash
# Gemini Shadow - After Agent Hook
# Notifies ShadowBridge after agent turn completes (session stop equivalent)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/companion-common.sh"

# Read hook input from stdin
HOOK_INPUT=$(read_hook_input)

log_debug "after-agent: Hook invoked"
log_debug "after-agent: Input: $HOOK_INPUT"

# Extract agent result info
SESSION_ID=$(echo "$HOOK_INPUT" | jq -r '.session_id // .sessionId // ""' 2>/dev/null)
RESPONSE=$(echo "$HOOK_INPUT" | jq -r '.response // .message // .content // ""' 2>/dev/null)
STOP_REASON=$(echo "$HOOK_INPUT" | jq -r '.stop_reason // .stopReason // "complete"' 2>/dev/null)
CWD=$(echo "$HOOK_INPUT" | jq -r '.cwd // ""' 2>/dev/null)

# Get project name
PROJECT_NAME=""
if [ -n "$CWD" ]; then
    PROJECT_NAME=$(get_project_name "$CWD")
fi

# Create response preview
RESPONSE_PREVIEW="$RESPONSE"
if [ ${#RESPONSE_PREVIEW} -gt 200 ]; then
    RESPONSE_PREVIEW="${RESPONSE_PREVIEW:0:200}..."
fi

log_debug "after-agent: session=$SESSION_ID, stop_reason=$STOP_REASON"

# Escape for JSON
ESCAPED_PREVIEW=$(echo "$RESPONSE_PREVIEW" | jq -Rs '.' | sed 's/^"//;s/"$//')

# Build agent complete message
MESSAGE=$(cat << EOF
{
    "type": "agent_complete",
    "id": "$(generate_id)",
    "sessionId": "$SESSION_ID",
    "timestamp": $(timestamp_ms),
    "provider": "gemini",
    "payload": {
        "stopReason": "$STOP_REASON",
        "responsePreview": "$ESCAPED_PREVIEW",
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

log_debug "after-agent: Message sent"

# Always continue
exit 0
