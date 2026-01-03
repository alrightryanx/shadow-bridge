#!/usr/bin/env bash
# Gemini Shadow - After Tool Hook
# Notifies ShadowBridge after tool execution completes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/companion-common.sh"

# Read hook input from stdin
HOOK_INPUT=$(read_hook_input)

log_debug "after-tool: Hook invoked"
log_debug "after-tool: Input: $HOOK_INPUT"

# Extract tool result info
SESSION_ID=$(echo "$HOOK_INPUT" | jq -r '.session_id // .sessionId // ""' 2>/dev/null)
TOOL_NAME=$(echo "$HOOK_INPUT" | jq -r '.tool_name // .toolName // .name // ""' 2>/dev/null)
TOOL_OUTPUT=$(echo "$HOOK_INPUT" | jq -c '.tool_output // .output // .result // {}' 2>/dev/null)
SUCCESS=$(echo "$HOOK_INPUT" | jq -r '.success // true' 2>/dev/null)
ERROR=$(echo "$HOOK_INPUT" | jq -r '.error // ""' 2>/dev/null)
CWD=$(echo "$HOOK_INPUT" | jq -r '.cwd // ""' 2>/dev/null)

# Get friendly tool description
TOOL_DESCRIPTION=$(get_friendly_tool_description "$TOOL_NAME" "$TOOL_OUTPUT")

log_debug "after-tool: tool=$TOOL_NAME, success=$SUCCESS"

# Update context cache with last tool
CONTEXT_CACHE="$HOME/.gemini-shadow-context.json"
if [ -f "$CONTEXT_CACHE" ]; then
    # Update existing cache
    EXISTING=$(cat "$CONTEXT_CACHE")
    ESCAPED_DESC=$(echo "$TOOL_DESCRIPTION" | jq -Rs '.' | sed 's/^"//;s/"$//')
    echo "$EXISTING" | jq --arg tool "$ESCAPED_DESC" '.lastTool = $tool' > "$CONTEXT_CACHE" 2>/dev/null || true
fi

# Build tool result message (for tracking/logging purposes)
MESSAGE=$(cat << EOF
{
    "type": "tool_result",
    "id": "$(generate_id)",
    "sessionId": "$SESSION_ID",
    "timestamp": $(timestamp_ms),
    "provider": "gemini",
    "payload": {
        "toolName": "$TOOL_NAME",
        "success": $SUCCESS,
        "error": "$ERROR",
        "hostname": "$(hostname)",
        "cwd": "$CWD",
        "provider": "gemini"
    }
}
EOF
)

# Send to bridge (fire and forget)
send_fire_and_forget "$MESSAGE"

log_debug "after-tool: Message sent"

# Always continue
exit 0
