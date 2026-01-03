#!/usr/bin/env bash
# Gemini Shadow - Before Agent Hook (User Prompt Submit equivalent)
# Notifies ShadowBridge when user submits a prompt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/companion-common.sh"

# Read hook input from stdin
HOOK_INPUT=$(read_hook_input)

log_debug "before-agent: Hook invoked"
log_debug "before-agent: Input: $HOOK_INPUT"

# Extract prompt info
SESSION_ID=$(echo "$HOOK_INPUT" | jq -r '.session_id // .sessionId // ""' 2>/dev/null)
PROMPT=$(echo "$HOOK_INPUT" | jq -r '.prompt // .message // .content // ""' 2>/dev/null)
CWD=$(echo "$HOOK_INPUT" | jq -r '.cwd // ""' 2>/dev/null)

# Generate message ID
MESSAGE_ID=$(generate_id)

# Get project name
PROJECT_NAME=""
if [ -n "$CWD" ]; then
    PROJECT_NAME=$(get_project_name "$CWD")
fi

# Create preview of prompt (first 100 chars)
PROMPT_PREVIEW="$PROMPT"
if [ ${#PROMPT_PREVIEW} -gt 100 ]; then
    PROMPT_PREVIEW="${PROMPT_PREVIEW:0:100}..."
fi

log_debug "before-agent: session=$SESSION_ID, prompt_preview=$PROMPT_PREVIEW"

# Escape for JSON
ESCAPED_PROMPT=$(echo "$PROMPT" | jq -Rs '.' | sed 's/^"//;s/"$//')
ESCAPED_PREVIEW=$(echo "$PROMPT_PREVIEW" | jq -Rs '.' | sed 's/^"//;s/"$//')

# Build user prompt message
MESSAGE=$(cat << EOF
{
    "type": "user_prompt",
    "id": "$MESSAGE_ID",
    "sessionId": "$SESSION_ID",
    "timestamp": $(timestamp_ms),
    "provider": "gemini",
    "payload": {
        "prompt": "$ESCAPED_PROMPT",
        "promptPreview": "$ESCAPED_PREVIEW",
        "hostname": "$(hostname)",
        "cwd": "$CWD",
        "projectName": "$PROJECT_NAME",
        "provider": "gemini"
    }
}
EOF
)

# Send to bridge (fire and forget - don't block user input)
send_fire_and_forget "$MESSAGE"

# Cache context for notifications
CONTEXT_CACHE="$HOME/.gemini-shadow-context.json"
cat > "$CONTEXT_CACHE" << EOF
{
    "timestamp": $(date +%s),
    "lastPromptPreview": "$ESCAPED_PREVIEW",
    "sessionId": "$SESSION_ID",
    "projectName": "$PROJECT_NAME"
}
EOF

log_debug "before-agent: Message sent, context cached"

# Always allow prompt to proceed
exit 0
