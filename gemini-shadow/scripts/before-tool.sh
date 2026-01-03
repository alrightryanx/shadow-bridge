#!/usr/bin/env bash
# Gemini Shadow - Before Tool Hook (Permission Request equivalent)
# Sends tool execution requests to ShadowBridge for approval

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/companion-common.sh"

# Read hook input from stdin
HOOK_INPUT=$(read_hook_input)

log_debug "before-tool: Hook invoked"
log_debug "before-tool: Input: $HOOK_INPUT"

# Extract tool info from Gemini CLI hook data
SESSION_ID=$(echo "$HOOK_INPUT" | jq -r '.session_id // .sessionId // ""' 2>/dev/null)
TOOL_NAME=$(echo "$HOOK_INPUT" | jq -r '.tool_name // .toolName // .name // ""' 2>/dev/null)
TOOL_INPUT=$(echo "$HOOK_INPUT" | jq -c '.tool_input // .toolInput // .input // {}' 2>/dev/null)
CWD=$(echo "$HOOK_INPUT" | jq -r '.cwd // ""' 2>/dev/null)

# Generate request ID
REQUEST_ID="req_$(date +%s)_$(head -c 4 /dev/urandom | xxd -p 2>/dev/null || echo $RANDOM)"

# Get friendly tool description
TOOL_DESCRIPTION=$(get_friendly_tool_description "$TOOL_NAME" "$TOOL_INPUT")

# Get project name
PROJECT_NAME=""
if [ -n "$CWD" ]; then
    PROJECT_NAME=$(get_project_name "$CWD")
fi

log_debug "before-tool: tool=$TOOL_NAME, desc=$TOOL_DESCRIPTION, session=$SESSION_ID"

# Check if this is a dangerous tool that needs approval
NEEDS_APPROVAL=false
case "$TOOL_NAME" in
    run_shell_command|write_file|replace)
        NEEDS_APPROVAL=true
        ;;
esac

if [ "$NEEDS_APPROVAL" = "true" ]; then
    # Build permission request message (escape for JSON)
    ESCAPED_DESC=$(echo "$TOOL_DESCRIPTION" | jq -Rs '.' | sed 's/^"//;s/"$//')

    MESSAGE=$(cat << EOF
{
    "type": "permission_request",
    "id": "$(generate_id)",
    "sessionId": "$SESSION_ID",
    "timestamp": $(timestamp_ms),
    "provider": "gemini",
    "payload": {
        "requestId": "$REQUEST_ID",
        "toolName": "$TOOL_NAME",
        "toolInput": $TOOL_INPUT,
        "description": "$ESCAPED_DESC",
        "hostname": "$(hostname)",
        "cwd": "$CWD",
        "projectName": "$PROJECT_NAME",
        "options": ["Approve", "Deny", "Always Allow"],
        "promptType": "PERMISSION",
        "provider": "gemini"
    }
}
EOF
    )

    # Send to bridge and wait for response
    RESPONSE=$(send_to_bridge "$MESSAGE" 60)

    log_debug "before-tool: Got response: $RESPONSE"

    # Check response
    RESPONSE_TYPE=$(echo "$RESPONSE" | jq -r '.type // ""' 2>/dev/null)
    APPROVED=$(echo "$RESPONSE" | jq -r '.approved // .allow // false' 2>/dev/null)

    if [ "$RESPONSE_TYPE" = "approval_response" ]; then
        if [ "$APPROVED" = "true" ]; then
            log_debug "before-tool: Tool approved"
            # Output approval for Gemini CLI
            echo '{"continue": true}'
            exit 0
        else
            log_debug "before-tool: Tool denied"
            echo '{"continue": false, "reason": "User denied permission"}'
            exit 0
        fi
    fi
fi

# Default: allow tool execution
log_debug "before-tool: Auto-allowing tool (no approval needed or no response)"
echo '{"continue": true}'
exit 0
