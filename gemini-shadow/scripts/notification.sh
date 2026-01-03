#!/usr/bin/env bash
# Gemini Shadow - Notification Hook
# Forwards Gemini CLI notifications to ShadowBridge

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/companion-common.sh"

# Read hook input from stdin
HOOK_INPUT=$(read_hook_input)

log_debug "notification: Hook invoked"
log_debug "notification: Input: $HOOK_INPUT"

# Extract notification info
SESSION_ID=$(echo "$HOOK_INPUT" | jq -r '.session_id // .sessionId // ""' 2>/dev/null)
NOTIFICATION_MESSAGE=$(echo "$HOOK_INPUT" | jq -r '.message // ""' 2>/dev/null)
NOTIFICATION_TYPE=$(echo "$HOOK_INPUT" | jq -r '.notification_type // .type // "info"' 2>/dev/null)
CWD=$(echo "$HOOK_INPUT" | jq -r '.cwd // ""' 2>/dev/null)

# Generate notification ID
NOTIFICATION_ID="notif_$(date +%s)_$(head -c 4 /dev/urandom | xxd -p 2>/dev/null || echo $RANDOM)"

# Get project name
PROJECT_NAME=""
if [ -n "$CWD" ]; then
    PROJECT_NAME=$(get_project_name "$CWD")
fi

# Build summary based on notification content
SUMMARY="Gemini"
if [ -n "$PROJECT_NAME" ]; then
    SUMMARY="Gemini [$PROJECT_NAME]"
fi

# Detect notification type for better summary
if echo "$NOTIFICATION_MESSAGE" | grep -qiE "error|exception|fail"; then
    SUMMARY="Error"
    [ -n "$PROJECT_NAME" ] && SUMMARY="Error [$PROJECT_NAME]"
elif echo "$NOTIFICATION_MESSAGE" | grep -qiE "complet|finish|done|success"; then
    SUMMARY="Complete"
    [ -n "$PROJECT_NAME" ] && SUMMARY="Complete [$PROJECT_NAME]"
elif echo "$NOTIFICATION_MESSAGE" | grep -qiE "waiting.*input|user.*input|need.*input"; then
    SUMMARY="Input Needed"
    [ -n "$PROJECT_NAME" ] && SUMMARY="Input Needed [$PROJECT_NAME]"
fi

log_debug "notification: sessionId=$SESSION_ID, summary=$SUMMARY"

# Build notification message (escape special chars for JSON)
ESCAPED_MESSAGE=$(echo "$NOTIFICATION_MESSAGE" | jq -Rs '.' | sed 's/^"//;s/"$//')
ESCAPED_SUMMARY=$(echo "$SUMMARY" | jq -Rs '.' | sed 's/^"//;s/"$//')

MESSAGE=$(cat << EOF
{
    "type": "notification",
    "id": "$(generate_id)",
    "sessionId": "$SESSION_ID",
    "timestamp": $(timestamp_ms),
    "provider": "gemini",
    "payload": {
        "notificationId": "$NOTIFICATION_ID",
        "message": "$ESCAPED_MESSAGE",
        "summary": "$ESCAPED_SUMMARY",
        "originalMessage": "$ESCAPED_MESSAGE",
        "notificationType": "$NOTIFICATION_TYPE",
        "hostname": "$(hostname)",
        "cwd": "$CWD",
        "projectName": "$PROJECT_NAME",
        "options": ["Reply", "Dismiss"],
        "promptType": "NOTIFICATION",
        "allowReply": true,
        "provider": "gemini"
    }
}
EOF
)

# Send to bridge (fire and forget - notifications don't wait)
send_fire_and_forget "$MESSAGE"

log_debug "notification: Message sent"

# Always allow notification to proceed
exit 0
