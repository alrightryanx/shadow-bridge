#!/usr/bin/env bash
# Gemini Before Tool Hook
# Intercepts tool execution (permission-like behavior)

SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Read hook input
HOOK_INPUT=$(cat)

# Extract fields
SESSION_ID=$(echo "$HOOK_INPUT" | grep -oP '"session_id"' | grep -oP '"[^"]*"' | cut -d'"' -f2)
TOOL_NAME=$(echo "$HOOK_INPUT" | grep -oP '"tool_name"' | grep -oP '"[^"]*"' | cut -d'"' -f2)
TOOL_INPUT=$(echo "$HOOK_INPUT" | sed 's/.*"tool_input"://')

# Generate approval ID
APPROVAL_ID="approval_$(date +%Y%m%d%H%M%S)_$(uuidgen | head -c 8)"

echo "Before tool: tool=$TOOL_NAME"

# Build approval request message (similar to Claude PermissionRequest)
MESSAGE="{"type":"approval_request","id":"msg_$(date +%s%N)","sessionId":"$SESSION_ID","timestamp":$(date +%s)000,"payload":{"approvalId":"$APPROVAL_ID","toolName":"$TOOL_NAME","toolInput":$TOOL_INPUT,"prompt":"Tool execution permission","promptType":"PERMISSION","options":["Approve","Deny","Reply"],"allowReply":true}}"

# Send to ShadowBridge and wait for response
RESPONSE=$(echo "$MESSAGE" | nc -q -w 60 localhost 19286)

# Process response
if echo "$RESPONSE" | grep -q '"approved":true'; then
    # Approved
    OUTPUT='{"hookSpecificOutput":{"hookEventName":"BeforeTool","decision":{"behavior":"allow","message":"Approved from ShadowAI"}}}'
    echo "$OUTPUT"
    exit 0
elif echo "$RESPONSE" | grep -q '"approved":false'; then
    # Denied
    OUTPUT='{"hookSpecificOutput":{"hookEventName":"BeforeTool","decision":{"behavior":"deny","message":"Denied from ShadowAI"}}}'
    echo "$OUTPUT"
    exit 0
else
    # Timeout - dismiss
    DISMISS="{"type":"approval_dismiss","sessionId":"$SESSION_ID","payload":{"approvalId":"$APPROVAL_ID","reason":"timeout"}}"
    echo "$DISMISS" | nc -q -w 2 localhost 19286
    exit 0
fi
