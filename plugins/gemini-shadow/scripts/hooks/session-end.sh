#!/usr/bin/env bash
# Gemini Session End Hook
# Sends session end notification to ShadowBridge

SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Read hook input
HOOK_INPUT=$(cat)

# Extract session_id
SESSION_ID=$(echo "$HOOK_INPUT" | grep -oP '"session_id"' | grep -oP '"[^"]*"' | cut -d'"' -f2)

echo "Session end: sessionId=$SESSION_ID"

# Build session end message
MESSAGE="{"type":"session_end","id":"msg_$(date +%s%N)","sessionId":"$SESSION_ID","timestamp":$(date +%s)000,"payload":{"hostname":"$(hostname)"}}"

# Send to ShadowBridge
echo "$MESSAGE" | nc -q localhost 19286

exit 0
