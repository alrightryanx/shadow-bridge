#!/usr/bin/env bash
# Gemini Session Start Hook
# Sends session start notification to ShadowBridge

SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Read hook input
HOOK_INPUT=$(cat)

# Extract session_id
SESSION_ID=$(echo "$HOOK_INPUT" | grep -oP '"session_id"' | grep -oP '"[^"]*"' | cut -d'"' -f2)

echo "Session start: sessionId=$SESSION_ID"

# Build session start message
MESSAGE="{"type":"session_start","id":"msg_$(date +%s%N)","sessionId":"$SESSION_ID","timestamp":$(date +%s)000,"payload":{"hostname":"$(hostname)","provider":"gemini-cli","model":"$(echo "$HOOK_INPUT" | grep -oP '"model"' | grep -oP '"[^"]*"' | head -1)"}}"

# Send to ShadowBridge
echo "$MESSAGE" | nc -q localhost 19286

exit 0
