#!/usr/bin/env bash
# Gemini Before Agent Hook
# Captures user prompts before agent processing

SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Read hook input
HOOK_INPUT=$(cat)

# Extract fields
PROMPT=$(echo "$HOOK_INPUT" | grep -oP '"prompt"' | grep -oP '"[^"]*"' | cut -d'"' -f2)
SESSION_ID=$(echo "$HOOK_INPUT" | grep -oP '"session_id"' | grep -oP '"[^"]*"' | cut -d'"' -f2)

echo "Before agent: prompt=${PROMPT:0:50}..."

# Send as session message (user prompt)
MESSAGE="{"type":"session_message","id":"msg_$(date +%s%N)","sessionId":"$SESSION_ID","timestamp":$(date +%s)000,"payload":{"role":"user","content":"$PROMPT","hostname":"$(hostname)"}}"

# Send to ShadowBridge (fire and forget)
echo "$MESSAGE" | nc -q -w 5 localhost 19286

exit 0
