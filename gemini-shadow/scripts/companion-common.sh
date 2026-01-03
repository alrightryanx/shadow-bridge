#!/usr/bin/env bash
# Gemini Shadow - Common Functions
# Shared utilities for all hook scripts
# Converted from claude-shadow PowerShell to Bash

set -e

# Configuration
BRIDGE_PORT=19286
BRIDGE_HOST="127.0.0.1"
CONFIG_FILE="$HOME/.gemini-shadow-config.json"
DEBUG_LOG="$HOME/.gemini-shadow-debug.log"
PROVIDER="gemini"

# Load configuration
load_config() {
    if [ -f "$CONFIG_FILE" ]; then
        BRIDGE_HOST=$(jq -r '.bridgeHost // "127.0.0.1"' "$CONFIG_FILE" 2>/dev/null || echo "127.0.0.1")
        BRIDGE_PORT=$(jq -r '.bridgePort // 19286' "$CONFIG_FILE" 2>/dev/null || echo "19286")
    fi
}

# Log message to debug file
log_debug() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $1" >> "$DEBUG_LOG"
}

# Generate unique message ID
generate_id() {
    echo "msg_$(date +%s)_$(head -c 4 /dev/urandom | xxd -p)"
}

# Get current timestamp in milliseconds
timestamp_ms() {
    echo $(($(date +%s) * 1000))
}

# Send length-prefixed JSON message via TCP
# Uses Python for reliable binary protocol handling
send_to_bridge() {
    local message="$1"
    local timeout="${2:-60}"

    load_config

    log_debug "send_to_bridge: Connecting to $BRIDGE_HOST:$BRIDGE_PORT"
    log_debug "send_to_bridge: Message type=$(echo "$message" | jq -r '.type')"

    # Use Python for TCP communication (more reliable than nc/bash for binary protocol)
    local response
    response=$(python3 - "$BRIDGE_HOST" "$BRIDGE_PORT" "$message" "$timeout" << 'PYTHON_SCRIPT'
import sys
import socket
import json
import struct

def main():
    host = sys.argv[1]
    port = int(sys.argv[2])
    message = json.loads(sys.argv[3])
    timeout = int(sys.argv[4])

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))

        # Send handshake
        handshake = json.dumps({"type": "handshake"}).encode('utf-8')
        sock.sendall(struct.pack('>I', len(handshake)) + handshake)

        # Read handshake ack
        length_bytes = sock.recv(4)
        if len(length_bytes) < 4:
            print('{"error": "handshake_failed"}')
            return
        length = struct.unpack('>I', length_bytes)[0]
        ack_data = sock.recv(length)
        ack = json.loads(ack_data.decode('utf-8'))

        if ack.get('type') != 'handshake_ack':
            print('{"error": "handshake_rejected"}')
            return

        # Send actual message
        msg_bytes = json.dumps(message).encode('utf-8')
        sock.sendall(struct.pack('>I', len(msg_bytes)) + msg_bytes)

        # Read response
        length_bytes = sock.recv(4)
        if len(length_bytes) >= 4:
            length = struct.unpack('>I', length_bytes)[0]
            response_data = sock.recv(length)
            print(response_data.decode('utf-8'))
        else:
            print('{"type": "ack"}')

        sock.close()
    except socket.timeout:
        print('{"error": "timeout"}')
    except ConnectionRefusedError:
        print('{"error": "connection_refused"}')
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == '__main__':
    main()
PYTHON_SCRIPT
    ) 2>/dev/null

    if [ -n "$response" ]; then
        log_debug "send_to_bridge: Got response: $response"
        echo "$response"
    else
        log_debug "send_to_bridge: No response received"
        echo '{"error": "no_response"}'
    fi
}

# Send message without waiting for response (fire and forget)
send_fire_and_forget() {
    local message="$1"
    send_to_bridge "$message" 5 > /dev/null 2>&1 &
}

# Read hook input from stdin (JSON)
read_hook_input() {
    local input
    input=$(cat)

    if [ -z "$input" ]; then
        log_debug "read_hook_input: Empty input"
        echo "{}"
        return 1
    fi

    log_debug "read_hook_input: Got ${#input} chars"
    echo "$input"
}

# Write hook output to stdout (JSON)
write_hook_output() {
    echo "$1"
}

# Get friendly tool description (maps Gemini tools to readable names)
get_friendly_tool_description() {
    local tool_name="$1"
    local tool_input="$2"

    case "$tool_name" in
        run_shell_command)
            local cmd=$(echo "$tool_input" | jq -r '.command // ""' 2>/dev/null)
            if [ ${#cmd} -gt 80 ]; then
                cmd="${cmd:0:80}..."
            fi
            echo "Run: $cmd"
            ;;
        read_file)
            local path=$(echo "$tool_input" | jq -r '.file_path // .path // ""' 2>/dev/null)
            local short_path=$(basename "$path")
            echo "Read: $short_path"
            ;;
        write_file)
            local path=$(echo "$tool_input" | jq -r '.file_path // .path // ""' 2>/dev/null)
            local short_path=$(basename "$path")
            echo "Create: $short_path"
            ;;
        replace)
            local path=$(echo "$tool_input" | jq -r '.file_path // .path // ""' 2>/dev/null)
            local short_path=$(basename "$path")
            echo "Edit: $short_path"
            ;;
        glob)
            local pattern=$(echo "$tool_input" | jq -r '.pattern // ""' 2>/dev/null)
            echo "Find files: $pattern"
            ;;
        search_file_content)
            local pattern=$(echo "$tool_input" | jq -r '.pattern // .query // ""' 2>/dev/null)
            if [ ${#pattern} -gt 40 ]; then
                pattern="${pattern:0:40}..."
            fi
            echo "Search: $pattern"
            ;;
        *)
            echo "$tool_name"
            ;;
    esac
}

# Extract project name from path
get_project_name() {
    local cwd="$1"
    if [ -n "$cwd" ]; then
        basename "$cwd" | sed 's/-main$//' | sed 's/-master$//' | sed 's/-dev$//'
    fi
}
