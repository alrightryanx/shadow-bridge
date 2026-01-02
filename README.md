# ShadowBridge

<p align="center">
  <img src="images/shadow-bridge.png" alt="ShadowBridge" width="400"/>
</p>

Secure bridge between your Android device and PC for ShadowAI Android app - enables clipboard sync, SSH key exchange, and Claude Code companion relay.

---

## Shadow Web Dashboard

<p align="center">
  <img src="images/shadow-web.png" alt="Shadow Web Dashboard" width="600"/>
</p>

ShadowBridge includes a web dashboard at `http://localhost:6767` for managing projects, notes, and team collaboration.

---

## Features

- **Clipboard Sync**: Share clipboard between PC and phone
- **SSH Key Exchange**: Quick Connect for ShadowAI SSH authentication
- **Claude Code Companion**: Relay notifications and approvals between Claude Code and ShadowAI
- **System Tray**: Runs quietly in the background

## Requirements

- Windows 10/11
- Python 3.8+
- [ShadowAI](https://play.google.com/store/apps/details?id=com.shadowai.release) on your Android device

## Installation

```bash
pip install -r requirements.txt
python shadow_bridge_gui.py
```

## Ports

| Port | Purpose |
|------|---------|
| 19284 | Data receiver (clipboard, keys) |
| 19285 | Discovery broadcast |
| 19286 | Claude Code Companion relay |

## Usage

1. Run ShadowBridge on your PC
2. Open ShadowAI on your phone
3. Use Quick Connect to pair (scan QR or network discovery)
4. Done! Clipboard sync and SSH keys are now shared

## Claude Code Companion

ShadowBridge includes a relay server for the [claude-shadow](https://github.com/alrightryanx/claude-shadow) plugin:

- Receives approval requests from Claude Code
- Forwards to ShadowAI on your phone
- Returns approval/denial responses
- Syncs replies to PC clipboard

## Privacy

- All communication is direct between your devices (local network or via [Tailscale](https://tailscale.com))
- No data is sent to external servers
- Works across any network with Tailscale VPN

## License

MIT License - See [LICENSE](LICENSE) for details.

## Links

- [ShadowAI Website](https://ryancartwright.com/shadowai)
- [Claude Shadow Plugin](https://github.com/alrightryanx/claude-shadow)
