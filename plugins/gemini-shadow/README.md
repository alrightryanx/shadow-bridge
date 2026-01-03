# gemini-shadow

Gemini CLI session persistence plugin for ShadowAI

## Installation

### Option 1: Auto-Migration from Claude (Recommended)

If you have `claude-shadow` installed, use Gemini CLI's built-in migration:

```bash
gemini hooks migrate --from-claude
```

This automatically:
- Reads your `.claude/hooks.json`
- Converts event names (Claude → Gemini)
- Translates tool names
- Writes to `.gemini/settings.json`

### Option 2: Manual Installation

1. Copy `gemini-shadow` directory to your project:
   ```bash
   cp -r gemini-shadow/ /path/to/project/.gemini/
   ```

2. Verify hooks are registered:
   ```bash
   gemini hooks panel
   ```

## Features

- **Session Lifecycle**: SessionStart, SessionEnd
- **Agent Events**: BeforeAgent, AfterAgent
- **Model Events**: BeforeModel, AfterModel
- **Tool Events**: BeforeTool, AfterTool
- **Permission Requests**: Tools requiring approval
- **Conversation Sync**: User prompts sent to phone
- **Mobile Approval**: Approve/deny from phone
- **Reply via Phone**: Clipboard sync

## Configuration

### ShadowBridge Connection

The plugin communicates with ShadowBridge on port 19286. Edit plugin config:

```json
{
  "bridgeHost": "127.0.0.1",
  "bridgePort": 19286,
  "enabled": true
}
```

### Gemini Settings

Edit `.gemini/settings.json` to customize hooks:

```json
{
  "hooks": {
    "SessionStart": [{
      "command": "shadow-bridge/plugins/gemini-shadow/scripts/hooks/session-start.sh",
      "timeout": 30000
    }]
  }
}
```

## Troubleshooting

### Hooks not triggering

1. Verify file permissions: `chmod +x scripts/hooks/*.sh`
2. Check Gemini CLI version: `gemini --version`
3. Test with debug logging: `gemini --debug`
4. Review debug log: `~/.gemini-shadow-debug.log`

### Connection issues

- Ensure ShadowBridge is running: `python shadow_bridge_gui.py`
- Check firewall allows connections to port 19286
- Verify host/port in configuration

## Debugging

Enable debug logging by checking log file:

- `~/.gemini-shadow-debug.log` - Contains detailed operation logs
- Review for hook invocations, message sends, and connection issues

## License

MIT License - See LICENSE file for details

## Support

- [ShadowAI Website](https://ryancartwright.com/shadowai)
- [Gemini CLI Docs](https://geminicli.com/docs/hooks/)
- [GitHub Issues](https://github.com/alrightryanx/shadowai-plugins/issues)
