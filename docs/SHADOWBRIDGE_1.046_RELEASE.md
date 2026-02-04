# ShadowBridge v1.046 Release Notes

## New MSI Installer
**Location:** `C:\shadow\shadow-bridge\dist\ShadowBridge-1.46-win64.msi`

## Fixed Issues

### 1. MSI Error 2810 (v1.044)
**Problem:** Windows Installer dialog control flow error prevented installation from completing.

**Root Cause:**
- Malformed `AIInstallCheckbox` control with incorrect parameter count
- `InstallAIDeps` custom action tried to execute with command-line arguments (not supported in MSI)
- Invalid `ControlEvent` referencing broken action

**Fix:**
- Removed malformed dialog controls
- Simplified to only launch ShadowBridge after installation
- Fixed custom action type for proper async app launch

---

### 2. Web Dashboard Crash Loop (v1.045)
**Problem:** Web dashboard failed to start, showing repeated TypeError crashes:
```
TypeError: DiscoveryServer.__init__() got an unexpected keyword argument 'ssh_port'
```

**Root Cause:**
The `run_web_dashboard_server` function was calling:
```python
discovery_server = DiscoveryServer(ssh_port=ssh_port)
```

But `DiscoveryServer.__init__` expects a `connection_info` dict, not `ssh_port` parameter.

**Fix:**
- Build complete `connection_info` dict with network details (tailscale_ip, local_ip, hosts_to_try, encryption_salt)
- Pass `connection_info` to `DiscoveryServer` instead of ssh_port
- Web dashboard now starts successfully with all services operational

---

### 3. Messy Connection Banner Header (v1.046)
**Problem:** Redundant backend/model information displayed twice in the header:
```
Backend [Gemini CLI (SSH)] Model [Gemini 1.5 Pro]
Gemini CLI (SSH) · Gemini 1.5 Pro ← redundant!
No active session · Bridge online
```

**Fix:**
- Hide `connection-selection-hint` element (display: none in CSS)
- Keep clean header with just dropdowns and health status
- Result:
  ```
  Backend [dropdown] Model [dropdown]
  No active session · Bridge online
  ```

---

## Installation Instructions

1. **Uninstall old version** (if present):
   - Press `Win + R`, type `appwiz.cpl`
   - Find "ShadowBridge" and click "Uninstall"

2. **Install v1.046**:
   - Double-click: `C:\shadow\shadow-bridge\dist\ShadowBridge-1.46-win64.msi`
   - Follow installation wizard
   - ShadowBridge will launch automatically after install

3. **Verify**:
   - Web dashboard should be accessible at http://localhost:6767
   - No error 2810 at end of installation
   - No TypeError crashes in logs
   - Clean header without duplicate backend/model info

---

## What's Working Now

✅ MSI installer completes without errors
✅ ShadowBridge launches successfully
✅ Web dashboard starts and serves pages
✅ Discovery server runs with proper connection_info
✅ Data receiver operational on port 19284
✅ Clean UI without redundant labels
✅ All services initialized correctly

---

## Log Files

If you encounter issues, check these log files:
- Main log: `%USERPROFILE%\.shadowai\shadowbridge.log`
- Web log: `%USERPROFILE%\.shadowai\shadowbridge_web.log`

---

## Commits

- **ec06872** - fix: resolve MSI error 2810 by removing malformed dialog controls
- **4a136bb** - fix: resolve DiscoveryServer TypeError in web dashboard mode
- **41d0b21** - fix: clean up messy connection banner header

---

## Next Steps

The system is now fully operational. You can:
1. Connect Android devices via Quick Connect
2. Access the web dashboard at http://localhost:6767
3. Manage projects, notes, and sessions
4. Monitor telemetry data (Ouroboros system)
5. Generate images, audio, and video through the dashboard
