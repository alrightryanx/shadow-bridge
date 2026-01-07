# ShadowBridge for Linux

## Prerequisites
- Python 3.10 or higher
- `python3-venv` package (on Debian/Ubuntu: `sudo apt install python3-venv`)

## Building
1. Open a terminal in this directory.
2. Make the script executable: `chmod +x build_linux.sh`
3. Run the build script: `./build_linux.sh`

The executable will be generated in `dist/ShadowBridge/ShadowBridge`.

## Troubleshooting
If the app fails to launch, you might be missing Qt system dependencies.
On Ubuntu/Debian:
```bash
sudo apt install libxcb-cursor0 libxcb-xinerama0
```
