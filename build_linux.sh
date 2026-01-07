#!/bin/bash
set -e

echo "Building ShadowBridge for Linux..."

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found."
    exit 1
fi

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt
pip install pyinstaller

# Build
echo "Running PyInstaller..."
pyinstaller ShadowBridge.spec

echo "Build complete. Executable is located at dist/ShadowBridge/ShadowBridge"
