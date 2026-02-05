#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "Starting ShadowBridge Linux build and packaging..."

if [ ! -x "./build_linux.sh" ]; then
    echo "build_linux.sh is not executable. Making it executable..."
    chmod +x ./build_linux.sh
fi

./build_linux.sh

if [ ! -d "dist/ShadowBridge" ]; then
    echo "Build output not found at dist/ShadowBridge"
    exit 1
fi

PACKAGE_NAME="ShadowBridge-linux-$(date +%Y%m%d-%H%M%S).tar.gz"
PACKAGE_PATH="dist/${PACKAGE_NAME}"

echo "Creating package ${PACKAGE_NAME}..."
tar -czf "${PACKAGE_PATH}" -C dist/ShadowBridge .

echo "Linux package ready: ${PACKAGE_PATH}"
