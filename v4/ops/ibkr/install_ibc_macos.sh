#!/usr/bin/env bash
set -euo pipefail

IBC_VERSION="${IBC_VERSION:-3.23.0}"
IBC_URL="${IBC_URL:-https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCMacos-${IBC_VERSION}.zip}"
BASE_DIR="${IBC_BASE_DIR:-$HOME/.autoresearch-trading/ibc}"
DOWNLOAD_DIR="${IBC_DOWNLOAD_DIR:-$HOME/.autoresearch-trading/downloads}"
ZIP_PATH="$DOWNLOAD_DIR/IBCMacos-${IBC_VERSION}.zip"
INSTALL_DIR="$BASE_DIR/${IBC_VERSION}"

mkdir -p "$DOWNLOAD_DIR" "$BASE_DIR"
if [[ ! -f "$ZIP_PATH" ]]; then
  curl -L "$IBC_URL" -o "$ZIP_PATH"
fi

mkdir -p "$INSTALL_DIR"
unzip -q -o "$ZIP_PATH" -d "$INSTALL_DIR"
chmod +x "$INSTALL_DIR/scripts/"*.sh "$INSTALL_DIR/"*.sh 2>/dev/null || true

echo "$INSTALL_DIR"
