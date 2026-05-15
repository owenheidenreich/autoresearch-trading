#!/usr/bin/env bash
set -euo pipefail
RUNTIME_DIR="$HOME/.autoresearch-trading/launchd"
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs/autoresearch-trading" "$RUNTIME_DIR"
cp "/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/install_ibc_macos.sh" "$RUNTIME_DIR/"
cp "/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/start_ib_gateway_paper_ibc.sh" "$RUNTIME_DIR/"
cp "/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/run_protocol101_paper_preflight.sh" "$RUNTIME_DIR/"
cp "/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/wait_for_ibkr_api.py" "$RUNTIME_DIR/"
cp "/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/write_ibc_runtime_config.py" "$RUNTIME_DIR/"
cp "/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/probe_ibkr_api.py" "$RUNTIME_DIR/"
chmod 700 "$RUNTIME_DIR"/*.sh
chmod 600 "$RUNTIME_DIR"/*.py
launchctl bootout "gui/$UID/com.autoresearch.ibgateway.paper" 2>/dev/null || true
launchctl bootout "gui/$UID/com.autoresearch.protocol101.paper-preflight" 2>/dev/null || true
cp "v4/ops/launchd/com.autoresearch.ibgateway.paper.plist" "$HOME/Library/LaunchAgents/"
cp "v4/ops/launchd/com.autoresearch.protocol101.paper-preflight.plist" "$HOME/Library/LaunchAgents/"
launchctl bootstrap "gui/$UID" "$HOME/Library/LaunchAgents/com.autoresearch.ibgateway.paper.plist"
launchctl bootstrap "gui/$UID" "$HOME/Library/LaunchAgents/com.autoresearch.protocol101.paper-preflight.plist"
launchctl enable "gui/$UID/com.autoresearch.ibgateway.paper"
launchctl enable "gui/$UID/com.autoresearch.protocol101.paper-preflight"
echo "Installed IB Gateway paper autostart and Protocol101 preflight LaunchAgents."
