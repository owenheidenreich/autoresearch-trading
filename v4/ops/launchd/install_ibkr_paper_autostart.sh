#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs/autoresearch-trading"
launchctl bootout "gui/$UID/com.autoresearch.ibgateway.paper" 2>/dev/null || true
launchctl bootout "gui/$UID/com.autoresearch.protocol101.paper-preflight" 2>/dev/null || true
cp "v4/ops/launchd/com.autoresearch.ibgateway.paper.plist" "$HOME/Library/LaunchAgents/"
cp "v4/ops/launchd/com.autoresearch.protocol101.paper-preflight.plist" "$HOME/Library/LaunchAgents/"
launchctl bootstrap "gui/$UID" "$HOME/Library/LaunchAgents/com.autoresearch.ibgateway.paper.plist"
launchctl bootstrap "gui/$UID" "$HOME/Library/LaunchAgents/com.autoresearch.protocol101.paper-preflight.plist"
launchctl enable "gui/$UID/com.autoresearch.ibgateway.paper"
launchctl enable "gui/$UID/com.autoresearch.protocol101.paper-preflight"
echo "Installed IB Gateway paper autostart and Protocol101 preflight LaunchAgents."
