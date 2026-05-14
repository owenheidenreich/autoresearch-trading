#!/usr/bin/env bash
set -euo pipefail
launchctl bootout "gui/$UID/com.autoresearch.ibgateway.paper" 2>/dev/null || true
launchctl bootout "gui/$UID/com.autoresearch.protocol101.paper-preflight" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/com.autoresearch.ibgateway.paper.plist"
rm -f "$HOME/Library/LaunchAgents/com.autoresearch.protocol101.paper-preflight.plist"
echo "Removed IB Gateway paper autostart LaunchAgents."
