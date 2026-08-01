#!/usr/bin/env bash
set -euo pipefail
launchctl bootout "gui/$UID/com.autoresearch.ibgateway.paper" 2>/dev/null || true
launchctl bootout "gui/$UID/com.autoresearch.protocol101.paper-preflight" 2>/dev/null || true
launchctl bootout "gui/$UID/com.autoresearch.protocol101.paper-session" 2>/dev/null || true
launchctl bootout "gui/$UID/com.autoresearch.protocol101.daily-monitor" 2>/dev/null || true
launchctl bootout "gui/$UID/com.autoresearch.premiumblend.no-order-surface-check" 2>/dev/null || true
launchctl bootout "gui/$UID/com.autoresearch.ibgateway.paper-shutdown" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/com.autoresearch.ibgateway.paper.plist"
rm -f "$HOME/Library/LaunchAgents/com.autoresearch.protocol101.paper-preflight.plist"
rm -f "$HOME/Library/LaunchAgents/com.autoresearch.protocol101.paper-session.plist"
rm -f "$HOME/Library/LaunchAgents/com.autoresearch.protocol101.daily-monitor.plist"
rm -f "$HOME/Library/LaunchAgents/com.autoresearch.premiumblend.no-order-surface-check.plist"
rm -f "$HOME/Library/LaunchAgents/com.autoresearch.ibgateway.paper-shutdown.plist"
echo "Removed IB Gateway paper autostart LaunchAgents."
