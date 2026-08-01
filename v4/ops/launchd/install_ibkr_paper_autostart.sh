#!/usr/bin/env bash
set -euo pipefail
RUNTIME_DIR="$HOME/.autoresearch-trading/launchd"
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs/autoresearch-trading" "$RUNTIME_DIR"
cp "/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/install_ibc_macos.sh" "$RUNTIME_DIR/"
cp "/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/start_ib_gateway_paper_ibc.sh" "$RUNTIME_DIR/"
cp "/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/run_daily_paper_autopilot.sh" "$RUNTIME_DIR/"
cp "/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/run_protocol101_paper_preflight.sh" "$RUNTIME_DIR/"
cp "/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/run_protocol101_paper_session.sh" "$RUNTIME_DIR/"
cp "/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/run_protocol101_daily_monitor.sh" "$RUNTIME_DIR/"
cp "/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/run_premium_blend_live_surface_autotest.sh" "$RUNTIME_DIR/"
cp "/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/shutdown_ibkr_paper_stack.sh" "$RUNTIME_DIR/"
cp "/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/run_ibkr_autostart_status.sh" "$RUNTIME_DIR/"
cp "/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/wait_for_ibkr_api.py" "$RUNTIME_DIR/"
cp "/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/write_ibc_runtime_config.py" "$RUNTIME_DIR/"
cp "/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/probe_ibkr_api.py" "$RUNTIME_DIR/"
cp "/Users/gduby/Documents/autoresearch-trading/v4/ops/ibkr/warm_launchd_python_deps.py" "$RUNTIME_DIR/"
chmod 700 "$RUNTIME_DIR"/*.sh
chmod 600 "$RUNTIME_DIR"/*.py
launchctl bootout "gui/$UID/com.autoresearch.ibgateway.paper" 2>/dev/null || true
launchctl bootout "gui/$UID/com.autoresearch.protocol101.paper-preflight" 2>/dev/null || true
launchctl bootout "gui/$UID/com.autoresearch.protocol101.paper-session" 2>/dev/null || true
launchctl bootout "gui/$UID/com.autoresearch.protocol101.daily-monitor" 2>/dev/null || true
launchctl bootout "gui/$UID/com.autoresearch.premiumblend.no-order-surface-check" 2>/dev/null || true
launchctl bootout "gui/$UID/com.autoresearch.ibgateway.paper-shutdown" 2>/dev/null || true
cp "v4/ops/launchd/com.autoresearch.ibgateway.paper.plist" "$HOME/Library/LaunchAgents/"
cp "v4/ops/launchd/com.autoresearch.protocol101.paper-preflight.plist" "$HOME/Library/LaunchAgents/"
cp "v4/ops/launchd/com.autoresearch.protocol101.paper-session.plist" "$HOME/Library/LaunchAgents/"
cp "v4/ops/launchd/com.autoresearch.protocol101.daily-monitor.plist" "$HOME/Library/LaunchAgents/"
cp "v4/ops/launchd/com.autoresearch.premiumblend.no-order-surface-check.plist" "$HOME/Library/LaunchAgents/"
cp "v4/ops/launchd/com.autoresearch.ibgateway.paper-shutdown.plist" "$HOME/Library/LaunchAgents/"
launchctl enable "gui/$UID/com.autoresearch.ibgateway.paper"
launchctl enable "gui/$UID/com.autoresearch.protocol101.paper-preflight"
launchctl enable "gui/$UID/com.autoresearch.protocol101.paper-session"
launchctl enable "gui/$UID/com.autoresearch.protocol101.daily-monitor"
launchctl enable "gui/$UID/com.autoresearch.premiumblend.no-order-surface-check"
launchctl enable "gui/$UID/com.autoresearch.ibgateway.paper-shutdown"
launchctl bootstrap "gui/$UID" "$HOME/Library/LaunchAgents/com.autoresearch.ibgateway.paper.plist"
launchctl bootstrap "gui/$UID" "$HOME/Library/LaunchAgents/com.autoresearch.protocol101.paper-preflight.plist"
launchctl bootstrap "gui/$UID" "$HOME/Library/LaunchAgents/com.autoresearch.protocol101.paper-session.plist"
launchctl bootstrap "gui/$UID" "$HOME/Library/LaunchAgents/com.autoresearch.protocol101.daily-monitor.plist"
launchctl bootstrap "gui/$UID" "$HOME/Library/LaunchAgents/com.autoresearch.premiumblend.no-order-surface-check.plist"
launchctl bootstrap "gui/$UID" "$HOME/Library/LaunchAgents/com.autoresearch.ibgateway.paper-shutdown.plist"
echo "Installed daily paper autopilot: IB Gateway paper autostart, preflight, paper session, daily monitor, premium-blend no-order surface autotest, and post-close shutdown LaunchAgents."
