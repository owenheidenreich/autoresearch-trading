#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STATE_FILE="$PROJECT_ROOT/.deploy-state"
DATA_PT="${V3_DATA_PATH:-$PROJECT_ROOT/v2/data.pt}"
SSH_PASS="${DEPLOY_SSH_PASS:-autoresearch2026}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "ERROR: $*"; exit 1; }

load_state() {
    [[ -f "$STATE_FILE" ]] || die "No .deploy-state found. Run './v3/ops/deploy.sh boot' first."
    # shellcheck disable=SC1090
    source "$STATE_FILE"
}

ssh_cmd() {
    SSHPASS="$SSH_PASS" sshpass -e ssh -T \
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=15 -o LogLevel=ERROR \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -o PubkeyAuthentication=no \
        -p "$SSH_PORT" "root@$SSH_HOST" "$@"
}

scp_cmd() {
    SSHPASS="$SSH_PASS" sshpass -e scp \
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=15 -o LogLevel=ERROR \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -o PubkeyAuthentication=no \
        -P "$SSH_PORT" "$@"
}

cmd_passthrough() {
    exec "$PROJECT_ROOT/v2/ops/deploy.sh" "$1"
}

cmd_start() {
    load_state
    log "Uploading v3 workspace snapshot..."
    local bundle
    bundle="/tmp/autoresearch-v3-workspace-$$.tgz"
    tar -czf "$bundle" -C "$PROJECT_ROOT" \
        --no-mac-metadata --no-xattrs \
        --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
        --exclude='archive' --exclude='results' \
        --exclude='v2/artifacts' --exclude='v3/artifacts' --exclude='v3/models' \
        v2 v3 tests pyproject.toml README.md CLAUDE.md
    scp_cmd "$bundle" "root@$SSH_HOST:/root/v3-workspace.tgz"
    rm -f "$bundle"
    ssh_cmd "rm -rf /root/v3 /root/tests && mkdir -p /root && tar -xzf /root/v3-workspace.tgz -C /root && mkdir -p /root/v3/models /root/v3/artifacts && rm -f /root/v3-workspace.tgz"
    [[ -f "$DATA_PT" ]] || die "data.pt not found: $DATA_PT"
    scp_cmd "$DATA_PT" "root@$SSH_HOST:/root/v2/data.pt"
    ssh_cmd "command -v pip3 >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq python3-pip > /dev/null 2>&1)"
    ssh_cmd "pip3 install -q -r /root/v3/ops/requirements-gpu.txt || python3 -m pip install -q -r /root/v3/ops/requirements-gpu.txt"
    ssh_cmd "python3 -m v3.ops.pre_run_gate --data /root/v2/data.pt --skip-train-smoke"
    log "v3 workspace ready on GPU node"
}

cmd_run_one() {
    local exp_id="${1:-}"
    [[ -n "$exp_id" ]] || die "Usage: ./v3/ops/deploy.sh run_one <exp_id> [run_experiment args]"
    shift || true
    load_state
    "$SCRIPT_DIR/deploy.sh" start
    log "Running v3 experiment $exp_id on remote GPU..."
    local extra_args=("$@")
    local remote_cmd="cd /root && python3 -u -m v3.ops.run_experiment --id '$exp_id' --data /root/v2/data.pt --device cuda"
    if [[ ${#extra_args[@]} -gt 0 ]]; then
        remote_cmd+=" $(printf "%q " "${extra_args[@]}")"
    fi
    local remote_log="/root/${exp_id}.log"
    # Launch with nohup so training survives SSH drops; tail the log for live output
    ssh_cmd "nohup bash -c '$remote_cmd' > $remote_log 2>&1 & echo PID=\$!"
    log "Training launched in background on GPU node (log: $remote_log)"
    log "Tailing remote log — Ctrl-C safe, training continues on GPU..."
    # Tail until the process exits; poll every 10s to detect completion
    while true; do
        ssh_cmd "cat $remote_log 2>/dev/null" > /tmp/"$exp_id".v3.log 2>/dev/null || true
        # Check if process is still running
        local running
        running=$(ssh_cmd "ps aux | grep 'run_experiment.*$exp_id' | grep -v grep | wc -l" 2>/dev/null || echo "0")
        # Show new lines
        tail -1 /tmp/"$exp_id".v3.log 2>/dev/null
        if [[ "$running" == "0" ]]; then
            log "Training process finished"
            # Final log pull
            ssh_cmd "cat $remote_log 2>/dev/null" > /tmp/"$exp_id".v3.log 2>/dev/null || true
            cat /tmp/"$exp_id".v3.log
            break
        fi
        sleep 30
    done
    mkdir -p "$PROJECT_ROOT/v3/artifacts"
    scp_cmd -r "root@$SSH_HOST:/root/v3/artifacts/$exp_id" "$PROJECT_ROOT/v3/artifacts/" 2>/dev/null || true
    scp_cmd "root@$SSH_HOST:/root/v3/results.tsv" "$PROJECT_ROOT/v3/results.tsv" 2>/dev/null || true
    log "Downloaded artifact and updated results.tsv"
}

cmd_status() {
    load_state
    echo "Akash SSH: $SSH_HOST:$SSH_PORT"
    ssh_cmd "cd /root && python3 -m v3.ops.status_report --data /root/v2/data.pt" || true
}

case "${1:-}" in
    boot|ssh|stop)
        cmd_passthrough "$1"
        ;;
    start)
        cmd_start
        ;;
    run_one)
        shift
        cmd_run_one "$@"
        ;;
    status)
        cmd_status
        ;;
    *)
        cat <<'EOF'
Usage:
  ./v3/ops/deploy.sh boot
  ./v3/ops/deploy.sh start
  ./v3/ops/deploy.sh run_one <exp_id> [run_experiment args]
  ./v3/ops/deploy.sh status
  ./v3/ops/deploy.sh ssh
  ./v3/ops/deploy.sh stop
EOF
        exit 1
        ;;
esac
