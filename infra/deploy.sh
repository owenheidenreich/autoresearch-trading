#!/usr/bin/env bash
# ===========================================================================
# Akash GPU — Two Commands
# ===========================================================================
#   ./deploy.sh boot    → Deploy GPU container on Akash, wait for SSH
#   ./deploy.sh start   → Upload code + data, start autoresearch loop
#
# Optional:
#   ./deploy.sh sync    → Foreground sync (auto-sync runs with start)
#   ./deploy.sh ssh     → Drop into SSH shell on the H100
#   ./deploy.sh logs    → Tail the autoresearch loop log
#   ./deploy.sh status  → Show GPU, loop PID, active run status (from current_run.txt)
#   ./deploy.sh stop    → Kill loop + close Akash deployment
# ===========================================================================
set -euo pipefail

# --- Config ---
AKASH_NODE="https://akash-rpc.polkachu.com:443"
AKASH_CHAIN_ID="akashnet-2"
AKASH_KEYRING_BACKEND="os"
AKASH_FROM="trinity-wallet"
AKASH_OWNER="akash155hphg6qyy3vtr584p38wlngtqxzdr0l6jutmp"
AKASH_GAS="auto"
AKASH_GAS_ADJUSTMENT="1.5"
AKASH_GAS_PRICES="0.025uakt"
AKASH_SIGN_MODE="amino-json"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SDL_FILE="$SCRIPT_DIR/deploy-autoresearch.yaml"
DATA_PT="$HOME/.cache/autoresearch-trading/features/data.pt"
STATE_FILE="$PROJECT_ROOT/.deploy-state"

SSH_PASS="autoresearch2026"
# Auto-source .env if ANTHROPIC_API_KEY is not already set
if [[ -z "${ANTHROPIC_API_KEY:-}" && -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi
ANTHROPIC_KEY="${ANTHROPIC_API_KEY:-}"
# Optional override to force a specific provider for bidding.
AKASH_PROVIDER_OVERRIDE="${AKASH_PROVIDER_OVERRIDE:-}"
# Preferred GPU model order (comma separated) for provider selection.
# Default policy: H100 first, A100 fallback.
AKASH_GPU_PRIORITY="${AKASH_GPU_PRIORITY:-h100,a100}"

export AKASH_NODE AKASH_CHAIN_ID AKASH_KEYRING_BACKEND AKASH_FROM
export AKASH_GAS AKASH_GAS_ADJUSTMENT AKASH_GAS_PRICES AKASH_SIGN_MODE

# --- Helpers ---
log()  { echo "[$(date +%H:%M:%S)] $*"; }
die()  { log "ERROR: $*"; exit 1; }

save_state() {
    printf 'DSEQ=%s\nGSEQ=%s\nASEQ=%s\nPROVIDER=%s\nSSH_HOST=%s\nSSH_PORT=%s\n' \
        "$DSEQ" "$GSEQ" "$ASEQ" "$PROVIDER" "$SSH_HOST" "$SSH_PORT" \
        > "$STATE_FILE"
    log "State saved → $STATE_FILE"
}

load_state() {
    [[ -f "$STATE_FILE" ]] || die "No .deploy-state found. Run './deploy.sh boot' first."
    source "$STATE_FILE"
    log "Loaded: DSEQ=$DSEQ  SSH=$SSH_HOST:$SSH_PORT"
}

ssh_cmd() {
    SSHPASS="$SSH_PASS" sshpass -e ssh \
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=15 -o LogLevel=ERROR \
        -p "$SSH_PORT" "root@$SSH_HOST" "$@"
}

scp_cmd() {
    SSHPASS="$SSH_PASS" sshpass -e scp \
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=15 -o LogLevel=ERROR \
        -P "$SSH_PORT" "$@"
}

get_remote_run_name() {
    local run_name
    run_name=$(ssh_cmd "test -f /root/results/current_run.txt && tr -d '\\r\\n' < /root/results/current_run.txt" 2>/dev/null || true)
    [[ -n "$run_name" ]] || die "Active run pointer missing: /root/results/current_run.txt"
    ssh_cmd "test -d /root/results/$run_name" &>/dev/null || \
        die "Active run pointer invalid: /root/results/$run_name not found"
    echo "$run_name"
}

write_local_current_run_pointer() {
    local run_name="$1"
    local dest_root="${2:-$PROJECT_ROOT/results}"
    mkdir -p "$dest_root"
    printf '%s\n' "$run_name" > "$dest_root/current_run.txt"
    # Keep canonical project pointer mirrored even when sync/download uses custom destination.
    if [[ "$dest_root" != "$PROJECT_ROOT/results" ]]; then
        mkdir -p "$PROJECT_ROOT/results"
        printf '%s\n' "$run_name" > "$PROJECT_ROOT/results/current_run.txt"
    fi
}

sync_promoted_ledger_local() {
    local dest_root="${1:-$PROJECT_ROOT/results}"
    local dest_promoted="$dest_root/promoted"
    mkdir -p "$dest_promoted"
    scp_cmd "root@$SSH_HOST:/root/results/promoted/history.jsonl" "$dest_promoted/history.jsonl" 2>/dev/null || true
    scp_cmd "root@$SSH_HOST:/root/results/promoted/current.txt" "$dest_promoted/current.txt" 2>/dev/null || true
    if [[ "$dest_root" != "$PROJECT_ROOT/results" ]]; then
        mkdir -p "$PROJECT_ROOT/results/promoted"
        [[ -f "$dest_promoted/history.jsonl" ]] && cp "$dest_promoted/history.jsonl" "$PROJECT_ROOT/results/promoted/history.jsonl"
        [[ -f "$dest_promoted/current.txt" ]] && cp "$dest_promoted/current.txt" "$PROJECT_ROOT/results/promoted/current.txt"
    fi
}

wait_for_ssh() {
    log "Waiting for SSH..."
    for _ in $(seq 1 60); do
        if ssh_cmd "echo OK" &>/dev/null; then
            log "SSH ready!"
            return 0
        fi
        sleep 5
        printf "."
    done
    die "SSH not available after 5 minutes"
}

# ===================================================================
# BOOT — deploy H100 on Akash, wait until SSH is accessible
# ===================================================================
cmd_boot() {
    [[ -n "$ANTHROPIC_KEY" ]] || die "Set ANTHROPIC_API_KEY environment variable before deploying"
    log "=== BOOT: Creating Akash GPU Deployment ==="
    [[ -f "$SDL_FILE" ]] || die "SDL not found: $SDL_FILE"

    # 1. Submit deployment TX
    # NOTE: Default escrow deposit is 0.5 AKT which drains in ~2 minutes
    # at H100 pricing.  Deposit enough for the planned run duration.
    # For short tests: 5 AKT (~1 hour).  For full runs: 50 AKT (~8 hours).
    DEPOSIT_AKT="${DEPOSIT_AKT:-15}"
    DEPOSIT_UAKT=$((DEPOSIT_AKT * 1000000))
    log "Submitting deployment TX (deposit=${DEPOSIT_AKT} AKT)..."
    TX_OUTPUT=$(provider-services tx deployment create "$SDL_FILE" \
        --deposit "${DEPOSIT_UAKT}uakt" \
        --from "$AKASH_FROM" --yes --output json 2>&1)

    TXHASH=$(echo "$TX_OUTPUT" | grep -o '"txhash":"[^"]*"' | head -1 | cut -d'"' -f4)
    [[ -n "$TXHASH" ]] || { echo "$TX_OUTPUT"; die "No txhash returned"; }
    log "TX: $TXHASH"

    # 2. Confirm TX
    sleep 8
    TX_RESULT=$(provider-services query tx "$TXHASH" --node "$AKASH_NODE" --output json 2>&1)
    TX_CODE=$(echo "$TX_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('code',1))" 2>/dev/null || echo "1")
    [[ "$TX_CODE" == "0" ]] || { echo "$TX_RESULT" | head -30; die "TX failed (code=$TX_CODE)"; }
    log "TX confirmed ✓"

    # 3. Get DSEQ
    sleep 3
    DSEQ=$(provider-services query deployment list \
        --owner "$AKASH_OWNER" --state active \
        --node "$AKASH_NODE" --output json 2>/dev/null | \
        python3 -c "import sys,json; deps=json.load(sys.stdin).get('deployments',[]); print(deps[-1]['deployment']['id']['dseq'] if deps else '')" 2>/dev/null)
    [[ -n "$DSEQ" ]] || die "Could not find DSEQ"
    GSEQ=1; ASEQ=1
    log "DSEQ: $DSEQ"

    # 4. Wait for bids
    log "Waiting 45s for provider bids (GPU priority: $AKASH_GPU_PRIORITY)..."
    sleep 45

    BIDS_JSON=$(provider-services query market bid list \
        --owner "$AKASH_OWNER" --dseq "$DSEQ" \
        --node "$AKASH_NODE" --output json 2>&1)

    PROVIDER=$(BIDS_JSON="$BIDS_JSON" python3 - "$AKASH_PROVIDER_OVERRIDE" "$AKASH_GPU_PRIORITY" "$AKASH_NODE" <<'PY'
import json
import os
import subprocess
import sys

override = (sys.argv[1] or "").strip()
priority = [p.strip().lower() for p in (sys.argv[2] or "").split(",") if p.strip()]
node = sys.argv[3]

blocked = {
    # Known unreliable for this workflow (stuck startup / no provisioning).
    "akash1kqzpqqhm39umt06wu8m4hx63v5hefhrfmjf9dj",
    "akash1ggfvyhr9sar4uxjs4hth3p4kzrwk7lysnenj3g",
    "akash1sevd2ymtty3dpq9ycxgkhuzzk4fe6mchqdwd4e",
}

raw = os.environ.get("BIDS_JSON", "")
try:
    data = json.loads(raw)
except Exception:
    print("")
    sys.exit(0)

bids = [b for b in data.get("bids", []) if b.get("bid", {}).get("state") == "open"]
if not bids:
    bids = data.get("bids", [])
if not bids:
    print("")
    sys.exit(0)

bids.sort(key=lambda b: float(b.get("bid", {}).get("price", {}).get("amount", "9e18")))

if override:
    for b in bids:
        if b.get("bid", {}).get("id", {}).get("provider") == override:
            print(override)
            sys.exit(0)


def get_provider_models(provider: str) -> set[str]:
    try:
        out = subprocess.check_output(
            [
                "provider-services",
                "query",
                "provider",
                "get",
                provider,
                "--node",
                node,
                "-o",
                "json",
            ],
            stderr=subprocess.DEVNULL,
            timeout=12,
            text=True,
        )
        payload = json.loads(out)
        attrs = payload.get("attributes", []) or payload.get("provider", {}).get("attributes", [])
        models = set()
        for attr in attrs:
            key = str(attr.get("key", "")).lower()
            if "/model/" in key:
                model = key.split("/model/", 1)[1].split("/")[0]
                if model:
                    models.add(model)
        return models
    except Exception:
        return set()


ranked = []
for b in bids:
    provider = b.get("bid", {}).get("id", {}).get("provider", "")
    if not provider:
        continue
    price = float(b.get("bid", {}).get("price", {}).get("amount", "9e18"))
    models = get_provider_models(provider)

    # Lower is better.
    blocked_rank = 1 if provider in blocked else 0
    gpu_rank = len(priority) + 1
    for i, wanted in enumerate(priority):
        if wanted in models:
            gpu_rank = i
            break
    if models and gpu_rank == len(priority) + 1:
        gpu_rank = len(priority)
    ranked.append((blocked_rank, gpu_rank, price, provider))

if not ranked:
    print("")
    sys.exit(0)

ranked.sort()
for blocked_rank, _gpu_rank, _price, provider in ranked:
    if blocked_rank == 0:
        print(provider)
        sys.exit(0)

# Last resort: all candidates were blocked.
print(ranked[0][3])
PY
)
    [[ -n "$PROVIDER" ]] || { echo "$BIDS_JSON" | head -30; die "No bids received"; }
    log "Provider: $PROVIDER"

    # 5. Create lease
    log "Creating lease..."
    LEASE_TX=$(provider-services tx market lease create \
        --dseq "$DSEQ" --gseq "$GSEQ" --oseq "$ASEQ" \
        --provider "$PROVIDER" --from "$AKASH_FROM" \
        --yes --output json 2>&1)
    LEASE_HASH=$(echo "$LEASE_TX" | grep -o '"txhash":"[^"]*"' | head -1 | cut -d'"' -f4)
    log "Lease TX: $LEASE_HASH"
    sleep 8

    # 6. Send manifest (retry 3x)
    log "Sending manifest..."
    for attempt in 1 2 3; do
        provider-services send-manifest "$SDL_FILE" \
            --dseq "$DSEQ" --provider "$PROVIDER" \
            --from "$AKASH_FROM" 2>&1 && break
        log "  Attempt $attempt failed, retrying in 10s..."
        sleep 10
    done

    # 7. Wait for container, get SSH port
    log "Waiting 30s for container startup..."
    sleep 30

    LEASE_STATUS=""
    for attempt in $(seq 1 6); do
        LEASE_STATUS=$(provider-services lease-status \
            --dseq "$DSEQ" --gseq "$GSEQ" --oseq "$ASEQ" \
            --provider "$PROVIDER" --from "$AKASH_FROM" 2>&1) && break
        log "  lease-status attempt $attempt — retrying in 15s..."
        sleep 15
    done

    SSH_INFO=$(echo "$LEASE_STATUS" | python3 -c "
import sys, json
data = json.load(sys.stdin)
fwd = data.get('forwarded_ports', {})
for ports in fwd.values():
    for p in ports:
        if p.get('port') == 22:
            print(f\"{p['host']}:{p['externalPort']}\")
            sys.exit(0)
for svc in data.get('services', {}).values():
    for uri in svc.get('uris', []):
        print(uri); sys.exit(0)
print('')
" 2>/dev/null)

    [[ -n "$SSH_INFO" ]] || { echo "$LEASE_STATUS"; die "No SSH endpoint found"; }

    if [[ "$SSH_INFO" == *":"* ]]; then
        SSH_HOST="${SSH_INFO%%:*}"; SSH_PORT="${SSH_INFO##*:}"
    else
        SSH_HOST="$SSH_INFO"; SSH_PORT=22
    fi

    save_state

    # 8. Wait for SSH + verify GPU
    wait_for_ssh
    GPU=$(ssh_cmd "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>/dev/null || echo "unknown")
    log ""
    log "=== GPU READY ==="
    log "GPU:  $GPU"
    log "SSH:  ssh -p $SSH_PORT root@$SSH_HOST  (pass: $SSH_PASS)"
    log ""
    log "Next: ./deploy.sh start"
}

# ===================================================================
# START — upload files + launch autoresearch loop
# Usage: ./deploy.sh start [--hours H] [--max-experiments N]
# ===================================================================
cmd_start() {
    [[ -n "$ANTHROPIC_KEY" ]] || die "Set ANTHROPIC_API_KEY environment variable before deploying"
    load_state
    log "=== START: Uploading files + launching loop ==="

    # Collect extra args to pass through to start_loop.sh
    LOOP_ARGS="${EXTRA_ARGS:-}"

    wait_for_ssh

    # Upload exact local workspace snapshot (no git/network pulls on remote).
    local bundle source_git_sha source_dirty_count bundle_sha
    bundle="$(mktemp /tmp/autoresearch-workspace.XXXXXX.tgz)"
    source_git_sha=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo "nogit")
    source_dirty_count=$(git -C "$PROJECT_ROOT" status --porcelain 2>/dev/null | wc -l | tr -d ' ' || echo "0")
    log "Packaging local workspace snapshot..."
    tar -czf "$bundle" -C "$PROJECT_ROOT" \
        --no-mac-metadata --no-xattrs \
        --exclude='.git' \
        --exclude='.venv' \
        --exclude='.pytest_cache' \
        --exclude='results' \
        --exclude='archive' \
        --exclude='training/best_model.pt' \
        --exclude='training/trade_log.csv' \
        training infra pyproject.toml README.md
    bundle_sha=$(shasum -a 256 "$bundle" | awk '{print $1}')
    log "Uploading workspace snapshot ($(du -h "$bundle" | cut -f1), sha256=$bundle_sha)..."
    ssh_cmd "rm -rf /root/autoresearch-trading && mkdir -p /root/autoresearch-trading"
    scp_cmd "$bundle" "root@$SSH_HOST:/root/autoresearch-trading.tgz"
    rm -f "$bundle"
    ssh_cmd "tar -xzf /root/autoresearch-trading.tgz -C /root/autoresearch-trading && rm -f /root/autoresearch-trading.tgz"

    # Link runtime entrypoints from the uploaded snapshot into /root.
    log "Linking runtime files from snapshot..."
    ssh_cmd "set -e
ln -sfn /root/autoresearch-trading/training/train.py /root/train.py
ln -sfn /root/autoresearch-trading/training/prepare.py /root/prepare.py
ln -sfn /root/autoresearch-trading/training/program.md /root/program.md
ln -sfn /root/autoresearch-trading/training/lab_notebook.md /root/lab_notebook.md
ln -sfn /root/autoresearch-trading/training/run_loop.py /root/run_loop.py
ln -sfn /root/autoresearch-trading/infra/start_loop.sh /root/start_loop.sh
ln -sfn /root/autoresearch-trading/infra/watchdog.sh /root/watchdog.sh
if [ -f /root/autoresearch-trading/training/best_train.py ]; then
  ln -sfn /root/autoresearch-trading/training/best_train.py /root/best_train.py
else
  rm -f /root/best_train.py
fi"

    # Integrity check: confirm remote run_loop.py matches local snapshot.
    local local_runloop_sha remote_runloop_sha
    local_runloop_sha=$(shasum -a 256 "$PROJECT_ROOT/training/run_loop.py" | awk '{print $1}')
    remote_runloop_sha=$(ssh_cmd "sha256sum /root/run_loop.py | awk '{print \$1}'" 2>/dev/null || true)
    [[ "$remote_runloop_sha" == "$local_runloop_sha" ]] || die "Snapshot integrity check failed (run_loop.py hash mismatch)"

    # Record deploy source metadata on remote for auditability.
    local deployed_at
    deployed_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    ssh_cmd "printf '%s\n' '{\"bundle_sha256\":\"$bundle_sha\",\"git_sha\":\"$source_git_sha\",\"dirty_files\":$source_dirty_count,\"deployed_at\":\"$deployed_at\",\"source\":\"local_workspace_snapshot\"}' > /root/deploy_source.json"

    # Upload data.pt to the path prepare.py expects: ~/.cache/autoresearch-trading/features/
    [[ -f "$DATA_PT" ]] || die "data.pt not found: $DATA_PT"
    log "Uploading data.pt ($(du -h "$DATA_PT" | cut -f1))..."
    ssh_cmd "mkdir -p /root/.cache/autoresearch-trading/features"
    scp_cmd "$DATA_PT" "root@$SSH_HOST:/root/.cache/autoresearch-trading/features/data.pt"

    # Upload best_model.pt if it exists — warm-start from previous training run
    if [[ -f "$PROJECT_ROOT/training/best_model.pt" ]]; then
        log "Uploading best_model.pt ($(du -h "$PROJECT_ROOT/training/best_model.pt" | cut -f1)) for warm-start..."
        ssh_cmd "rm -f /root/best_model.pt"
        scp_cmd "$PROJECT_ROOT/training/best_model.pt" "root@$SSH_HOST:/root/best_model.pt"
    else
        log "No best_model.pt found — starting from scratch"
    fi

    # Seed remote promoted-history ledger so prompt context persists across deployments.
    ssh_cmd "mkdir -p /root/results/promoted"
    if [[ -f "$PROJECT_ROOT/results/promoted/history.jsonl" ]]; then
        log "Uploading promoted history ledger..."
        scp_cmd "$PROJECT_ROOT/results/promoted/history.jsonl" "root@$SSH_HOST:/root/results/promoted/history.jsonl"
    fi
    if [[ -f "$PROJECT_ROOT/results/promoted/current.txt" ]]; then
        scp_cmd "$PROJECT_ROOT/results/promoted/current.txt" "root@$SSH_HOST:/root/results/promoted/current.txt"
    fi

    # Verify
    log "Files on remote GPU node:"
    ssh_cmd "ls -lh /root/*.py /root/*.sh /root/deploy_source.json /root/.cache/autoresearch-trading/features/data.pt"

    # Pre-flight dry-run on GPU node — verify data, GPU, API before consuming time
    log "Running pre-flight checks on remote GPU node..."
    if ! ssh_cmd "cd /root && ANTHROPIC_API_KEY='$ANTHROPIC_KEY' /opt/conda/bin/python -u run_loop.py --dry-run"; then
        die "Pre-flight failed on remote GPU node. Fix issues before running."
    fi
    log "Pre-flight passed!"

    # Launch loop via start_loop.sh — pass API key as env var (not sed)
    log "Starting autoresearch loop... $LOOP_ARGS"
    ssh_cmd "chmod +x /root/start_loop.sh && ANTHROPIC_API_KEY='$ANTHROPIC_KEY' /root/start_loop.sh $LOOP_ARGS"

    # Auto-launch sync in background — no more forgetting to run sync in 2nd terminal
    local sync_dest="$PROJECT_ROOT/results"
    mkdir -p "$sync_dest"
    log "Starting auto-sync to $sync_dest ..."
    _run_sync "$sync_dest" > "$sync_dest/sync.log" 2>&1 &
    local sync_pid=$!
    echo "$sync_pid" > "$PROJECT_ROOT/.sync-pid"

    log ""
    log "=== LOOP RUNNING ==="
    log "Auto-sync: PID $sync_pid → $sync_dest"
    log "  Sync log: tail -f $sync_dest/sync.log"
    log "  Manual:   ./deploy.sh sync    ← foreground sync if you prefer"
    log "Logs:   ./deploy.sh logs"
    log "Status: ./deploy.sh status"
    log "SSH:    ./deploy.sh ssh"
}

# ===================================================================
# Convenience commands
# ===================================================================
cmd_ssh() {
    load_state
    log "Connecting to H100..."
    SSHPASS="$SSH_PASS" sshpass -e ssh \
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR -p "$SSH_PORT" "root@$SSH_HOST"
}

cmd_logs() {
    load_state
    log "Tailing loop.log (Ctrl-C to stop)..."
    ssh_cmd "tail -f /root/loop.log"
}

cmd_status() {
    load_state
    local run_name
    run_name=$(get_remote_run_name)

    # Rich dashboard via heredoc Python script (avoids nested quoting)
    ssh_cmd python3 - <<PYEOF
import json, subprocess, os, textwrap, sys

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except:
        return ""

run_name = "${run_name}"
run_dir = f"/root/results/{run_name}"
status_path = os.path.join(run_dir, "status.json")
exp_path = os.path.join(run_dir, "experiments.v2.jsonl")

W = 76  # dashboard width

def bar(label=""):
    if label:
        pad = W - len(label) - 4
        print(f"== {label} " + "=" * pad)
    else:
        print("=" * W)

def wrap(text, indent=4, width=W-4):
    lines = textwrap.wrap(text, width=width)
    return "\n".join(" " * indent + l for l in lines)

bar()

# --- Header: GPU + Loop ---
gpu = run("nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader")
if gpu:
    parts = [p.strip() for p in gpu.split(",")]
    gpu_name = parts[0] if len(parts) > 0 else "?"
    gpu_mem = f"{parts[1]} / {parts[2]}" if len(parts) > 2 else "?"
    gpu_util = parts[3] if len(parts) > 3 else "?"
    print(f"  GPU:  {gpu_name}  |  Mem: {gpu_mem}  |  Util: {gpu_util}")
else:
    print("  GPU:  unavailable")

pid = run("pgrep -f '[r]un_loop.py'")
if pid:
    pid_line = pid.split("\n")[0]
    uptime = run(f"ps -o etime= -p {pid_line}").strip()
    print(f"  Loop: RUNNING (PID {pid_line}, uptime {uptime})")
else:
    print(f"  Loop: NOT RUNNING")

bar("Run")
print(f"  Active: {run_name}")

# --- Progress ---
if not os.path.exists(status_path):
    print(f"ERROR: Missing status file: {status_path}")
    sys.exit(2)

with open(status_path) as f:
    s = json.load(f)
phase = s.get("phase", "?")
best = s.get("best_score", 0)
kept = s.get("kept", 0)
failed = s.get("failed", 0)
total = s.get("total", 0)
remaining = s.get("time_remaining_h", 0)
exp_id = s.get("experiment_id", "?")

bar("Progress")
print(f"  Current:    Experiment #{exp_id}  ({phase})")
if isinstance(best, (int, float)):
    print(f"  Best Score: {best:.4f}")
else:
    print(f"  Best Score: {best}")
print(f"  Results:    {kept} kept / {failed} failed / {total} total")
if isinstance(remaining, (int, float)):
    print(f"  Time Left:  {remaining:.1f}h")
else:
    print(f"  Time Left:  {remaining}")
checksum = s.get("contract_checksum")
if checksum:
    print(f"  Contract:   {checksum}")
last_failure = s.get("last_failure_type")
last_flags = s.get("last_anomaly_flags", [])
if last_failure or last_flags:
    if not isinstance(last_flags, list):
        last_flags = []
    flags_txt = ",".join(last_flags) if last_flags else "none"
    print(f"  Reliability: failure={last_failure or 'none'} anomalies={flags_txt}")

# --- Experiment History ---
bar("Experiment History")
if os.path.exists(exp_path):
    with open(exp_path) as f:
        lines = f.readlines()

    # Table header
    print(f"  {'#':>3}  {'Score':>8}  {'PF':>5}  {'TPD':>5}  {'Sharpe':>7}  {'WR':>5}  {'Kept':>4}  Failure")
    print(f"  {'---':>3}  {'-----':>8}  {'--':>5}  {'---':>5}  {'------':>7}  {'--':>5}  {'----':>4}  -------")

    for line in lines:
        try:
            e = json.loads(line)
            eid = e.get("experiment_id", "?")
            score = e.get("score", -999)
            kept_flag = e.get("kept", False)
            failure = e.get("failure_type", "none")

            if score == -999:
                print(f"  {eid:>3}  {'FAIL':>8}  {'':>5}  {'':>5}  {'':>7}  {'':>5}  {'':>4}  {str(failure)[:25]}")
            else:
                pf = e.get("profit_factor", 0)
                tpd = e.get("trades_per_day", 0)
                sharpe = e.get("trade_sharpe", 0)
                wr = e.get("win_rate", 0)
                mark = " <--" if kept_flag else ""
                print(f"  {eid:>3}  {score:>8.3f}  {pf:>5.2f}  {tpd:>5.1f}  {sharpe:>7.2f}  {wr:>4.0%}  {'YES' if kept_flag else '':>4}{mark}  {str(failure)[:25]}")
        except:
            pass
else:
    print("  (no experiments yet)")

# --- Latest Experiment Detail ---
if os.path.exists(exp_path):
    with open(exp_path) as f:
        lines = f.readlines()
    if lines:
        last = json.loads(lines[-1])
        bar(f"Latest: Experiment #{last.get('experiment_id', '?')}")

        reasoning = last.get("reasoning", "")
        change = last.get("change_summary", "")
        err = last.get("error", "")

        if change:
            print(f"  Changes:")
            for part in change.split(";"):
                part = part.strip()
                if part:
                    print(f"    {part}")

        if reasoning:
            print(f"\n  Reasoning:")
            print(wrap(reasoning, indent=4, width=W-6))

        if err and last.get("score", 0) == -999:
            print(f"\n  Error:")
            print(wrap(err[:200], indent=4, width=W-6))
        failure_type = last.get("failure_type")
        anomaly_flags = last.get("anomaly_flags", [])
        if failure_type or anomaly_flags:
            if not isinstance(anomaly_flags, list):
                anomaly_flags = []
            print(f"\n  Reliability:")
            print(f"    failure_type={failure_type or 'none'}")
            print(f"    anomaly_flags={','.join(anomaly_flags) if anomaly_flags else 'none'}")

        score = last.get("score", -999)
        if score != -999:
            print(f"\n  Result: score={score:.4f}  pf={last.get('profit_factor',0):.2f}"
                  f"  tpd={last.get('trades_per_day',0):.1f}"
                  f"  sharpe={last.get('trade_sharpe',0):.2f}"
                  f"  wr={last.get('win_rate',0):.0%}"
                  f"  {'KEPT' if last.get('kept') else 'reverted'}")

bar()
PYEOF
}

cmd_download() {
    load_state
    local dest="${EXTRA_ARGS:-$PROJECT_ROOT/results}"
    mkdir -p "$dest"
    local run_name
    run_name=$(get_remote_run_name)
    local remote_run_dir="/root/results/$run_name"
    local local_run_dir="$dest/$run_name"

    log "Downloading run $run_name to $dest ..."
    write_local_current_run_pointer "$run_name" "$dest"
    scp_cmd -r "root@$SSH_HOST:$remote_run_dir" "$dest/" 2>/dev/null || \
        die "Failed to download canonical run folder: $remote_run_dir"
    log "  ↓ $run_name ($(du -h "$local_run_dir" | cut -f1))"
    log "  ↳ Updated current_run.txt -> $run_name"
    sync_promoted_ledger_local "$dest"

    if [[ -f "$PROJECT_ROOT/tools/ingest_evidence.py" ]]; then
        python3 "$PROJECT_ROOT/tools/ingest_evidence.py" \
            --results-root "$PROJECT_ROOT/results" \
            --output-root "$PROJECT_ROOT/results/analysis" >/dev/null 2>&1 || \
            log "  ⚠ evidence ingest failed (non-blocking)"
    fi

    # Preserve artifacts in training/ for next deployment
    if [[ -f "$local_run_dir/best_model.pt" ]]; then
        cp "$local_run_dir/best_model.pt" "$PROJECT_ROOT/training/best_model.pt"
        log "  ↳ Copied best_model.pt → training/ (warm-start for next run)"
    fi
    if [[ -f "$local_run_dir/best_train.py" ]]; then
        cp "$local_run_dir/best_train.py" "$PROJECT_ROOT/training/best_train.py"
        # best_train.py IS the best code — use it as train.py for next run
        cp "$local_run_dir/best_train.py" "$PROJECT_ROOT/training/train.py"
        log "  ↳ Copied best_train.py → training/train.py + best_train.py"
    fi

    log "Done! Results in: $local_run_dir"
}

# ===================================================================
# _run_sync — core sync loop (used by cmd_start background + cmd_sync foreground)
# Usage: _run_sync <dest_dir>
# ===================================================================
_run_sync() {
    local dest_root="$1"
    mkdir -p "$dest_root"

    local last_kept=-1
    local last_total=-1
    local last_run=""
    local poll_interval=30

    log "=== AUTO-SYNC ==="
    log "  Remote:   $SSH_HOST:$SSH_PORT"
    log "  Save to:  $dest_root"
    log "  Polling every ${poll_interval}s"
    log ""

    while true; do
        local run_name
        run_name=$(ssh_cmd "test -f /root/results/current_run.txt && tr -d '\\r\\n' < /root/results/current_run.txt" 2>/dev/null || true)
        if [[ -z "$run_name" ]]; then
            log "WARN: Active run pointer missing — retrying in ${poll_interval}s..."
            sleep "$poll_interval"
            continue
        fi
        local remote_run_dir="/root/results/$run_name"
        if ! ssh_cmd "test -d $remote_run_dir" &>/dev/null; then
            log "WARN: Run dir $remote_run_dir not found — retrying in ${poll_interval}s..."
            sleep "$poll_interval"
            continue
        fi
        local local_run_dir="$dest_root/$run_name"
        mkdir -p "$local_run_dir"
        write_local_current_run_pointer "$run_name" "$dest_root"
        sync_promoted_ledger_local "$dest_root"
        if [[ "$last_run" != "$run_name" ]]; then
            log "  Active run: $run_name"
            last_run="$run_name"
            last_kept=-1
            last_total=-1
        fi

        # Fetch remote run-folder status.json
        local raw
        raw=$(ssh_cmd "cat $remote_run_dir/status.json 2>/dev/null") 2>/dev/null || {
            log "  ⚠ Status not ready at $remote_run_dir/status.json — retrying in ${poll_interval}s"
            sleep "$poll_interval"
            continue
        }

        local kept total phase best_score
        eval "$(echo "$raw" | python3 -c "
import sys, json
s = json.load(sys.stdin)
print(f'kept={s.get(\"kept\",0)}')
print(f'total={s.get(\"total\",0)}')
print(f'phase={s.get(\"phase\",\"unknown\")}')
print(f'best_score={s.get(\"best_score\",0)}')
" 2>/dev/null)" || { sleep "$poll_interval"; continue; }

        # Defaults — prevent set -e crash if python3 returned partial output
        kept=${kept:-0}
        total=${total:-0}
        phase=${phase:-unknown}
        best_score=${best_score:-0}

        # First poll — seed counters without downloading
        if [[ "$last_kept" -eq -1 ]]; then
            last_kept=$kept
            last_total=$total
            log "  Baseline: run=$run_name kept=$kept total=$total best=$best_score phase=$phase"
            sleep "$poll_interval"
            continue
        fi

        # --- New improvement: full canonical run-folder sync ---
        if [[ "$kept" -gt "$last_kept" ]]; then
            log "★ IMPROVEMENT #$kept (score=$best_score) — syncing run folder..."
            scp_cmd -r "root@$SSH_HOST:$remote_run_dir" "$dest_root/" 2>/dev/null || true
            if [[ -f "$PROJECT_ROOT/tools/ingest_evidence.py" ]]; then
                python3 "$PROJECT_ROOT/tools/ingest_evidence.py" \
                    --results-root "$PROJECT_ROOT/results" \
                    --output-root "$PROJECT_ROOT/results/analysis" >/dev/null 2>&1 || true
            fi
            # Copy best model + code to training/ for warm-start continuity
            if [[ -f "$local_run_dir/best_model.pt" ]]; then
                cp "$local_run_dir/best_model.pt" "$PROJECT_ROOT/training/best_model.pt"
                log "  ↳ Updated training/best_model.pt"
            fi
            if [[ -f "$local_run_dir/best_train.py" ]]; then
                cp "$local_run_dir/best_train.py" "$PROJECT_ROOT/training/best_train.py"
                cp "$local_run_dir/best_train.py" "$PROJECT_ROOT/training/train.py"
                log "  ↳ Updated training/train.py + best_train.py"
            fi
            last_kept=$kept
            last_total=$total
            log "  Synced. Best score: $best_score"

        # --- Experiment finished but no improvement: sync metadata + artifacts ---
        elif [[ "$total" -gt "$last_total" ]]; then
            log "  Exp #$total done (not kept). Syncing metadata + artifacts..."
            for f in experiments.v2.jsonl status.json run_metadata.json data_quality_report.json; do
                scp_cmd "root@$SSH_HOST:$remote_run_dir/$f" "$local_run_dir/$f" 2>/dev/null || true
            done
            # Sync full artifacts for new experiments (reasoning, prompts, candidate code)
            mkdir -p "$local_run_dir/artifacts"
            for exp_num in $(seq $((last_total + 1)) $total); do
                scp_cmd -r "root@$SSH_HOST:$remote_run_dir/artifacts/exp-$exp_num" \
                    "$local_run_dir/artifacts/" 2>/dev/null || true
            done
            if [[ -f "$PROJECT_ROOT/tools/ingest_evidence.py" ]]; then
                python3 "$PROJECT_ROOT/tools/ingest_evidence.py" \
                    --results-root "$PROJECT_ROOT/results" \
                    --output-root "$PROJECT_ROOT/results/analysis" >/dev/null 2>&1 || true
            fi
            last_total=$total

        # --- Heartbeat: show sync is alive even when nothing changed ---
        else
            local remaining
            remaining=$(echo "$raw" | python3 -c "import sys,json; print(f\"{json.load(sys.stdin).get('time_remaining_h','?'):.1f}h\")" 2>/dev/null || echo "?")
            log "  ♻ Polling... run=$run_name exp=$total kept=$kept best=$best_score remaining=$remaining"
        fi

        # --- Loop completed: final full canonical run-folder sync + exit ---
        if [[ "$phase" == "completed" ]]; then
            log ""
            log "=== LOOP COMPLETED (run=$run_name score=$best_score, kept=$kept) ==="
            log "Final sync..."
            scp_cmd -r "root@$SSH_HOST:$remote_run_dir" "$dest_root/" 2>/dev/null || true
            if [[ -f "$PROJECT_ROOT/tools/ingest_evidence.py" ]]; then
                python3 "$PROJECT_ROOT/tools/ingest_evidence.py" \
                    --results-root "$PROJECT_ROOT/results" \
                    --output-root "$PROJECT_ROOT/results/analysis" >/dev/null 2>&1 || true
            fi
            # Final warm-start copy
            if [[ -f "$local_run_dir/best_model.pt" ]]; then
                cp "$local_run_dir/best_model.pt" "$PROJECT_ROOT/training/best_model.pt"
                log "  ↳ Updated training/best_model.pt"
            fi
            if [[ -f "$local_run_dir/best_train.py" ]]; then
                cp "$local_run_dir/best_train.py" "$PROJECT_ROOT/training/best_train.py"
                cp "$local_run_dir/best_train.py" "$PROJECT_ROOT/training/train.py"
                log "  ↳ Updated training/train.py + best_train.py"
            fi
            log "All results saved to: $local_run_dir"
            # Clean up sync PID file if we were the background sync
            rm -f "$PROJECT_ROOT/.sync-pid"
            break
        fi

        sleep "$poll_interval"
    done
}

cmd_sync() {
    load_state

    # Guard against duplicate sync — check if background sync is already running
    if [[ -f "$PROJECT_ROOT/.sync-pid" ]]; then
        local spid
        spid=$(cat "$PROJECT_ROOT/.sync-pid")
        if kill -0 "$spid" 2>/dev/null; then
            log "Auto-sync already running (PID $spid)."
            log "Kill it first with: kill $spid && rm '$PROJECT_ROOT/.sync-pid'"
            exit 0
        else
            rm -f "$PROJECT_ROOT/.sync-pid"
        fi
    fi

    local dest="${EXTRA_ARGS:-$PROJECT_ROOT/results}"
    log "Running foreground sync (Ctrl-C to stop)..."
    _run_sync "$dest"
}

cmd_stop() {
    load_state
    echo ""
    echo "This will: kill loop → download results → close deployment"

    # Support -y flag for non-interactive usage
    if [[ "${EXTRA_ARGS:-}" == *"-y"* ]]; then
        log "Non-interactive mode (-y)"
    else
        read -p "Continue? [y/N] " confirm
        [[ "$confirm" =~ ^[Yy]$ ]] || { log "Aborted."; exit 0; }
    fi

    # Kill background sync if running
    if [[ -f "$PROJECT_ROOT/.sync-pid" ]]; then
        local spid
        spid=$(cat "$PROJECT_ROOT/.sync-pid")
        kill "$spid" 2>/dev/null || true
        rm -f "$PROJECT_ROOT/.sync-pid"
        log "Stopped background sync (PID $spid)"
    fi

    log "Killing loop..."
    ssh_cmd "pkill -f run_loop.py 2>/dev/null || true" || true

    # Wait for loop to actually die (up to 30s) — prevents partial file downloads
    for _ in $(seq 1 30); do
        if ! ssh_cmd "pgrep -f run_loop.py" &>/dev/null; then
            break
        fi
        sleep 1
    done
    # Force kill if still alive
    ssh_cmd "pkill -9 -f run_loop.py 2>/dev/null || true" 2>/dev/null || true
    sleep 3  # let filesystem flush before downloading

    log "Downloading results before closing..."
    EXTRA_ARGS="" cmd_download || log "WARNING: Download failed (container may be dead). Proceeding to close deployment."

    log "Closing Akash deployment DSEQ=$DSEQ..."
    provider-services tx deployment close \
        --dseq "$DSEQ" --from "$AKASH_FROM" --yes 2>&1 \
        || log "WARNING: Close TX failed (deployment may already be closed)"
    rm -f "$STATE_FILE"
    log "Deployment closed. Results saved."
}

# ===================================================================
# Dispatch
# ===================================================================
CMD="${1:-help}"
shift 2>/dev/null || true   # remove command name, rest are extra args
EXTRA_ARGS="$*"
export EXTRA_ARGS

case "$CMD" in
    boot)     cmd_boot     ;;
    start)    cmd_start    ;;
    ssh)      cmd_ssh      ;;
    logs)     cmd_logs     ;;
    status)   cmd_status   ;;
    download) cmd_download ;;
    sync)     cmd_sync     ;;
    stop)     cmd_stop     ;;
    *)
        echo "Usage: ./deploy.sh <command> [options]"
        echo ""
        echo "  boot      Deploy H100 container on Akash (~2 min)"
        echo "  start     Upload code + data, start training loop"
        echo "            Options: --hours H  --max-experiments N"
        echo "  ssh       SSH into the H100"
        echo "  logs      Tail the autoresearch loop log"
        echo "  status    GPU, loop PID, active run dashboard (fails fast if pointer missing)"
        echo "  download  Download active canonical run folder once"
        echo "            Optional: ./deploy.sh download /path/to/save"
        echo "  sync      Foreground sync (auto-sync runs automatically with start)"
        echo "            Optional: ./deploy.sh sync /path/to/save"
        echo "  stop      Kill loop + close Akash deployment"
        exit 1
        ;;
esac
