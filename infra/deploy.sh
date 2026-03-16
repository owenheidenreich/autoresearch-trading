#!/usr/bin/env bash
# ===========================================================================
# Akash H100 — Two Commands
# ===========================================================================
#   ./deploy.sh boot    → Deploy H100 container on Akash, wait for SSH
#   ./deploy.sh start   → Upload code + data, start autoresearch loop
#
# Optional:
#   ./deploy.sh sync    → Auto-download results as improvements are found
#   ./deploy.sh ssh     → Drop into SSH shell on the H100
#   ./deploy.sh logs    → Tail the autoresearch loop log
#   ./deploy.sh status  → Show GPU, loop PID, last experiment score
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
# Only require API key for commands that need it (boot, start)
ANTHROPIC_KEY="${ANTHROPIC_API_KEY:-}"

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
    log "=== BOOT: Creating Akash H100 Deployment ==="
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
    log "Waiting 45s for H100 bids..."
    sleep 45

    BIDS_JSON=$(provider-services query market bid list \
        --owner "$AKASH_OWNER" --dseq "$DSEQ" \
        --node "$AKASH_NODE" --output json 2>&1)

    PROVIDER=$(echo "$BIDS_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
bids = [b for b in data.get('bids', []) if b['bid']['state'] == 'open']
if not bids: bids = data.get('bids', [])
if not bids: print(''); sys.exit(0)
bids.sort(key=lambda b: float(b['bid']['price']['amount']))
print(bids[0]['bid']['id']['provider'])
" 2>/dev/null)
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
    log "=== H100 READY ==="
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

    # Upload data.pt to the path prepare.py expects: ~/.cache/autoresearch-trading/features/
    [[ -f "$DATA_PT" ]] || die "data.pt not found: $DATA_PT"
    log "Uploading data.pt ($(du -h "$DATA_PT" | cut -f1))..."
    ssh_cmd "mkdir -p /root/.cache/autoresearch-trading/features"
    scp_cmd "$DATA_PT" "root@$SSH_HOST:/root/.cache/autoresearch-trading/features/data.pt"

    # Upload code + launcher + watchdog
    log "Uploading train.py, prepare.py, program.md, run_loop.py, start_loop.sh, watchdog.sh..."
    scp_cmd \
        "$PROJECT_ROOT/training/train.py" \
        "$PROJECT_ROOT/training/prepare.py" \
        "$PROJECT_ROOT/training/program.md" \
        "$PROJECT_ROOT/training/run_loop.py" \
        "$SCRIPT_DIR/start_loop.sh" \
        "$SCRIPT_DIR/watchdog.sh" \
        "root@$SSH_HOST:/root/"

    # Upload best_model.pt if it exists — warm-start from previous training run
    if [[ -f "$PROJECT_ROOT/training/best_model.pt" ]]; then
        log "Uploading best_model.pt ($(du -h "$PROJECT_ROOT/training/best_model.pt" | cut -f1)) for warm-start..."
        scp_cmd "$PROJECT_ROOT/training/best_model.pt" "root@$SSH_HOST:/root/best_model.pt"
    else
        log "No best_model.pt found — starting from scratch"
    fi

    # Verify
    log "Files on H100:"
    ssh_cmd "ls -lh /root/*.py /root/*.sh /root/.cache/autoresearch-trading/features/data.pt"

    # Pre-flight dry-run on H100 — verify data, GPU, API before consuming time
    log "Running pre-flight checks on H100..."
    if ! ssh_cmd "cd /root && ANTHROPIC_API_KEY='$ANTHROPIC_KEY' /opt/conda/bin/python -u run_loop.py --dry-run"; then
        die "Pre-flight failed on H100. Fix issues before running."
    fi
    log "Pre-flight passed!"

    # Launch loop via start_loop.sh — pass API key as env var (not sed)
    log "Starting autoresearch loop... $LOOP_ARGS"
    ssh_cmd "chmod +x /root/start_loop.sh && ANTHROPIC_API_KEY='$ANTHROPIC_KEY' /root/start_loop.sh $LOOP_ARGS"

    log ""
    log "=== LOOP RUNNING ==="
    log "Sync:   ./deploy.sh sync    ← run in 2nd terminal to auto-download results"
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
    ssh_cmd bash -c "'
        echo \"=== GPU ===\"
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
        echo \"\"
        echo \"=== Loop ===\"
        if pgrep -f run_loop.py > /dev/null 2>&1; then
            echo \"PID: \$(pgrep -f run_loop.py)\"
            echo \"Uptime: \$(ps -o etime= -p \$(pgrep -f run_loop.py) 2>/dev/null)\"
        else
            echo \"NOT RUNNING\"
        fi
        echo \"\"
        echo \"=== Last 3 experiments ===\"
        tail -3 /root/experiments.jsonl 2>/dev/null | python3 -c \"
import sys, json
for line in sys.stdin:
    e = json.loads(line)
    print(f\\\"  #{e.get(\\\\\\\"iteration\\\\\\\",\\\\\\\"?\\\\\\\")}: score={e.get(\\\\\\\"score\\\\\\\",\\\\\\\"?\\\\\\\"):.4f}  sharpe={e.get(\\\\\\\"val_sharpe\\\\\\\",\\\\\\\"?\\\\\\\"):.4f}  trades/day={e.get(\\\\\\\"trades_per_day\\\\\\\",\\\\\\\"?\\\\\\\"):.2f}\\\")
\" 2>/dev/null || echo \"  (no experiments yet)\"
        echo \"\"
        echo \"=== Last 5 log lines ===\"
        tail -5 /root/loop.log 2>/dev/null || echo \"(no log)\"
    '"
}

cmd_download() {
    load_state
    local dest="${EXTRA_ARGS:-$PROJECT_ROOT/results/run-$(date +%Y-%m-%d)}"
    mkdir -p "$dest"
    log "Downloading results to $dest ..."

    for f in experiments.jsonl status.json loop.log best_train.py train.py trade_log.csv diagnostics.log best_model.pt; do
        scp_cmd "root@$SSH_HOST:/root/$f" "$dest/$f" 2>/dev/null && \
            log "  $f ($(du -h "$dest/$f" | cut -f1))" || \
            rm -f "$dest/$f"
    done

    # Preserve best_model.pt in training/ for next deployment's warm-start
    if [[ -f "$dest/best_model.pt" ]]; then
        cp "$dest/best_model.pt" "$PROJECT_ROOT/training/best_model.pt"
        log "  ↳ Copied best_model.pt → training/ (warm-start for next run)"
    fi

    log "Done! Results in: $dest"
}

cmd_sync() {
    load_state
    local dest="${EXTRA_ARGS:-$PROJECT_ROOT/results/run-$(date +%Y-%m-%d)}"
    mkdir -p "$dest"

    local last_kept=-1
    local last_total=-1
    local poll_interval=60

    log "=== AUTO-SYNC ==="
    log "  Remote:   $SSH_HOST:$SSH_PORT"
    log "  Save to:  $dest"
    log "  Polling every ${poll_interval}s  (Ctrl-C to stop)"
    log ""

    while true; do
        # Fetch remote status.json (small file, cheap over SSH)
        local raw
        raw=$(ssh_cmd "cat /root/status.json 2>/dev/null") 2>/dev/null || {
            log "  ⚠ SSH unreachable — retrying in ${poll_interval}s"
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
            log "  Baseline: kept=$kept total=$total best=$best_score phase=$phase"
            sleep "$poll_interval"
            continue
        fi

        # --- New improvement: full download (model + trade log + logs) ---
        if [[ "$kept" -gt "$last_kept" ]]; then
            log "★ IMPROVEMENT #$kept (score=$best_score) — downloading all results..."
            for f in experiments.jsonl status.json loop.log best_train.py \
                     trade_log.csv diagnostics.log best_model.pt; do
                if scp_cmd "root@$SSH_HOST:/root/$f" "$dest/$f" 2>/dev/null; then
                    log "  ↓ $f  ($(du -h "$dest/$f" | cut -f1))"
                fi
            done
            last_kept=$kept
            last_total=$total
            log "  Synced. Best score: $best_score"

        # --- Experiment finished but no improvement: sync logs only ---
        elif [[ "$total" -gt "$last_total" ]]; then
            log "  Exp #$total done (not kept). Syncing logs..."
            for f in experiments.jsonl status.json loop.log diagnostics.log; do
                scp_cmd "root@$SSH_HOST:/root/$f" "$dest/$f" 2>/dev/null || true
            done
            last_total=$total

        # --- Heartbeat: show sync is alive even when nothing changed ---
        else
            local remaining
            remaining=$(echo "$raw" | python3 -c "import sys,json; print(f\"{json.load(sys.stdin).get('time_remaining_h','?'):.1f}h\")" 2>/dev/null || echo "?")
            log "  ♻ Polling... exp=$total kept=$kept best=$best_score remaining=$remaining"
        fi

        # --- Loop completed: final full download + exit ---
        if [[ "$phase" == "completed" ]]; then
            log ""
            log "=== LOOP COMPLETED (score=$best_score, kept=$kept) ==="
            log "Final download..."
            for f in experiments.jsonl status.json loop.log best_train.py \
                     train.py trade_log.csv diagnostics.log best_model.pt; do
                if scp_cmd "root@$SSH_HOST:/root/$f" "$dest/$f" 2>/dev/null; then
                    log "  ↓ $f  ($(du -h "$dest/$f" | cut -f1))"
                fi
            done
            log "All results saved to: $dest"
            break
        fi

        sleep "$poll_interval"
    done
}

cmd_stop() {
    load_state
    echo ""
    echo "This will: kill loop → download results → close deployment"
    read -p "Continue? [y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || { log "Aborted."; exit 0; }

    log "Killing loop..."
    ssh_cmd "pkill -f run_loop.py 2>/dev/null || true"

    # Wait for loop to actually die (up to 15s) — prevents partial file downloads
    for _ in $(seq 1 15); do
        if ! ssh_cmd "pgrep -f run_loop.py" &>/dev/null; then
            break
        fi
        sleep 1
    done
    # Force kill if still alive
    ssh_cmd "pkill -9 -f run_loop.py 2>/dev/null || true" 2>/dev/null
    sleep 1

    log "Downloading results before closing..."
    cmd_download

    log "Closing Akash deployment DSEQ=$DSEQ..."
    provider-services tx deployment close \
        --dseq "$DSEQ" --from "$AKASH_FROM" --yes 2>&1
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
        echo "  status    GPU, loop PID, last experiments"
        echo "  download  Download results once (experiments, logs, trade_log, best model)"
        echo "            Optional: ./deploy.sh download /path/to/save"
        echo "  sync      Auto-download results as improvements are found (run in 2nd terminal)"
        echo "            Optional: ./deploy.sh sync /path/to/save"
        echo "  stop      Kill loop + close Akash deployment"
        exit 1
        ;;
esac
