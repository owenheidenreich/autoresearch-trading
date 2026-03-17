#!/usr/bin/env bash
# ===========================================================================
# Akash H100 — Two Commands
# ===========================================================================
#   ./deploy.sh boot    → Deploy H100 container on Akash, wait for SSH
#   ./deploy.sh start   → Upload code + data, start autoresearch loop
#
# Optional:
#   ./deploy.sh sync    → Foreground sync (auto-sync runs with start)
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

    # Upload best_train.py if it exists — architecture lock reference
    if [[ -f "$PROJECT_ROOT/training/best_train.py" ]]; then
        log "Uploading best_train.py (architecture lock reference)..."
        scp_cmd "$PROJECT_ROOT/training/best_train.py" "root@$SSH_HOST:/root/best_train.py"
    fi

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

    # Auto-launch sync in background — no more forgetting to run sync in 2nd terminal
    local sync_dest="$PROJECT_ROOT/results/run-$(date +%Y-%m-%d)"
    mkdir -p "$sync_dest"
    log "Starting auto-sync to $sync_dest ..."
    _run_sync "$sync_dest" > "$sync_dest/sync.log" 2>&1 &
    local sync_pid=$!
    echo "$sync_pid" > "$PROJECT_ROOT/.sync-pid"

    log ""
    log "=== LOOP RUNNING ==="
    log "Auto-sync: PID $sync_pid → $sync_dest (log: sync.log)"
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

    # Rich dashboard via heredoc Python script (avoids nested quoting)
    ssh_cmd python3 - << 'PYEOF'
import json, subprocess, os, textwrap

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except:
        return ""

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

pid = run("pgrep -f run_loop.py")
if pid:
    pid_line = pid.split("\n")[0]
    uptime = run(f"ps -o etime= -p {pid_line}").strip()
    print(f"  Loop: RUNNING (PID {pid_line}, uptime {uptime})")
else:
    print(f"  Loop: NOT RUNNING")

# --- Progress ---
if os.path.exists("/root/status.json"):
    with open("/root/status.json") as f:
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
    print(f"  Best Score: {best:.4f}")
    print(f"  Results:    {kept} kept / {failed} failed / {total} total")
    print(f"  Time Left:  {remaining:.1f}h")

# --- Experiment History ---
bar("Experiment History")
if os.path.exists("/root/experiments.jsonl"):
    with open("/root/experiments.jsonl") as f:
        lines = f.readlines()

    # Table header
    print(f"  {'#':>3}  {'Score':>8}  {'PF':>5}  {'TPD':>5}  {'Sharpe':>7}  {'WR':>5}  {'Kept':>4}  Change")
    print(f"  {'---':>3}  {'-----':>8}  {'--':>5}  {'---':>5}  {'------':>7}  {'--':>5}  {'----':>4}  ------")

    for line in lines:
        try:
            e = json.loads(line)
            eid = e.get("experiment_id", "?")
            score = e.get("score", -999)
            kept_flag = e.get("kept", False)
            err = e.get("error", "")
            change = e.get("change_summary", "")

            # Truncate change summary to fit
            if change:
                change = change.replace("\n", " ")[:40]

            if score == -999 and err:
                # Safety rejection or crash
                reason = err.split("\n")[0][:40]
                print(f"  {eid:>3}  {'FAIL':>8}  {'':>5}  {'':>5}  {'':>7}  {'':>5}  {'':>4}  {reason}")
            else:
                pf = e.get("profit_factor", 0)
                tpd = e.get("trades_per_day", 0)
                sharpe = e.get("trade_sharpe", 0)
                wr = e.get("win_rate", 0)
                mark = " <--" if kept_flag else ""
                print(f"  {eid:>3}  {score:>8.3f}  {pf:>5.2f}  {tpd:>5.1f}  {sharpe:>7.2f}  {wr:>4.0%}  {'YES' if kept_flag else '':>4}{mark}")
        except:
            pass
else:
    print("  (no experiments yet)")

# --- Latest Experiment Detail ---
if os.path.exists("/root/experiments.jsonl"):
    with open("/root/experiments.jsonl") as f:
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
    local dest="${EXTRA_ARGS:-$PROJECT_ROOT/results/run-$(date +%Y-%m-%d)}"
    mkdir -p "$dest"
    log "Downloading results to $dest ..."

    for f in experiments.jsonl status.json loop.log best_train.py train.py trade_log.csv diagnostics.log best_model.pt; do
        scp_cmd "root@$SSH_HOST:/root/$f" "$dest/$f" 2>/dev/null && \
            log "  $f ($(du -h "$dest/$f" | cut -f1))" || \
            rm -f "$dest/$f"
    done

    # Preserve artifacts in training/ for next deployment
    if [[ -f "$dest/best_model.pt" ]]; then
        cp "$dest/best_model.pt" "$PROJECT_ROOT/training/best_model.pt"
        log "  ↳ Copied best_model.pt → training/ (warm-start for next run)"
    fi
    if [[ -f "$dest/best_train.py" ]]; then
        cp "$dest/best_train.py" "$PROJECT_ROOT/training/best_train.py"
        # best_train.py IS the best code — use it as train.py for next run
        cp "$dest/best_train.py" "$PROJECT_ROOT/training/train.py"
        log "  ↳ Copied best_train.py → training/train.py + best_train.py"
    fi

    log "Done! Results in: $dest"
}

# ===================================================================
# _run_sync — core sync loop (used by cmd_start background + cmd_sync foreground)
# Usage: _run_sync <dest_dir>
# ===================================================================
_run_sync() {
    local dest="$1"
    mkdir -p "$dest"

    local last_kept=-1
    local last_total=-1
    local poll_interval=30

    log "=== AUTO-SYNC ==="
    log "  Remote:   $SSH_HOST:$SSH_PORT"
    log "  Save to:  $dest"
    log "  Polling every ${poll_interval}s"
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
            # Copy best model + code to training/ for warm-start continuity
            if [[ -f "$dest/best_model.pt" ]]; then
                cp "$dest/best_model.pt" "$PROJECT_ROOT/training/best_model.pt"
                log "  ↳ Updated training/best_model.pt"
            fi
            if [[ -f "$dest/best_train.py" ]]; then
                cp "$dest/best_train.py" "$PROJECT_ROOT/training/best_train.py"
                cp "$dest/best_train.py" "$PROJECT_ROOT/training/train.py"
                log "  ↳ Updated training/train.py + best_train.py"
            fi
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
            # Final warm-start copy
            if [[ -f "$dest/best_model.pt" ]]; then
                cp "$dest/best_model.pt" "$PROJECT_ROOT/training/best_model.pt"
                log "  ↳ Updated training/best_model.pt"
            fi
            if [[ -f "$dest/best_train.py" ]]; then
                cp "$dest/best_train.py" "$PROJECT_ROOT/training/best_train.py"
                cp "$dest/best_train.py" "$PROJECT_ROOT/training/train.py"
                log "  ↳ Updated training/train.py + best_train.py"
            fi
            log "All results saved to: $dest"
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

    local dest="${EXTRA_ARGS:-$PROJECT_ROOT/results/run-$(date +%Y-%m-%d)}"
    log "Running foreground sync (Ctrl-C to stop)..."
    _run_sync "$dest"
}

cmd_stop() {
    load_state
    echo ""
    echo "This will: kill loop → download results → close deployment"
    read -p "Continue? [y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || { log "Aborted."; exit 0; }

    # Kill background sync if running
    if [[ -f "$PROJECT_ROOT/.sync-pid" ]]; then
        local spid
        spid=$(cat "$PROJECT_ROOT/.sync-pid")
        kill "$spid" 2>/dev/null || true
        rm -f "$PROJECT_ROOT/.sync-pid"
        log "Stopped background sync (PID $spid)"
    fi

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
        echo "  sync      Foreground sync (auto-sync runs automatically with start)"
        echo "            Optional: ./deploy.sh sync /path/to/save"
        echo "  stop      Kill loop + close Akash deployment"
        exit 1
        ;;
esac
