#!/usr/bin/env bash
# ===========================================================================
# Akash GPU — v2 Deployment  (post harness-integrity repair 2026-04-17)
# ===========================================================================
# CV pipeline (Claude drives each experiment):
#   ./deploy.sh boot                    → Deploy GPU container on Akash
#   ./deploy.sh start                   → Upload v2 codebase + data, install deps
#   ./deploy.sh run_screen_latest ID    → Single-fold parity debug (no artifact)
#   ./deploy.sh run_screen_mini ID      → 3-fold regime triage (no artifact)
#   ./deploy.sh run_cv ID               → Full 5-fold CV, emits CV_EVAL artifact
#   ./deploy.sh run_final_train SRC_ID  → Train deployable model from chosen CV
#                                          (only FINAL_TRAIN artifacts promotable)
#
# Promotion:
#   python3 -m v2.ops.model_manage keep    (FINAL_TRAIN only; CV_EVAL rejected)
#   python3 -m v2.ops.model_manage revert
#
# Utilities:
#   ./deploy.sh ssh         → SSH into the H100
#   ./deploy.sh status      → GPU + experiment dashboard
#   ./deploy.sh download    → Download all v2 results + artifacts
#   ./deploy.sh stop        → Kill experiment + close Akash deployment
# ===========================================================================
set -euo pipefail
trap '_rc=$?; log "FATAL: command failed (exit $_rc) at line ${LINENO}: ${BASH_COMMAND}"; exit $_rc' ERR

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
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SDL_FILE="$SCRIPT_DIR/deploy-autoresearch.yaml"
DATA_PT="${V2_DATA_PATH:-$PROJECT_ROOT/v2/data.pt}"
STATE_FILE="$PROJECT_ROOT/.deploy-state"

SSH_PASS="${DEPLOY_SSH_PASS:-autoresearch2026}"
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

scp_retry() {
    local attempt
    for attempt in 1 2 3; do
        if scp_cmd "$@"; then return 0; fi
        (( attempt < 3 )) || return 1
        log "SCP attempt $attempt/3 failed — retrying in $((attempt * 10))s..."
        sleep $((attempt * 10))
    done
}

# Auto-append CV run results to results.tsv using the CVReport schema.
# Usage: _append_results_tsv <exp_id> <run_output>
_append_results_tsv() {
    local exp_id="$1"
    local output="$2"
    local results_file="$PROJECT_ROOT/v2/results.tsv"

    # Ensure new-schema header exists
    if [[ ! -f "$results_file" ]]; then
        python3 -c "from v2.core.cv_report import header_line; print(header_line())" > "$results_file"
    fi

    # Extract RESULTS_JSON line from output
    local results_json
    results_json=$(echo "$output" | grep '^RESULTS_JSON:' | sed 's/^RESULTS_JSON://' | tail -1)
    if [[ -z "$results_json" ]]; then
        log "WARNING: No RESULTS_JSON in output — skipping results.tsv append"
        return
    fi

    # Build TSV row using scope-named fields from CVReport
    local tsv_line
    tsv_line=$(python3 -c "
import json, sys
r = json.loads(sys.argv[1])
exp_id = sys.argv[2]

# Only 'full' mode writes to results.tsv as official CV
mode = r.get('screening_mode', 'unknown')
folds = r.get('per_fold_scores', [])
pfs = ','.join(f'{x:.3f}' for x in folds)
any_gate = r.get('any_fold_gate_failure', False)
error = r.get('error', '')

if error:
    desc = f'ERROR={error}'
elif any_gate:
    desc = f'GATE_FAILURE mode={mode} folds=[{pfs}]'
else:
    desc = f'mode={mode} folds=[{pfs}]'

cols = [
    exp_id,
    mode,
    'revert',  # flipped to 'keep' only by model_manage.keep()
    f\"{r.get('stability_score', -999.0):.6f}\",
    f\"{r.get('pooled_profit_factor', 0.0):.4f}\",
    f\"{r.get('pooled_max_account_drawdown', 0.0):.4f}\",
    str(r.get('pooled_total_trades', 0)),
    str(r.get('pooled_traded_days', 0)),
    'true' if any_gate else 'false',
    f'[{pfs}]',
    desc,
]
print('\t'.join(cols))
" "$results_json" "$exp_id" 2>/dev/null)

    if [[ -n "$tsv_line" ]]; then
        echo "$tsv_line" >> "$results_file"
        log "Appended $exp_id to results.tsv (stability_score=$(echo "$tsv_line" | cut -f4))"
    else
        log "WARNING: Failed to parse RESULTS_JSON — results.tsv not updated"
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

run_local_pre_run_gate() {
    [[ -f "$DATA_PT" ]] || die "data.pt not found: $DATA_PT"
    log "Running local pre-GPU integrity gate..."
    python3 -m v2.ops.pre_run_gate --data "$DATA_PT"
}

# ===================================================================
# BOOT — deploy H100 on Akash, wait until SSH is accessible
# ===================================================================
cmd_boot() {
    log "=== BOOT: Creating Akash GPU Deployment ==="
    [[ -f "$SDL_FILE" ]] || die "SDL not found: $SDL_FILE"

    # 1. Submit deployment TX
    # BME (Mainnet 17): deposits are now in ACT (USD-pegged compute credit), not AKT.
    # Mint ACT first: provider-services tx bme mint-act <amount>uakt
    # Use `deploy.sh fund <ACT>` to add funds before starting experiments.
    DEPOSIT_ACT="${DEPOSIT_ACT:-5}"
    DEPOSIT_UACT=$((DEPOSIT_ACT * 1000000))

    # Auto-mint ACT if balance insufficient.
    # BME min_mint = 10 ACT. Exchange rate ~7 AKT per 1 ACT (varies).
    # Query on-chain balance and mint if needed.
    local _act_bal
    _act_bal=$(provider-services query bank balances "$AKASH_OWNER" \
        --node "$AKASH_NODE" --output json 2>/dev/null | \
        python3 -c "
import sys, json
d = json.load(sys.stdin)
for b in d.get('balances', []):
    if b['denom'] == 'uact':
        print(b['amount'])
        sys.exit(0)
print('0')
" 2>/dev/null || echo "0")
    if [[ "$_act_bal" -lt "$DEPOSIT_UACT" ]]; then
        local _deficit_act=$(( (DEPOSIT_UACT - _act_bal + 999999) / 1000000 ))
        # BME minimum mint is 10 ACT. Mint at least 15 ACT to have buffer.
        local _mint_act=$(( _deficit_act > 15 ? _deficit_act : 15 ))
        # Exchange rate ~7 AKT per ACT (query vault for precision)
        local _akt_per_act
        _akt_per_act=$(provider-services query bme vault-state --node "$AKASH_NODE" --output json 2>/dev/null | \
            python3 -c "
import sys, json
d = json.load(sys.stdin)
vs = d.get('vault_state', d)
uakt = int([b['amount'] for b in vs['balances'] if b['denom']=='uakt'][0])
uact = int([b['amount'] for b in vs['balances'] if b['denom']=='uact'][0])
print(int(uakt / uact) + 2)  # +2 for spread/rounding safety
" 2>/dev/null || echo "8")
        local _mint_uakt=$(( _mint_act * _akt_per_act * 1000000 ))
        log "ACT balance insufficient ($((_act_bal/1000000)) ACT < ${DEPOSIT_ACT} ACT). Minting ${_mint_act} ACT from $((_mint_uakt/1000000)) AKT..."
        local _mint_out _mint_exit=0 _mint_code
        _mint_out=$(provider-services tx bme mint-act "${_mint_uakt}uakt" \
            --from "$AKASH_FROM" --yes --output json 2>&1) || _mint_exit=$?
        if [[ $_mint_exit -ne 0 ]]; then
            log "WARNING: provider-services mint-act exited with code $_mint_exit"
            echo "$_mint_out"
        fi
        _mint_code=$(echo "$_mint_out" | python3 -c "import sys,json; print(json.load(sys.stdin).get('code',1))" 2>/dev/null || echo "1")
        [[ "$_mint_code" == "0" ]] || { echo "$_mint_out"; die "ACT mint failed (exit=$_mint_exit, tx_code=$_mint_code). Raw output above."; }
        log "Minted ${_mint_act} ACT — waiting 10s for chain confirmation..."
        sleep 10
    fi

    log "Submitting deployment TX (deposit=${DEPOSIT_ACT} ACT ~\$${DEPOSIT_ACT} — use 'fund' to add more)..."
    local _deploy_exit=0
    TX_OUTPUT=$(provider-services tx deployment create "$SDL_FILE" \
        --deposit "${DEPOSIT_UACT}uact" \
        --from "$AKASH_FROM" --yes --output json 2>&1) || _deploy_exit=$?
    [[ $_deploy_exit -eq 0 ]] || { echo "$TX_OUTPUT"; die "Deployment create failed (exit=$_deploy_exit)"; }

    TXHASH=$(echo "$TX_OUTPUT" | grep -o '"txhash":"[^"]*"' | head -1 | cut -d'"' -f4)
    [[ -n "$TXHASH" ]] || { echo "$TX_OUTPUT"; die "No txhash returned"; }
    log "TX: $TXHASH"

    # 2. Confirm TX
    sleep 8
    local _tx_exit=0
    TX_RESULT=$(provider-services query tx "$TXHASH" --node "$AKASH_NODE" --output json 2>&1) || _tx_exit=$?
    [[ $_tx_exit -eq 0 ]] || { echo "$TX_RESULT" | head -30; die "TX query failed (exit=$_tx_exit)"; }
    TX_CODE=$(echo "$TX_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('code',1))" 2>/dev/null || echo "1")
    [[ "$TX_CODE" == "0" ]] || { echo "$TX_RESULT" | head -30; die "TX failed (code=$TX_CODE)"; }
    log "TX confirmed"

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
    local _lease_exit=0
    LEASE_TX=$(provider-services tx market lease create \
        --dseq "$DSEQ" --gseq "$GSEQ" --oseq "$ASEQ" \
        --provider "$PROVIDER" --from "$AKASH_FROM" \
        --yes --output json 2>&1) || _lease_exit=$?
    [[ $_lease_exit -eq 0 ]] || { echo "$LEASE_TX"; die "Lease create failed (exit=$_lease_exit)"; }
    LEASE_HASH=$(echo "$LEASE_TX" | grep -o '"txhash":"[^"]*"' | head -1 | cut -d'"' -f4)
    [[ -n "$LEASE_HASH" ]] || { echo "$LEASE_TX"; die "No lease txhash returned"; }
    log "Lease TX: $LEASE_HASH"
    sleep 8

    # 5b. Verify lease TX succeeded (prevents silent failures)
    local _lquery_exit=0
    LEASE_RESULT=$(provider-services query tx "$LEASE_HASH" --node "$AKASH_NODE" --output json 2>&1) || _lquery_exit=$?
    [[ $_lquery_exit -eq 0 ]] || { echo "$LEASE_RESULT" | head -30; die "Lease TX query failed (exit=$_lquery_exit)"; }
    LEASE_CODE=$(echo "$LEASE_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('code',1))" 2>/dev/null || echo "1")
    [[ "$LEASE_CODE" == "0" ]] || { echo "$LEASE_RESULT" | head -30; die "Lease TX failed (code=$LEASE_CODE). Provider may have rejected."; }
    log "Lease confirmed"

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

    # 8. Wait for SSH + verify GPU (auto-retry: close and re-deploy if SSH fails)
    log "Waiting for SSH (will auto-retry boot once if SSH fails)..."
    local ssh_ok=false
    for _ in $(seq 1 90); do
        if ssh_cmd "echo OK" &>/dev/null; then
            ssh_ok=true
            break
        fi
        sleep 5
        printf "."
    done

    if [[ "$ssh_ok" != "true" ]]; then
        log ""
        log "WARNING: SSH not available after 7.5 minutes — closing deployment and retrying boot..."
        provider-services tx deployment close \
            --dseq "$DSEQ" --from "$AKASH_FROM" \
            --gas auto --gas-adjustment 1.5 --gas-prices 0.025uakt \
            --yes --output json 2>&1 || true
        rm -f "$STATE_FILE"
        sleep 10
        # Guard against infinite recursion: only retry once
        if [[ "${_BOOT_RETRY:-0}" -ge 1 ]]; then
            die "SSH failed after retry. Check Akash Console or try a different provider."
        fi
        export _BOOT_RETRY=1
        log "Retrying boot (attempt 2)..."
        cmd_boot
        return
    fi

    log ""
    log "SSH ready!"
    GPU=$(ssh_cmd "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>/dev/null || echo "unknown")
    log ""
    log "=== GPU READY ==="
    log "GPU:  $GPU"
    log "SSH:  ssh -p $SSH_PORT root@$SSH_HOST  (pass: $SSH_PASS)"
    log ""
    log "Next: ./deploy.sh start"
}

# ===================================================================
# START — upload v2 codebase + data, prepare for experiments
# ===================================================================
cmd_start() {
    load_state
    log "=== START: Uploading v2 codebase + data ==="

    wait_for_ssh

    # Upload exact local workspace snapshot (no git/network pulls on remote).
    local bundle source_git_sha source_dirty_count bundle_sha
    bundle="/tmp/autoresearch-v2-workspace-$$.tgz"
    rm -f "$bundle"  # Clean up any stale file from previous run
    source_git_sha=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo "nogit")
    source_dirty_count=$(git -C "$PROJECT_ROOT" status --porcelain 2>/dev/null | wc -l | tr -d ' ' || echo "0")
    log "Packaging v2 workspace snapshot..."
    tar -h -czf "$bundle" -C "$PROJECT_ROOT" \
        --no-mac-metadata --no-xattrs \
        --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
        --exclude='*.pt' --exclude='results' --exclude='archive' \
        --exclude='archive_quarantine' \
        --exclude='v2/artifacts' \
        v2 pyproject.toml CLAUDE.md
    bundle_sha=$(shasum -a 256 "$bundle" | awk '{print $1}')
    log "Uploading workspace snapshot ($(du -h "$bundle" | cut -f1), sha256=$bundle_sha)..."
    scp_retry "$bundle" "root@$SSH_HOST:/root/v2-workspace.tgz"
    rm -f "$bundle"
    ssh_cmd "rm -rf /root/v2 && mkdir -p /root && tar -xzf /root/v2-workspace.tgz -C /root 2>/dev/null"
    ssh_cmd "mkdir -p /root/v2/models /root/v2/state /root/v2/output && rm -f /root/v2-workspace.tgz"

    # Integrity check: confirm remote train.py matches local snapshot.
    local local_train_sha remote_train_sha
    local_train_sha=$(shasum -a 256 "$PROJECT_ROOT/v2/train.py" | awk '{print $1}')
    remote_train_sha=$(ssh_cmd "sha256sum /root/v2/train.py | awk '{print \$1}'" 2>/dev/null || true)
    [[ "$remote_train_sha" == "$local_train_sha" ]] || die "Snapshot integrity check failed (v2/train.py hash mismatch)"

    # Record deploy source metadata on remote for auditability.
    local deployed_at
    deployed_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    ssh_cmd "printf '%s\n' '{\"bundle_sha256\":\"$bundle_sha\",\"git_sha\":\"$source_git_sha\",\"dirty_files\":$source_dirty_count,\"deployed_at\":\"$deployed_at\",\"source\":\"v2_local_workspace_snapshot\"}' > /root/deploy_source.json"

    # Upload the selected dataset artifact to v2/data.pt on the GPU node
    [[ -f "$DATA_PT" ]] || die "data.pt not found: $DATA_PT"
    log "Uploading dataset $(basename "$DATA_PT") ($(du -h "$DATA_PT" | cut -f1))..."
    scp_retry "$DATA_PT" "root@$SSH_HOST:/root/v2/data.pt"

    # Verify data.pt upload integrity via SHA256 comparison
    if [[ -f "${DATA_PT}.sha256" ]]; then
        local local_hash
        local_hash=$(cat "${DATA_PT}.sha256" | tr -d '[:space:]')
        local remote_hash
        remote_hash=$(ssh_cmd "sha256sum /root/v2/data.pt | cut -d' ' -f1" 2>/dev/null | tr -d '[:space:]')
        if [[ -n "$remote_hash" && "$local_hash" != "$remote_hash" ]]; then
            die "data.pt upload CORRUPTED! Local: ${local_hash:0:16}  Remote: ${remote_hash:0:16}"
        elif [[ -n "$remote_hash" ]]; then
            log "data.pt integrity verified (hash: ${local_hash:0:16})"
        else
            log "WARNING: Could not verify data.pt hash on remote (sha256sum unavailable)"
        fi
    else
        log "WARNING: No data.pt.sha256 sidecar — skipping integrity check"
    fi

    # Upload training-stripped sidecars (drops replay-only fields + float16
    # downcast). 3.5GB full → 518MB compressed bundle. Uses zstd (better
    # than gzip on numeric data). See v2/ops/strip_sidecars.py for field
    # list and precision verification. Full sidecars stay on disk for replay.
    local sidecar_rel sidecar_abs sidecar_bundle remote_sidecar_dir stripped_dir
    sidecar_rel=$(python3 - <<'PY' "$DATA_PT"
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(d.get("metadata", {}).get("chain_sidecar_dir", ""))
PY
)
    sidecar_rel=$(echo "$sidecar_rel" | tr -d '[:space:]')
    if [[ -n "$sidecar_rel" ]]; then
        sidecar_abs="$PROJECT_ROOT/$sidecar_rel"
        remote_sidecar_dir="/root/$sidecar_rel"
        [[ -d "$sidecar_abs" ]] || die "Dataset expects sidecar dir but it is missing: $sidecar_abs"

        stripped_dir="/tmp/autoresearch-stripped-sidecars-$$"
        sidecar_bundle="/tmp/autoresearch-v2-sidecars-$$.tar.zst"
        log "Stripping sidecars for training-only upload (field strip + float16 downcast)..."
        python3 -m v2.ops.strip_sidecars "$sidecar_abs" "$stripped_dir" || die "Sidecar stripping failed"

        log "Packaging stripped sidecars (zstd)..."
        tar --no-mac-metadata --no-xattrs -cf - -C "$(dirname "$stripped_dir")" "$(basename "$stripped_dir")" | zstd -3 -T0 -o "$sidecar_bundle"
        rm -rf "$stripped_dir"

        log "Uploading stripped sidecars ($(du -sh "$sidecar_bundle" | cut -f1))..."
        scp_retry "$sidecar_bundle" "root@$SSH_HOST:/root/v2-sidecars.tar.zst"
        rm -f "$sidecar_bundle"

        # Unpack on GPU — zstd installed by SDL command, rename stripped dir to expected path
        ssh_cmd "command -v zstd >/dev/null 2>&1" || die "zstd not available on remote (SDL command may have failed)"
        ssh_cmd "rm -rf '$remote_sidecar_dir' && mkdir -p '$(dirname "$remote_sidecar_dir")'"
        log "Decompressing sidecars on remote..."
        ssh_cmd "zstd -d /root/v2-sidecars.tar.zst -o /root/v2-sidecars.tar"
        log "Extracting sidecar archive on remote..."
        # Suppress stderr: macOS bsdtar embeds LIBARCHIVE.xattr metadata that GNU tar warns about.
        # Warnings cause non-zero exit on some tar versions. Verify extraction below.
        ssh_cmd "tar -xf /root/v2-sidecars.tar -C /tmp 2>/dev/null || true"
        ssh_cmd "test -d /tmp/autoresearch-stripped-sidecars-*" || die "Sidecar tar extraction failed on remote"
        ssh_cmd "mv /tmp/autoresearch-stripped-sidecars-* '$remote_sidecar_dir'"
        ssh_cmd "rm -f /root/v2-sidecars.tar.zst /root/v2-sidecars.tar"
        ssh_cmd "test -d '$remote_sidecar_dir'" || die "Remote sidecar upload failed: $remote_sidecar_dir missing"

        # Clean up macOS AppleDouble resource fork files (._*) from extraction
        ssh_cmd "find '$remote_sidecar_dir' -name '._*' -delete 2>/dev/null || true"

        # Verify sidecar count matches local (use source dir — stripped dir is already cleaned up)
        local local_count remote_count
        local_count=$(find "$sidecar_abs" -name '*.pt' ! -name '._*' 2>/dev/null | wc -l | tr -d ' ')
        remote_count=$(ssh_cmd "find '$remote_sidecar_dir' -name '*.pt' ! -name '._*' 2>/dev/null | wc -l" | tr -d ' ')
        if [ "$local_count" != "$remote_count" ]; then
            die "Sidecar count mismatch: local=$local_count remote=$remote_count"
        fi
        log "Sidecar count verified: $remote_count files"
    fi

    # Experiments always train from scratch. Keep the remote model path empty
    # until run_experiment_wf writes a fresh checkpoint for this harness era.
    ssh_cmd "mkdir -p /root/v2/models && rm -f /root/v2/models/model.pt"
    log "Remote v2/models/model.pt cleared (training from scratch; no warm-start upload)"

    # Verify files on remote
    log "Files on remote GPU node:"
    ssh_cmd "ls -lh /root/v2/train.py /root/v2/data.pt /root/deploy_source.json"

    # Verify GPU dependencies (torch, numpy, pandas, scipy).
    # These are installed by the SDL container command BEFORE sshd starts,
    # so they should already be present. If not, attempt a one-shot install.
    log "Verifying GPU dependencies..."
    if ! ssh_cmd "python3 -c 'import torch, numpy, pandas, scipy; print(f\"torch={torch.__version__} numpy={numpy.__version__} pandas={pandas.__version__} scipy={scipy.__version__}\")'"; then
        log "Dependencies missing — installing from requirements-gpu.txt..."
        if ! ssh_cmd "pip3 --version >/dev/null 2>&1"; then
            log "pip3 not found — installing..."
            ssh_cmd "apt-get update -qq && apt-get install -y -qq python3-pip >/dev/null 2>&1" \
                || log "WARNING: pip3 install failed (may not be needed if deps are pre-installed)"
        fi
        ssh_cmd "pip3 install -q numpy pandas scipy" \
            || ssh_cmd "python3 -m pip install -q numpy pandas scipy" \
            || die "Failed to install GPU dependencies."
        # Verify again — fatal if still missing
        ssh_cmd "python3 -c 'import torch, numpy, pandas, scipy'" \
            || die "GPU dependencies still missing after install. Container image may be broken."
    fi
    log "Dependencies verified."

    # Pre-flight: verify GPU, Python, PyTorch, and data on remote node
    log "Running pre-flight checks on remote GPU node..."
    if ! ssh_cmd "python3 /root/v2/ops/preflight.py" \
        && ! ssh_cmd "/opt/conda/bin/python /root/v2/ops/preflight.py"; then
        die "Pre-flight failed on remote GPU node. Fix issues before running."
    fi
    log "Pre-flight passed!"

    log ""
    log "=== GPU READY ==="
    log ""
    log "GPU READY. Run experiments with:"
    log "  ./v2/ops/deploy.sh run_screen_latest exp_NNN   # single-fold triage"
    log "  ./v2/ops/deploy.sh run_cv exp_NNN              # full 5-fold CV"
    log "  ./v2/ops/deploy.sh run_final_train exp_NNN     # produce deployable model"
    log ""
    log "Status: ./deploy.sh status"
    log "SSH:    ./deploy.sh ssh"
}

# ===================================================================
# Convenience commands
# ===================================================================
cmd_fund() {
    load_state
    local amount_act="${EXTRA_ARGS:-5}"
    [[ "$amount_act" =~ ^[0-9]+$ ]] || die "Usage: deploy.sh fund <ACT amount>"
    local amount_uact=$((amount_act * 1000000))
    log "Adding ${amount_act} ACT (~\$${amount_act}) to deployment DSEQ=${DSEQ}..."
    local tx_out
    tx_out=$(provider-services tx escrow deposit \
        deployment "${amount_uact}uact" \
        --dseq "$DSEQ" --from "$AKASH_FROM" \
        --gas auto --gas-adjustment 1.5 --gas-prices 0.025uakt \
        --yes --output json 2>&1)
    local txhash
    txhash=$(echo "$tx_out" | grep -o '"txhash":"[^"]*"' | head -1 | cut -d'"' -f4)
    if [[ -n "$txhash" ]]; then
        log "Deposited ${amount_act} ACT (TX: $txhash)"
    else
        echo "$tx_out"
        die "Escrow deposit failed"
    fi
}

cmd_ssh() {
    load_state
    log "Connecting to H100..."
    SSHPASS="$SSH_PASS" sshpass -e ssh \
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=15 -o LogLevel=ERROR \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -o PubkeyAuthentication=no \
        -p "$SSH_PORT" "root@$SSH_HOST"
}

cmd_logs() {
    load_state
    log "Tailing run.log (Ctrl-C to stop)..."
    ssh_cmd "tail -f /root/run.log"
}

cmd_status() {
    load_state

    # Rich dashboard via heredoc Python script (avoids nested quoting)
    ssh_cmd python3 - <<'PYEOF'
import json, subprocess, os, textwrap, sys

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except:
        return ""

state_path = "/root/v2/.inner_loop_state.json"
results_path = "/root/v2/results.tsv"

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

# --- Header: GPU + Process ---
gpu = run("nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader")
if gpu:
    parts = [p.strip() for p in gpu.split(",")]
    gpu_name = parts[0] if len(parts) > 0 else "?"
    gpu_mem = f"{parts[1]} / {parts[2]}" if len(parts) > 2 else "?"
    gpu_util = parts[3] if len(parts) > 3 else "?"
    print(f"  GPU:  {gpu_name}  |  Mem: {gpu_mem}  |  Util: {gpu_util}")
else:
    print("  GPU:  unavailable")

pid = run("pgrep -f '^python3 -m v2\\.ops\\.run_experiment_wf'")
if pid:
    pid_line = pid.split("\n")[0]
    uptime = run(f"ps -o etime= -p {pid_line}").strip()
    # Fold progress from checkpoint files
    import glob
    fold_files = sorted(glob.glob("/root/v2/models/model_fold*.pt"))
    n_folds_done = len(fold_files)
    fold_info = f", folds={n_folds_done}/5" if n_folds_done > 0 else ""
    print(f"  Experiment: RUNNING (PID {pid_line}, uptime {uptime}{fold_info})")
    # Last training output from run.log
    log_tail = run("tail -3 /root/run.log 2>/dev/null")
    if log_tail:
        for line in log_tail.strip().split("\n")[-2:]:
            line = line.strip()
            if line and not line.startswith("warning") and "UserWarning" not in line:
                print(f"  Latest: {line[:72]}")
else:
    print(f"  Experiment: NOT RUNNING")

# --- Session State ---
bar("Session State")

if not os.path.exists(state_path):
    print(f"  No state file found: {state_path}")
    print(f"  (No experiments have been run yet)")
else:
    with open(state_path) as f:
        s = json.load(f)
    exp_count = s.get("experiment_count", 0)
    best = s.get("best_score", 0)
    streak = s.get("no_improve_streak", 0)
    stopped = s.get("stopped", False)
    stop_reason = s.get("stop_reason", "")
    kept = s.get("kept_count", 0)
    reverted = exp_count - kept

    print(f"  Experiments: {exp_count}")
    print(f"  Best Score:  {best:.4f}" if isinstance(best, (int, float)) else f"  Best Score:  {best}")
    print(f"  Kept:        {kept}  |  Reverted: {reverted}")
    print(f"  No-improve:  {streak} consecutive")

    if stopped:
        print(f"  Status:      STOPPED ({stop_reason})")
    else:
        print(f"  Status:      ACTIVE")

# --- Experiment History (from results.tsv) ---
bar("Experiment History")

if os.path.exists(results_path):
    # Delegate to v2.ops.status_tsv so the parser stays testable and cannot
    # silently drift out of sync with the CVReport schema.
    try:
        sys.path.insert(0, "/root")
        from v2.ops.status_tsv import render as _render_history
        sys.stdout.write(_render_history(results_path))
    except Exception as _e:
        print(f"  ERROR rendering history: {_e}")
    else:
        print("  (no experiments yet)")
else:
    print("  (no results.tsv yet)")

bar()
PYEOF
}

cmd_download() {
    load_state
    log "=== DOWNLOAD: Syncing v2 results ==="

    # Download model.pt to staging (never overwrite best directly)
    log "Downloading model_candidate.pt..."
    mkdir -p "$PROJECT_ROOT/v2/models" "$PROJECT_ROOT/v2/state"
    scp_cmd "root@$SSH_HOST:/root/v2/models/model.pt" "$PROJECT_ROOT/v2/models/model_candidate.pt" 2>/dev/null || \
        log "  WARNING: v2/models/model.pt not found on remote"

    # Download artifacts/
    log "Downloading artifacts/..."
    mkdir -p "$PROJECT_ROOT/v2/artifacts"
    scp_cmd -r "root@$SSH_HOST:/root/v2/artifacts" "$PROJECT_ROOT/v2/" 2>/dev/null || \
        log "  WARNING: artifacts/ not found on remote"

    # results.tsv: NOT downloaded. Claude appends results locally after each experiment.
    # Downloading would overwrite local entries with stale GPU copy.

    # lab_notebook.md: NOT downloaded. Claude edits it locally after each experiment.
    # Downloading would overwrite local notes with stale GPU copy.

    # Download .best_score
    log "Downloading .best_score..."
    scp_cmd "root@$SSH_HOST:/root/v2/.best_score" "$PROJECT_ROOT/v2/state/best_score.txt" 2>/dev/null || \
        log "  WARNING: .best_score not found on remote"

    # Download .inner_loop_state.json
    log "Downloading .inner_loop_state.json..."
    scp_cmd "root@$SSH_HOST:/root/v2/.inner_loop_state.json" "$PROJECT_ROOT/v2/state/inner_loop_state.json" 2>/dev/null || \
        log "  WARNING: .inner_loop_state.json not found on remote"

    # Validate: best artifact model.pt exists locally
    if [[ -f "$PROJECT_ROOT/v2/state/inner_loop_state.json" ]]; then
        local best_id
        best_id=$(python3 -c "import json; print(json.load(open('$PROJECT_ROOT/v2/state/inner_loop_state.json')).get('best_artifact_id',''))" 2>/dev/null)
        if [[ -n "$best_id" ]]; then
            local best_model="$PROJECT_ROOT/v2/artifacts/$best_id/model.pt"
            if [[ -f "$best_model" ]]; then
                log "VERIFIED: Best artifact $best_id downloaded ($(du -h "$best_model" | cut -f1))"
            else
                log "ERROR: Best artifact $best_id missing model.pt locally! Retrying..."
                scp_cmd "root@$SSH_HOST:/root/v2/artifacts/$best_id/model.pt" "$best_model" 2>/dev/null || \
                    log "CRITICAL: Could not download best model. GPU may be down."
            fi
        fi
    fi

    log "Done! Results in: $PROJECT_ROOT/v2/"
}

# ===================================================================
# _run_sync — core sync loop (polls v2 inner_loop_state)
# ===================================================================
_run_sync() {
    local last_kept=-1
    local last_exp_count=-1
    local poll_interval=30
    local ssh_failures=0
    local max_ssh_failures=20  # ~10 min of failures before giving up

    log "=== AUTO-SYNC ==="
    log "  Remote:   $SSH_HOST:$SSH_PORT"
    log "  Polling every ${poll_interval}s"
    log ""

    while true; do
        # Fetch inner_loop_state.json from remote
        local raw
        raw=$(ssh_cmd "cat /root/v2/.inner_loop_state.json 2>/dev/null") 2>/dev/null || {
            ssh_failures=$((ssh_failures + 1))
            if [[ $ssh_failures -ge $max_ssh_failures ]]; then
                log "ERROR: $max_ssh_failures consecutive SSH failures — sync giving up."
                rm -f "$PROJECT_ROOT/.sync-pid"
                return 1
            fi
            log "WARN: Could not read state (attempt $ssh_failures/$max_ssh_failures) — retrying in ${poll_interval}s..."
            sleep "$poll_interval"
            continue
        }
        ssh_failures=0  # Reset on successful connection

        local experiment_count best_score no_improve_streak stopped stop_reason best_experiment_num
        eval "$(echo "$raw" | python3 -c "
import sys, json
s = json.load(sys.stdin)
print(f'experiment_count={s.get(\"experiment_count\",0)}')
print(f'best_score={s.get(\"best_score\",0)}')
print(f'no_improve_streak={s.get(\"no_improve_streak\",0)}')
print(f'stopped={s.get(\"stopped\",\"false\")}')
print(f'stop_reason=\"{s.get(\"stop_reason\",\"\")}\"')
print(f'best_experiment_num={s.get(\"best_experiment_num\",0)}')
" 2>/dev/null)" || { sleep "$poll_interval"; continue; }

        # Defaults — prevent set -e crash if python3 returned partial output
        experiment_count=${experiment_count:-0}
        best_score=${best_score:-0}
        no_improve_streak=${no_improve_streak:-0}
        stopped=${stopped:-false}
        stop_reason=${stop_reason:-""}
        best_experiment_num=${best_experiment_num:-0}

        # First poll — seed counters and download best if one already exists
        if [[ "$last_kept" -eq -1 ]]; then
            if [[ "$best_experiment_num" -gt 0 ]]; then
                log "* Initial sync: best is exp #$best_experiment_num (score=$best_score) -- downloading..."
                mkdir -p "$PROJECT_ROOT/v2/models"
                scp_cmd "root@$SSH_HOST:/root/v2/models/model.pt" "$PROJECT_ROOT/v2/models/model_candidate.pt" 2>/dev/null || true
                mkdir -p "$PROJECT_ROOT/v2/artifacts"
                scp_cmd -r "root@$SSH_HOST:/root/v2/artifacts" "$PROJECT_ROOT/v2/" 2>/dev/null || true
            fi
            last_kept=$best_experiment_num
            last_exp_count=$experiment_count
            log "  Baseline: exp=$experiment_count best_exp=#$best_experiment_num best=$best_score streak=$no_improve_streak"
            sleep "$poll_interval"
            continue
        fi

        # --- New improvement: download model + artifacts ---
        if [[ "$best_experiment_num" -gt "$last_kept" ]]; then
            log "* IMPROVEMENT at exp #$best_experiment_num (score=$best_score) — syncing model + artifacts..."
            mkdir -p "$PROJECT_ROOT/v2/models"
            scp_cmd "root@$SSH_HOST:/root/v2/models/model.pt" "$PROJECT_ROOT/v2/models/model_candidate.pt" 2>/dev/null || true
            mkdir -p "$PROJECT_ROOT/v2/artifacts"
            scp_cmd -r "root@$SSH_HOST:/root/v2/artifacts" "$PROJECT_ROOT/v2/" 2>/dev/null || true
            last_kept=$best_experiment_num
            last_exp_count=$experiment_count
            log "  Synced. Best score: $best_score"

        # --- Experiment finished but no improvement ---
        elif [[ "$experiment_count" -gt "$last_exp_count" ]]; then
            log "  Exp #$experiment_count done (not kept)."
            last_exp_count=$experiment_count

        # --- Heartbeat: show sync is alive even when nothing changed ---
        else
            log "  Polling... exp=$experiment_count best_exp=#$best_experiment_num best=$best_score streak=$no_improve_streak"
        fi

        # lab_notebook.md: NOT synced. Claude edits it locally.

        # --- Stopped: final full sync and exit ---
        if [[ "$stopped" == "true" || "$stopped" == "True" ]]; then
            log ""
            log "=== EXPERIMENT LOOP STOPPED (reason=$stop_reason, score=$best_score, best_exp=#$best_experiment_num) ==="
            log "Final sync..."
            ( cmd_download ) || log "WARNING: Final download failed"
            rm -f "$PROJECT_ROOT/.sync-pid"
            break
        fi

        sleep "$poll_interval"
    done
}

cmd_sync() {
    load_state

    # Guard against duplicate sync — skip if we ARE the background sync (our PID matches)
    if [[ -f "$PROJECT_ROOT/.sync-pid" ]]; then
        local spid
        spid=$(cat "$PROJECT_ROOT/.sync-pid")
        if [[ "$$" != "$spid" ]] && kill -0 "$spid" 2>/dev/null; then
            log "Auto-sync already running (PID $spid)."
            log "Kill it first with: kill $spid && rm '$PROJECT_ROOT/.sync-pid'"
            exit 0
        elif [[ "$$" != "$spid" ]]; then
            rm -f "$PROJECT_ROOT/.sync-pid"
        fi
    fi

    log "Running sync (PID $$)..."
    _run_sync
}

# ===================================================================
# Internal helper: upload mutable source to the GPU before any run.
# ===================================================================
_upload_mutable_sources() {
    for f in v2/train.py v2/core/policy.py v2/core/walkforward.py \
             v2/core/cv_report.py v2/core/artifact_kind.py \
             v2/ops/run_experiment_wf.py v2/ops/run_final_train.py \
             v2/ops/artifact.py v2/ops/model_manage.py \
             v2/replay.py v2/ops/pre_run_gate.py; do
        [[ -f "$PROJECT_ROOT/$f" ]] || continue
        scp_cmd "$PROJECT_ROOT/$f" "root@$SSH_HOST:/root/$f"
    done

    # Verify train.py integrity
    local local_sha remote_sha
    local_sha=$(shasum -a 256 "$PROJECT_ROOT/v2/train.py" | awk '{print $1}')
    remote_sha=$(ssh_cmd "sha256sum /root/v2/train.py | awk '{print \$1}'" 2>/dev/null)
    [[ "$remote_sha" == "$local_sha" ]] || die "train.py upload integrity check failed"

    # Sync dataset if needed
    [[ -f "$DATA_PT" ]] || die "data.pt not found: $DATA_PT"
    local local_data_sha remote_data_sha
    local_data_sha=$(shasum -a 256 "$DATA_PT" | awk '{print $1}')
    remote_data_sha=$(ssh_cmd "sha256sum /root/v2/data.pt | awk '{print \$1}'" 2>/dev/null || true)
    if [[ "$remote_data_sha" != "$local_data_sha" ]]; then
        log "Syncing dataset $(basename "$DATA_PT") to remote v2/data.pt..."
        scp_retry "$DATA_PT" "root@$SSH_HOST:/root/v2/data.pt"
    fi

    # Verify dependencies
    ssh_cmd "python3 -c 'import torch, numpy, pandas'" \
        || die "Dependencies missing on remote. Re-run: ./deploy.sh start"
}

# ===================================================================
# RUN_CV — upload code, run full 5-fold CV, download CV_EVAL artifact.
# ===================================================================
# Produces a CV_EVAL artifact. NOT a deployable model. Promotion requires
# a separate run_final_train pass followed by model_manage keep.
#
# Usage: ./deploy.sh run_cv exp_017
cmd_run_cv() {
    load_state
    local exp_id="${EXTRA_ARGS:-}"
    [[ -n "$exp_id" ]] || die "Usage: deploy.sh run_cv <exp_id>"

    log "=== CV EXPERIMENT: $exp_id (screening_mode=full) ==="
    run_local_pre_run_gate
    _upload_mutable_sources
    log "Sources uploaded. Running full walk-forward CV..."

    local env_prefix="${TRAIN_ENV:-}"
    ssh_cmd "echo '' > /root/run.log" 2>/dev/null || true
    local run_output
    run_output=$(ssh_cmd "cd /root && $env_prefix PYTHONUNBUFFERED=1 python3 -m v2.ops.run_experiment_wf --id $exp_id --screen-mode full 2>&1 | tee /root/run.log") || true
    echo "$run_output"

    # Download the CV_EVAL artifact directory. Do NOT fetch v2/models/model.pt
    # — walkforward no longer writes there.
    mkdir -p "$PROJECT_ROOT/v2/artifacts"
    scp_cmd -r "root@$SSH_HOST:/root/v2/artifacts/$exp_id" "$PROJECT_ROOT/v2/artifacts/" 2>/dev/null || \
        log "WARNING: CV_EVAL artifact not found on remote (experiment may have crashed)"

    # Append to results.tsv from RESULTS_JSON
    _append_results_tsv "$exp_id" "$run_output"

    log ""
    log "CV complete: $exp_id"
    log "Next: inspect artifact, then:"
    log "  ./deploy.sh run_final_train $exp_id     # if CV passes all gates"
    log "  python3 -m v2.ops.model_manage keep     # after final-train"
}

# ===================================================================
# RUN_SCREEN_LATEST — single-fold parity debug (matches fold 4 of full CV).
# ===================================================================
# No artifact, no results.tsv entry. Exists for exact-parity debugging only.
cmd_run_screen_latest() {
    load_state
    local exp_id="${EXTRA_ARGS:-}"
    [[ -n "$exp_id" ]] || die "Usage: deploy.sh run_screen_latest <exp_id>"

    local screen_id="${exp_id}_screen_latest"
    log "=== SCREEN_LATEST: $screen_id (fold 4 only, no artifact) ==="
    run_local_pre_run_gate
    _upload_mutable_sources
    log "Sources uploaded. Screening latest fold..."

    local env_prefix="${TRAIN_ENV:-}"
    ssh_cmd "echo '' > /root/run.log" 2>/dev/null || true
    local run_output
    run_output=$(ssh_cmd "cd /root && $env_prefix PYTHONUNBUFFERED=1 python3 -m v2.ops.run_experiment_wf --id $screen_id --screen-mode latest --no-artifacts 2>&1 | tee /root/run.log") || true
    echo "$run_output"

    log ""
    log "screen_latest complete. Debug-only — no artifact or results.tsv write."
    log "For hypothesis triage use: ./deploy.sh run_screen_mini $exp_id"
    log "For official cross-validation: ./deploy.sh run_cv $exp_id"
}

# ===================================================================
# RUN_SCREEN_MINI — 3-fold regime triage (folds 0, 2, 4).
# ===================================================================
cmd_run_screen_mini() {
    load_state
    local exp_id="${EXTRA_ARGS:-}"
    [[ -n "$exp_id" ]] || die "Usage: deploy.sh run_screen_mini <exp_id>"

    local screen_id="${exp_id}_screen_mini"
    log "=== SCREEN_MINI: $screen_id (folds 0, 2, 4; no artifact) ==="
    run_local_pre_run_gate
    _upload_mutable_sources
    log "Sources uploaded. Screening mini (early/mid/late regimes)..."

    local env_prefix="${TRAIN_ENV:-}"
    ssh_cmd "echo '' > /root/run.log" 2>/dev/null || true
    local run_output
    run_output=$(ssh_cmd "cd /root && $env_prefix PYTHONUNBUFFERED=1 python3 -m v2.ops.run_experiment_wf --id $screen_id --screen-mode mini --no-artifacts 2>&1 | tee /root/run.log") || true
    echo "$run_output"

    log ""
    log "screen_mini complete. Debug-only — no artifact or results.tsv write."
    log "For official cross-validation: ./deploy.sh run_cv $exp_id"
}

# ===================================================================
# RUN_FINAL_TRAIN — produce a promotable FINAL_TRAIN artifact from CV.
# ===================================================================
# This is the only path that creates a deployable model. Takes the config
# from a CV_EVAL artifact and trains one model on the full pre-shadow span.
#
# Usage: ./deploy.sh run_final_train exp_017       # produces exp_017_final
#        ./deploy.sh run_final_train exp_017 exp_017_deploy
cmd_run_final_train() {
    load_state
    local src_id final_id
    # Parse two-positional extra args
    set -- $EXTRA_ARGS
    src_id="${1:-}"
    final_id="${2:-}"
    [[ -n "$src_id" ]] || die "Usage: deploy.sh run_final_train <source_cv_exp_id> [final_exp_id]"

    [[ -n "$final_id" ]] || final_id="${src_id}_final"

    log "=== FINAL TRAIN: $final_id (from CV $src_id) ==="
    run_local_pre_run_gate
    _upload_mutable_sources

    # The source CV_EVAL artifact must exist on the remote
    ssh_cmd "test -f /root/v2/artifacts/${src_id}/cv_report.json" \
        || die "CV_EVAL artifact not found on remote: /root/v2/artifacts/${src_id}/cv_report.json"

    log "Sources uploaded. Training final deployable model..."
    # Intentional: TRAIN_ENV is NOT forwarded to run_final_train. The CV_EVAL
    # artifact carries the exact env overrides used in the CV, and
    # run_final_train replays them via os.environ. Forwarding an operator-set
    # TRAIN_ENV here would let the deployed model diverge from the CV that
    # selected it. If a knob needs to change, re-run CV under the new env.
    if [[ -n "${TRAIN_ENV:-}" ]]; then
        log "NOTE: TRAIN_ENV=$TRAIN_ENV is IGNORED for run_final_train. The source"
        log "      CV artifact's env_overrides are authoritative."
    fi
    ssh_cmd "echo '' > /root/run.log" 2>/dev/null || true
    local run_output
    run_output=$(ssh_cmd "cd /root && PYTHONUNBUFFERED=1 python3 -m v2.ops.run_final_train --config-from $src_id --id $final_id 2>&1 | tee /root/run.log") || true
    echo "$run_output"

    # Pull the FINAL_TRAIN artifact directory and stage the candidate locally
    mkdir -p "$PROJECT_ROOT/v2/artifacts"
    scp_cmd -r "root@$SSH_HOST:/root/v2/artifacts/$final_id" "$PROJECT_ROOT/v2/artifacts/" \
        || die "Failed to download FINAL_TRAIN artifact $final_id"
    mkdir -p "$PROJECT_ROOT/v2/models"
    cp "$PROJECT_ROOT/v2/artifacts/$final_id/model.pt" "$PROJECT_ROOT/v2/models/model_candidate.pt"

    log ""
    log "FINAL_TRAIN artifact downloaded: v2/artifacts/$final_id"
    log "Candidate staged at: v2/models/model_candidate.pt"
    log "Next: python3 -m v2.ops.model_manage keep"
    log "  (or: python3 -m v2.ops.model_manage revert)"
}

cmd_stop() {
    load_state
    echo ""
    echo "This will: kill experiment + download results + close deployment"

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

    log "Killing experiment process..."
    ssh_cmd "pkill -f '^python3 -m v2\\.ops\\.run_experiment_wf' 2>/dev/null || true" || true

    # Wait for process to actually die (up to 30s) — prevents partial file downloads
    for _ in $(seq 1 30); do
        if ! ssh_cmd "pgrep -f '^python3 -m v2\\.ops\\.run_experiment_wf'" &>/dev/null; then
            break
        fi
        sleep 1
    done
    # Force kill if still alive
    ssh_cmd "pkill -9 -f '^python3 -m v2\\.ops\\.run_experiment_wf' 2>/dev/null || true" 2>/dev/null || true
    sleep 3  # let filesystem flush before downloading

    log "Downloading results before closing..."
    # Run in subshell so die()/exit inside cmd_download doesn't kill the stop flow
    ( cmd_download ) || log "WARNING: Download failed (container may be dead). Proceeding to close deployment."

    log "Closing Akash deployment DSEQ=$DSEQ..."
    local close_out
    close_out=$(provider-services tx deployment close \
        --dseq "$DSEQ" --from "$AKASH_FROM" \
        --gas auto --gas-adjustment 1.5 --gas-prices 0.025uakt \
        --yes --output json 2>&1) \
        || log "WARNING: Close TX failed (deployment may already be closed)"
    if echo "$close_out" | jq -e '.txhash' &>/dev/null; then
        local txhash
        txhash=$(echo "$close_out" | jq -r '.txhash')
        log "Close TX: $txhash"
    fi
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
    boot)              cmd_boot              ;;
    start)             cmd_start             ;;
    run_cv)            cmd_run_cv            ;;
    run_screen_latest) cmd_run_screen_latest ;;
    run_screen_mini)   cmd_run_screen_mini   ;;
    run_final_train)   cmd_run_final_train   ;;
    fund)              cmd_fund              ;;
    ssh)               cmd_ssh               ;;
    logs)              cmd_logs              ;;
    status)            cmd_status            ;;
    download)          cmd_download          ;;
    stop)              cmd_stop              ;;
    # --- Removed footguns (harness-integrity repair 2026-04-17) ---
    run_one|run_screen)
        die "Command '$CMD' was removed. Use: run_cv (full 5-fold), run_screen_latest (fold 4 only), run_screen_mini (folds 0,2,4), or run_final_train (deploy). See: ./deploy.sh with no arguments."
        ;;
    *)
        echo "Usage: ./deploy.sh <command> [options]"
        echo ""
        echo "CV pipeline (Claude drives):"
        echo "  boot                     Deploy H100 container on Akash (~2 min)"
        echo "  start                    Upload v2 code + data, install deps"
        echo "  run_screen_latest ID     Single-fold parity debug (fold 4 only, no artifact)"
        echo "  run_screen_mini   ID     3-fold regime triage (folds 0, 2, 4, no artifact)"
        echo "  run_cv            ID     Full 5-fold CV, emits CV_EVAL artifact"
        echo "  run_final_train  SRC_ID  Train deployable model from chosen CV"
        echo ""
        echo "Promotion:"
        echo "  python3 -m v2.ops.model_manage keep   (FINAL_TRAIN only; rejects CV_EVAL)"
        echo "  python3 -m v2.ops.model_manage revert"
        echo ""
        echo "Utilities:"
        echo "  ssh        SSH into the H100"
        echo "  logs       Tail run.log"
        echo "  status     GPU + experiment dashboard"
        echo "  download   Download all v2 results + artifacts"
        echo "  fund       Add ACT to deployment escrow"
        echo "  stop       Kill experiment + close Akash deployment"
        exit 1
        ;;
esac
