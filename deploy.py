#!/usr/bin/env python3
"""
Akash Deployment Script for Autoresearch-Trading

Deploys an H100/A100 GPU training container on Akash Network.
Uses deploy-autoresearch.yaml SDL and handles the full lifecycle:
  create deployment → wait for bids → select provider → lease → manifest → wait → SSH info

Usage:
  python3 deploy.py                    # Interactive — pick a provider from bids
  python3 deploy.py --deposit 5.0      # Add extra AKT to escrow (default 0.5)
  python3 deploy.py --close-all        # Close all active deployments and exit
  python3 deploy.py --status           # Show status of active deployments

Requires:
  - provider-services CLI (Akash CLI)
  - Wallet 'trinity-wallet' in keyring
"""

import subprocess
import json
import sys
import time
import tempfile
import os
import re
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

AKASH_RPC_NODES = [
    "https://akash-rpc.polkachu.com:443",
    "https://akash-rpc.publicnode.com:443",
    "https://rpc-akash.ecostake.com:443",
    "https://rpc.akashnet.net:443",
]
AKASH_NODE = AKASH_RPC_NODES[0]
AKASH_CHAIN_ID = "akashnet-2"
WALLET_NAME = "trinity-wallet"

# SDL file path (same directory as this script)
SCRIPT_DIR = Path(__file__).parent.absolute()
SDL_PATH = SCRIPT_DIR / "deploy-autoresearch.yaml"

# Providers to avoid — known issues
BLOCKED_PROVIDERS = [
    "akash19yhu3jgw8h0320av98h8n5qczje3pj3u9u2amp",  # bdl.computer - times out
    "akash1sjwuwre4qprcaa34f6324yz7m8nn0awvc75gp5",  # quanglong.org - slow pulls
    "akash1adyrcsp2ptwd83txgv555eqc0vhfufc37wx040",  # airitdecomp.net - DNS fails
    "akash1kqzpqqhm39umt06wu8m4hx63v5hefhrfmjf9dj",  # leet.haus - containers fail
    "akash1ggfvyhr9sar4uxjs4hth3p4kzrwk7lysnenj3g",  # ghost provider
]

BLOCKED_PROVIDER_URIS = [
    "subangle.com",
    "leet.haus",
]


# =============================================================================
# INFRASTRUCTURE
# =============================================================================

def find_working_rpc():
    """Test RPC nodes with a real market query."""
    global AKASH_NODE
    for node in AKASH_RPC_NODES:
        try:
            result = subprocess.run(
                f"provider-services query market order list --node {node} --limit 1 -o json",
                shell=True, capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                AKASH_NODE = node
                print(f"  RPC: {node}")
                return node
            else:
                print(f"  {node} — query failed")
        except subprocess.TimeoutExpired:
            print(f"  {node} — timeout")
    print("All RPC nodes unreachable!")
    sys.exit(1)


def run_cmd(cmd, capture=True, timeout=120):
    """Run a shell command and return (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture, text=True, timeout=timeout
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "timeout", 1


def get_wallet_address():
    stdout, _, code = run_cmd(
        f"provider-services keys show {WALLET_NAME} --keyring-backend os -a 2>/dev/null"
    )
    return stdout.strip() if code == 0 else None


def get_wallet_balance(wallet_addr):
    stdout, _, code = run_cmd(
        f"provider-services query bank balances {wallet_addr} "
        f"--node {AKASH_NODE} -o json 2>/dev/null"
    )
    if code != 0:
        return None
    try:
        data = json.loads(stdout)
        for bal in data.get("balances", []):
            if bal.get("denom") == "uakt":
                return int(bal["amount"]) / 1_000_000
    except Exception:
        pass
    return None


def price_to_monthly(uakt_per_block):
    akt_per_block = float(uakt_per_block) / 1_000_000
    daily = akt_per_block * 14400  # ~6s blocks
    return daily * 30


def price_to_hourly(uakt_per_block):
    akt_per_block = float(uakt_per_block) / 1_000_000
    return akt_per_block * 600  # ~600 blocks/hour


# =============================================================================
# DEPLOYMENT OPERATIONS
# =============================================================================

def close_deployment(dseq):
    run_cmd(
        f"provider-services tx deployment close --dseq {dseq} "
        f"--from {WALLET_NAME} --keyring-backend os "
        f"--node {AKASH_NODE} --chain-id {AKASH_CHAIN_ID} "
        f"--gas-prices 0.025uakt --gas auto --gas-adjustment 1.5 "
        f"--yes 2>/dev/null"
    )


def close_all_deployments(wallet_addr):
    """Close ALL active deployments."""
    print("Closing active deployments...")
    stdout, _, code = run_cmd(
        f"provider-services query deployment list "
        f"--owner {wallet_addr} --state active "
        f"--node {AKASH_NODE} -o json 2>/dev/null"
    )
    if code != 0:
        print("  No active deployments found")
        return 0

    try:
        data = json.loads(stdout)
        deployments = data.get("deployments", [])
        closed = 0
        for dep in deployments:
            dseq = (dep.get("deployment", {}).get("deployment_id", {}).get("dseq", "")
                    or dep.get("deployment", {}).get("id", {}).get("dseq", ""))
            if not dseq:
                continue
            print(f"  Closing DSEQ {dseq}...")
            close_deployment(dseq)
            closed += 1
            time.sleep(2)
        print(f"  Closed {closed} deployment(s)" if closed else "  No deployments to close")
        return closed
    except Exception as e:
        print(f"  Warning: {e}")
        return 0


def show_active_deployments(wallet_addr):
    """Show status of all active deployments."""
    stdout, _, code = run_cmd(
        f"provider-services query deployment list "
        f"--owner {wallet_addr} --state active "
        f"--node {AKASH_NODE} -o json 2>/dev/null"
    )
    if code != 0:
        print("No active deployments.")
        return

    try:
        data = json.loads(stdout)
        deployments = data.get("deployments", [])
        if not deployments:
            print("No active deployments.")
            return

        print(f"\nActive deployments: {len(deployments)}")
        print("-" * 60)
        for dep in deployments:
            dseq = (dep.get("deployment", {}).get("deployment_id", {}).get("dseq", "")
                    or dep.get("deployment", {}).get("id", {}).get("dseq", ""))
            state = dep.get("deployment", {}).get("state", "unknown")
            print(f"  DSEQ {dseq}  state={state}")

            # Try to get lease info
            lease_stdout, _, lcode = run_cmd(
                f"provider-services query market lease list "
                f"--owner {wallet_addr} --dseq {dseq} "
                f"--node {AKASH_NODE} -o json 2>/dev/null"
            )
            if lcode == 0:
                try:
                    lease_data = json.loads(lease_stdout)
                    for lease in lease_data.get("leases", []):
                        provider = lease.get("lease", {}).get("id", {}).get("provider", "")
                        price = lease.get("lease", {}).get("price", {}).get("amount", "0")
                        monthly = price_to_monthly(float(price))
                        host, gpu = get_provider_host(provider)
                        print(f"    Provider: {host} ({gpu})")
                        print(f"    Cost: ${monthly:.2f}/mo")

                        # Get lease status for URI
                        status = get_lease_status(dseq, provider)
                        if status:
                            for svc_name, svc_info in status.get("services", {}).items():
                                uris = svc_info.get("uris", [])
                                avail = svc_info.get("available_replicas", 0)
                                forwarded = svc_info.get("forwarded_ports", {})
                                if uris:
                                    print(f"    URI: {uris[0]}  (replicas={avail})")
                                if forwarded:
                                    for svc, ports in forwarded.items():
                                        for p in ports:
                                            ext = p.get("externalPort", "?")
                                            host_uri = p.get("host", uris[0] if uris else "unknown")
                                            proto = p.get("proto", "TCP")
                                            print(f"    SSH: ssh root@{host_uri.split(':')[0] if ':' in str(host_uri) else host} -p {ext}")
                except Exception:
                    pass
        print("-" * 60)
    except Exception as e:
        print(f"Error: {e}")


def create_deployment(sdl_content, deposit_uakt=500000):
    """Create a deployment from SDL content, return DSEQ."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(sdl_content)
        temp_yaml = f.name

    try:
        stdout, stderr, code = run_cmd(
            f"provider-services tx deployment create {temp_yaml} "
            f"--deposit {deposit_uakt}uakt "
            f"--from {WALLET_NAME} --keyring-backend os "
            f"--node {AKASH_NODE} --chain-id {AKASH_CHAIN_ID} "
            f"--gas auto --gas-adjustment 1.5 --gas-prices 0.025uakt "
            f"--yes -o json"
        )
        if code != 0:
            print(f"  Deployment creation failed")
            if stderr:
                print(f"  {stderr[:500]}")
            return None

        match = re.search(r'dseq[^\d]*(\d+)', stdout)
        if match:
            return int(match.group(1))
        print(f"  Could not extract DSEQ from output")
        return None
    finally:
        os.unlink(temp_yaml)


def send_manifest(sdl_content, dseq, provider):
    """Send manifest to provider."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(sdl_content)
        temp_yaml = f.name

    try:
        stdout, _, code = run_cmd(
            f"provider-services send-manifest {temp_yaml} "
            f"--dseq {dseq} --provider {provider} "
            f"--from {WALLET_NAME} --keyring-backend os "
            f"--node {AKASH_NODE} 2>&1",
            timeout=60
        )
        return code == 0 and "FAIL" not in stdout
    finally:
        os.unlink(temp_yaml)


def create_lease(dseq, provider):
    _, _, code = run_cmd(
        f"provider-services tx market lease create "
        f"--dseq {dseq} --gseq 1 --oseq 1 --provider {provider} "
        f"--from {WALLET_NAME} --keyring-backend os "
        f"--node {AKASH_NODE} --chain-id {AKASH_CHAIN_ID} "
        f"--gas auto --gas-adjustment 1.5 --gas-prices 0.025uakt "
        f"--yes 2>&1"
    )
    return code == 0


def get_lease_status(dseq, provider):
    stdout, _, code = run_cmd(
        f"provider-services lease-status "
        f"--dseq {dseq} --provider {provider} "
        f"--from {WALLET_NAME} --keyring-backend os "
        f"--node {AKASH_NODE} 2>/dev/null"
    )
    if code == 0:
        try:
            return json.loads(stdout)
        except Exception:
            pass
    return None


def get_bids(wallet_addr, dseq):
    """Get all bids for a deployment, sorted by price."""
    stdout, stderr, code = run_cmd(
        f"provider-services query market bid list "
        f"--owner {wallet_addr} --dseq {dseq} "
        f"--node {AKASH_NODE} -o json"
    )
    if code != 0:
        return []

    try:
        data = json.loads(stdout)
        bids = data.get("bids", [])

        results = []
        for bid in bids:
            provider = bid.get("bid", {}).get("id", {}).get("provider", "")
            price_str = bid.get("bid", {}).get("price", {}).get("amount", "999999999")
            if not provider:
                continue
            try:
                price = float(price_str)
            except (ValueError, TypeError):
                price = 999999999.0
            monthly = price_to_monthly(price)
            hourly = price_to_hourly(price)
            results.append((provider, price, monthly, hourly))

        results.sort(key=lambda x: x[1])
        return results
    except Exception as e:
        print(f"  Error parsing bids: {e}")
        return []


def get_provider_host(provider_addr):
    """Look up provider host URI and GPU model."""
    stdout, _, code = run_cmd(
        f"provider-services query provider get {provider_addr} "
        f"--node {AKASH_NODE} -o json",
        timeout=15
    )
    if code != 0:
        return "unknown", ""
    try:
        data = json.loads(stdout)
        host = data.get("host_uri", "") or data.get("provider", {}).get("host_uri", "")
        host = host.replace("https://", "").replace("http://", "").rstrip("/")

        gpu_info = ""
        attributes = data.get("attributes", []) or data.get("provider", {}).get("attributes", [])
        for attr in attributes:
            key = attr.get("key", "")
            if key.startswith("capabilities/gpu/vendor/") and "/model/" in key:
                parts = key.split("/model/")
                if len(parts) >= 2:
                    model_name = parts[1].split("/")[0]
                    if model_name and model_name != "other":
                        gpu_info = model_name.upper()
                        break
            if key == "hardware-gpu" and attr.get("value", ""):
                gpu_info = attr["value"].upper()
                break

        return host or "unknown", gpu_info
    except Exception:
        return "unknown", ""


# =============================================================================
# CACHE MANAGEMENT — download/upload .pkl files to avoid re-downloading data
# =============================================================================

LOCAL_CACHE_DIR = Path.home() / ".cache" / "autoresearch-trading" / "data"
REMOTE_CACHE_DIR = "/root/.cache/autoresearch-trading/data"

CACHE_FILES = [
    "equity_raw_v4.pkl",         # Equity data only (saved before options download)
    "historical_options_5m.pkl", # Options bars + Greeks
    "intraday_raw_v4.pkl",      # Complete dataset (equity + options)
]


def _scp_base_args(host, port, password="autoresearch2026"):
    """Return base sshpass + scp args for Akash container."""
    return [
        "sshpass", "-p", password,
        "scp", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-P", str(port),
    ]


def download_cache(host, port):
    """SCP cache files from Akash container to local machine."""
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading cache files from {host}:{port}...")
    for fname in CACHE_FILES:
        remote = f"root@{host}:{REMOTE_CACHE_DIR}/{fname}"
        local = LOCAL_CACHE_DIR / fname
        cmd = _scp_base_args(host, port) + [remote, str(local)]
        print(f"  {fname}...", end=" ", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            size_mb = local.stat().st_size / (1024 * 1024)
            print(f"OK ({size_mb:.1f} MB)")
        else:
            print(f"SKIP (not found or error)")

    print(f"  Local cache: {LOCAL_CACHE_DIR}")


def upload_cache(host, port):
    """SCP cache files from local machine to Akash container."""
    print(f"\nUploading cache files to {host}:{port}...")

    # Ensure remote dir exists
    ssh_cmd = [
        "sshpass", "-p", "autoresearch2026",
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-p", str(port), f"root@{host}",
        f"mkdir -p {REMOTE_CACHE_DIR}"
    ]
    subprocess.run(ssh_cmd, capture_output=True, timeout=30)

    uploaded = 0
    for fname in CACHE_FILES:
        local = LOCAL_CACHE_DIR / fname
        if not local.exists():
            continue
        remote = f"root@{host}:{REMOTE_CACHE_DIR}/{fname}"
        cmd = _scp_base_args(host, port) + [str(local), remote]
        size_mb = local.stat().st_size / (1024 * 1024)
        print(f"  {fname} ({size_mb:.1f} MB)...", end=" ", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print("OK")
            uploaded += 1
        else:
            print(f"FAILED: {result.stderr.strip()}")

    if uploaded:
        print(f"  {uploaded} file(s) uploaded — prepare.py will skip downloads")
    else:
        print("  No cache files found locally. Run --download-cache first.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deploy autoresearch-trading to Akash")
    parser.add_argument("--deposit", type=float, default=0.5,
                        help="AKT escrow deposit (default: 0.5)")
    parser.add_argument("--close-all", action="store_true",
                        help="Close all active deployments and exit")
    parser.add_argument("--status", action="store_true",
                        help="Show active deployment status and exit")
    parser.add_argument("--no-close", action="store_true",
                        help="Don't close existing deployments before creating new one")
    parser.add_argument("--download-cache", nargs=2, metavar=("HOST", "PORT"),
                        help="SCP cache files from container: --download-cache HOST PORT")
    parser.add_argument("--upload-cache", nargs=2, metavar=("HOST", "PORT"),
                        help="SCP cache files to container: --upload-cache HOST PORT")
    args = parser.parse_args()

    # --- Cache management (no wallet needed) ---
    if args.download_cache:
        download_cache(args.download_cache[0], int(args.download_cache[1]))
        return
    if args.upload_cache:
        upload_cache(args.upload_cache[0], int(args.upload_cache[1]))
        return

    # --- Find RPC ---
    print("Testing RPC nodes...")
    find_working_rpc()

    # --- Wallet ---
    wallet_addr = get_wallet_address()
    if not wallet_addr:
        print("Could not get wallet address. Is provider-services installed?")
        sys.exit(1)
    print(f"  Wallet: {wallet_addr}")

    balance = get_wallet_balance(wallet_addr)
    if balance is not None:
        print(f"  Balance: {balance:.2f} AKT")

    # --- Status mode ---
    if args.status:
        show_active_deployments(wallet_addr)
        return

    # --- Close-all mode ---
    if args.close_all:
        close_all_deployments(wallet_addr)
        return

    # --- Check SDL exists ---
    if not SDL_PATH.exists():
        print(f"SDL file not found: {SDL_PATH}")
        sys.exit(1)

    sdl_content = SDL_PATH.read_text()

    # --- Close existing deployments (unless --no-close) ---
    if not args.no_close:
        close_all_deployments(wallet_addr)

    # --- Create deployment ---
    deposit_uakt = int(args.deposit * 1_000_000)
    print(f"\nCreating deployment (deposit: {args.deposit:.1f} AKT)...")
    print(f"  SDL: {SDL_PATH.name}")
    print(f"  GPU: H100 80GB (from SDL)")

    dseq = create_deployment(sdl_content, deposit_uakt)
    if not dseq:
        print("Failed to create deployment")
        sys.exit(1)
    print(f"  DSEQ: {dseq}")

    # --- Wait for bids ---
    print(f"\nWaiting for provider bids (30s)...")
    time.sleep(30)

    bids = get_bids(wallet_addr, dseq)
    if not bids:
        print("\nNo providers bid! H100s may be fully reserved.")
        print("Try again later or modify deploy-autoresearch.yaml to accept A100s.")
        close_deployment(dseq)
        sys.exit(1)

    # --- Look up provider details ---
    print(f"\n{len(bids)} provider(s) bid:")
    provider_hosts = {}
    provider_gpus = {}
    for provider, price, monthly, hourly in bids:
        host, gpu = get_provider_host(provider)
        provider_hosts[provider] = host
        provider_gpus[provider] = gpu or "GPU"

    # --- Display bids ---
    print()
    print("-" * 75)
    for i, (provider, price, monthly, hourly) in enumerate(bids, 1):
        host = provider_hosts.get(provider, "unknown")
        gpu = provider_gpus.get(provider, "GPU")
        flags = ""
        if provider in BLOCKED_PROVIDERS:
            flags = " [BLOCKED]"
        elif any(b in host for b in BLOCKED_PROVIDER_URIS):
            flags = " [WARN]"
        print(f"  {i}) ${hourly:.2f}/hr (${monthly:.0f}/mo)  {gpu:<10} {host}{flags}")
    print("-" * 75)

    # --- Select provider ---
    if len(bids) == 1:
        print(f"\nOnly 1 provider — selecting automatically.")
        idx = 0
    else:
        try:
            choice = input(f"\nSelect provider [1-{len(bids)}]: ").strip()
            idx = int(choice) - 1
            if not (0 <= idx < len(bids)):
                print("Invalid selection. Closing deployment.")
                close_deployment(dseq)
                sys.exit(1)
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\nCancelled. Closing deployment.")
            close_deployment(dseq)
            sys.exit(1)

    selected_provider, _, selected_monthly, selected_hourly = bids[idx]
    selected_host = provider_hosts[selected_provider]
    selected_gpu = provider_gpus[selected_provider]

    if selected_provider in BLOCKED_PROVIDERS:
        try:
            confirm = input("  This provider has known issues. Continue? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            confirm = "n"
        if confirm != "y":
            close_deployment(dseq)
            sys.exit(1)

    print(f"\n  Selected: {selected_host} ({selected_gpu}, ${selected_hourly:.2f}/hr)")

    # --- Create lease ---
    print("\nCreating lease...", end=" ", flush=True)
    if not create_lease(dseq, selected_provider):
        print("FAILED")
        close_deployment(dseq)
        sys.exit(1)
    print("OK")
    time.sleep(3)

    # --- Send manifest ---
    print("Sending manifest...", end=" ", flush=True)
    if not send_manifest(sdl_content, dseq, selected_provider):
        print("FAILED")
        close_deployment(dseq)
        sys.exit(1)
    print("OK")

    # --- Wait for container ---
    max_wait = 600  # 10 min — pulling pytorch image can be slow
    print(f"\nWaiting for container to start (up to {max_wait // 60} min)...")
    container_ready = False
    ssh_info = None
    start_time = time.time()

    for _ in range(max_wait // 10):
        time.sleep(10)
        elapsed = int(time.time() - start_time)

        if elapsed % 30 == 0:
            print(f"  [{elapsed // 60}m {elapsed % 60:02d}s] Waiting...")

        status = get_lease_status(dseq, selected_provider)
        if not status:
            continue

        services = status.get("services", {})
        forwarded_ports = status.get("forwarded_ports", {})

        for svc_name, svc_info in services.items():
            available = svc_info.get("available_replicas", 0)
            uris = svc_info.get("uris", [])

            if available >= 1:
                container_ready = True

                # Look for SSH port in forwarded_ports
                fwd = forwarded_ports.get(svc_name, []) if forwarded_ports else []
                if not fwd:
                    # Try alternative structure
                    fwd = svc_info.get("forwarded_ports", [])

                for p in fwd:
                    ext_port = p.get("externalPort") or p.get("external_port")
                    port = p.get("port", 0)
                    if port == 22 and ext_port:
                        host_for_ssh = p.get("host", "")
                        if not host_for_ssh and uris:
                            host_for_ssh = uris[0]
                        ssh_info = (host_for_ssh, ext_port)
                break

        if container_ready:
            break

    if not container_ready:
        print(f"\n  Container didn't start within {max_wait // 60} minutes.")
        print("  The deployment is still active — check with: python3 deploy.py --status")
        print(f"  To close: python3 deploy.py --close-all")
        sys.exit(1)

    # --- Output results ---
    print()
    print("=" * 60)
    print("  DEPLOYMENT SUCCESSFUL")
    print("=" * 60)
    print(f"  DSEQ:     {dseq}")
    print(f"  Provider: {selected_host} ({selected_gpu})")
    print(f"  Cost:     ${selected_hourly:.2f}/hr (${selected_monthly:.0f}/mo)")

    if ssh_info:
        ssh_host, ssh_port = ssh_info
        print(f"  SSH:      ssh root@{ssh_host} -p {ssh_port}")
        print(f"  Password: autoresearch2026")
    else:
        print(f"  SSH port not yet available — run: python3 deploy.py --status")

    print()
    print("  Next steps:")
    if LOCAL_CACHE_DIR.exists() and any((LOCAL_CACHE_DIR / f).exists() for f in CACHE_FILES):
        if ssh_info:
            print(f"    1. python3 deploy.py --upload-cache {ssh_host} {ssh_port}")
            print("    2. SSH in → uv run prepare.py   (skips download, just computes features)")
        else:
            print("    1. Upload cache once SSH is available:")
            print("       python3 deploy.py --upload-cache HOST PORT")
    else:
        print("    1. SSH into the container")
        print("    2. cd /workspace/autoresearch-trading")
        print(f"    3. export POLYGON_API_KEY=<your-key>")
        print("    4. uv run prepare.py")
    print("    5. uv run python train.py > run.log 2>&1")
    print("=" * 60)


if __name__ == "__main__":
    main()
