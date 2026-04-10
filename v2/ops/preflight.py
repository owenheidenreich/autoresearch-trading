#!/usr/bin/env python3
"""Pre-flight check for remote GPU node. Run after deploy.sh uploads workspace."""
import torch, os, sys

assert torch.cuda.is_available(), "No CUDA GPU detected"
gpu = torch.cuda.get_device_name(0)

data = "/root/v2/data.pt"
assert os.path.exists(data), f"data.pt missing at {data}"
d = torch.load(data, map_location="cpu", weights_only=False)
meta = d.get('metadata', {})
sidecar_dir = meta.get("chain_sidecar_dir")

train_py = "/root/v2/train.py"
assert os.path.exists(train_py), f"train.py missing at {train_py}"
if sidecar_dir:
    remote_sidecar_dir = os.path.join("/root", sidecar_dir)
    assert os.path.isdir(remote_sidecar_dir), f"chain sidecar dir missing at {remote_sidecar_dir}"

print(f"GPU: {gpu}")
print(f"data.pt: {len(d)} keys, X shape {d['X'].shape}")
print(f"Labels: {meta.get('label_version', 'unknown')}")
print(f"Split: {meta.get('split', {}).get('train_days', '?')}/{meta.get('split', {}).get('val_days', '?')}/{meta.get('split', {}).get('promote_days', '?')}/{meta.get('split', {}).get('shadow_days', '?')} days")
if sidecar_dir:
    print(f"chain sidecars: {sidecar_dir}")
print(f"train.py: OK")
print("Pre-flight OK")
