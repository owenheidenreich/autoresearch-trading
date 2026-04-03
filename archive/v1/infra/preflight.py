#!/usr/bin/env python3
"""Pre-flight check for remote GPU node. Run after deploy.sh uploads workspace."""
import torch, os, sys

assert torch.cuda.is_available(), "No CUDA GPU detected"
gpu = torch.cuda.get_device_name(0)

data = "/root/.cache/autoresearch-trading/features/data.pt"
assert os.path.exists(data), f"data.pt missing at {data}"
d = torch.load(data, map_location="cpu", weights_only=True)

train_py = "/root/autoresearch-trading/training/train.py"

print(f"GPU: {gpu}")
print(f"data.pt: {len(d)} keys, features shape {d['features'].shape}")
print(f"train.py: {os.path.exists(train_py)}")
print("Pre-flight OK")
