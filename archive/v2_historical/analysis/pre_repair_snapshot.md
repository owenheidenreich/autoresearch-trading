# Pre-Repair Harness Snapshot

Date: 2026-04-08
Branch: autoresearch/v2-gate-fix
Purpose: Preserve fingerprints before harness integrity repair (audit action plan Steps 0-13).

## File Fingerprints

| File | MD5 | Size | Modified |
|------|-----|------|----------|
| `v2/data.pt` | `1ccabd88d5e72014588f8ba155f9be79` | 125,475,205 bytes | 2026-04-08 22:11 |
| `v2/model_best.pt` | `49afa10b67065e5a192ad40a3e58c4b4` | 710,182 bytes | 2026-04-08 19:25 |
| `v2/model.pt` | `ed61a25995cef355f71b389811d76e03` | 705,431 bytes | 2026-04-08 22:16 |
| `v2/results.tsv` | `e8d5a244a2fe1a8cd89670946f0a3458` | 5,354 bytes | 2026-04-08 21:57 |
| `v2/.baseline_cache.json` | `1ac92ee2d50b8cbfc723fbb6a891be94` | 11,515 bytes | 2026-04-08 23:07 |

## Staging Convention

All rebuild / relabel work targets `v2/data_harness_repair.pt` until Step 11 validation passes.
Only then does it promote to `v2/data.pt`.

## Current Best Artifact

Best artifact by score: `exp_066` (score=5.5228). No promotion status field exists in manifests (audit finding confirmed).
Current `model.pt` MD5: `ed61a25995cef355f71b389811d76e03`
Current `model_best.pt` MD5: `49afa10b67065e5a192ad40a3e58c4b4`

## Rollback

To revert any harness repair, restore from git history on this branch at commit before repair work begins.
