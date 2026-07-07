# Protocol101 Foundation Canaries

- Status: `pass`
- Train: 82 sessions (2024_10, 2024_11, 2024_12, 2025_01), 236133 examples
- Eval: 40 sessions (2025_02, 2025_03), 115152 examples

## Canary Verdicts

- `shuffled_label_null_silent`: **PASS**
- `planted_edge_detected`: **PASS**
- `tampered_pickle_blocked`: **PASS**
- `lag_zero_build_rejected`: **PASS**

## Key Numbers

- True-data run (informational): win-rho=0.1101, selection z=-5.24
- Shuffled-label run: win-rho=-0.0030 (measured null band [-0.0073, 0.0074]), selection z=-1.03
- Planted-edge run: win-rho=0.9980, selection z=98.44
- Random-selection null band (top-decile mean PnL): p05=-31.24, p95=-14.74

## Guardrails

- Diagnostics tier only; no model persisted; no thresholds tuned; nothing promoted.
- The true-data run is informational and is NOT a strategy-performance claim.
