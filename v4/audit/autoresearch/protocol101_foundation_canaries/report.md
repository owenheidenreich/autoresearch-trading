# Protocol101 Foundation Canaries

- Status: `pass`
- Train: 82 sessions (2024_10, 2024_11, 2024_12, 2025_01), 550977 examples
- Eval: 40 sessions (2025_02, 2025_03), 268688 examples

## Canary Verdicts

- `true_exceeds_refit_null`: **PASS**
- `planted_edge_detected`: **PASS**
- `tampered_pickle_blocked`: **PASS**
- `lag_zero_build_rejected`: **PASS**

## Key Numbers

- True-data run (informational): win-rho=0.1962, selection z=15.72
- Refit-null envelope (5 shuffled retrainings): |win-rho| max=0.0454, selection z max=-6.62
- Planted-edge run: win-rho=0.9985, selection z=144.14
- Random-selection null band (top-decile mean PnL): p05=-75.88, p95=-58.53

## Guardrails

- Diagnostics tier only; no model persisted; no thresholds tuned; nothing promoted.
- The true-data run is informational and is NOT a strategy-performance claim.
