"""H3a + H3e look-ahead audit: prove features depend only on past+current data.

Discipline anchor: the 2026-04-25 oracle-gate label-leakage retraction. Every
new feature in trade_state must be causal — its value at bar t must be a
function of bars[0..t], NOT bars[t+1..].

H3a audit (trajectory features):
  Pure functions f(pnls[0..t]) -> scalar. Mutate pnls[t+1:], assert f(t)
  unchanged. 8401 bars audited, 0 leaks (committed previously).

H3e audit (per-bar contract Greeks):
  Greeks at bar t come from sidecar row at bar t for the chosen contract.
  Test: build paths, mutate sidecar rows at bars > t, rebuild paths,
  assert path.<greek>[t] unchanged for all t.

Both audits use the SAME implementations the L3 oracle uses, so a PASS
verdict here applies bit-exactly to production training rows.
"""
from __future__ import annotations

import pickle

import numpy as np

from v3.layer3.h3a_features import FEATURES
from v3.oracles.opportunity import _build_contract_paths


# ---------------------------------------------------------------------------
# Audit driver.
# ---------------------------------------------------------------------------


def audit_feature(
    name: str,
    fn,
    trajectories: list[dict],
    rng: np.random.Generator,
    max_trades: int = 50,
) -> tuple[int, int, list[str]]:
    """Run mutate-future test for a single feature. Returns (passes, fails, errors)."""
    passes = 0
    fails = 0
    errors: list[str] = []

    sample = trajectories[:max_trades] if len(trajectories) > max_trades else trajectories

    for traj in sample:
        bars = traj.get("pnl_per_bar", [])
        if len(bars) < 3:
            continue
        pnls = np.array([b["pnl"] for b in bars], dtype=np.float64)
        n = len(pnls)

        # Test every bar except the first and last (where future is empty)
        for t in range(0, n - 1):
            # Original feature value
            try:
                v_orig = fn(pnls, t)
            except Exception as e:
                errors.append(f"{name} trade={traj.get('trade_id','?')} t={t} orig raised {e!r}")
                fails += 1
                continue

            # Mutate future bars: replace pnls[t+1:] with random values in plausible range
            mutated = pnls.copy()
            future_n = n - (t + 1)
            mutated[t + 1 :] = rng.uniform(-1e4, 1e4, size=future_n)

            # Recompute on mutated trajectory
            try:
                v_mut = fn(mutated, t)
            except Exception as e:
                errors.append(f"{name} trade={traj.get('trade_id','?')} t={t} mut raised {e!r}")
                fails += 1
                continue

            # Bit-exact: a causal feature must produce identical output
            if v_orig == v_mut:
                passes += 1
            else:
                fails += 1
                if len(errors) < 5:
                    errors.append(
                        f"{name} trade={traj.get('trade_id','?')} t={t}: "
                        f"orig={v_orig!r} mut={v_mut!r} (LEAK)"
                    )

    return passes, fails, errors


def audit_h3e_path_greeks(rng: np.random.Generator) -> tuple[int, int, list[str]]:
    """Verify per-bar Greeks at bar t depend only on sidecar row at bar t.

    Synthesizes a sidecar with 30 bars and 1 contract, builds paths, snapshots
    path values at bar t, mutates sidecar rows at bars > t, rebuilds paths,
    asserts bit-exact equality for each Greek field at bar t.
    """
    n_bars = 30
    rows_per_bar = 1
    n_features = 22  # CONTRACT_FEATURE_FIELDS length

    passes = 0
    fails = 0
    errors: list[str] = []

    # Build a deterministic baseline sidecar
    base_features = rng.uniform(-1.0, 1.0, size=(n_bars * rows_per_bar, n_features)).astype(np.float64)
    # Force valid=1 so rows are kept
    base_features[:, 0] = 1.0
    # Realistic mid > 0 to avoid divide-by-zero in downstream norms
    base_features[:, 3] = rng.uniform(0.5, 5.0, size=base_features.shape[0])

    bar_ptrs = np.arange(0, (n_bars + 1) * rows_per_bar, rows_per_bar, dtype=np.int64)
    row_contract_idx = np.zeros(n_bars * rows_per_bar, dtype=np.int32)

    sidecar_orig = {
        "row_features": base_features.copy(),
        "row_contract_idx": row_contract_idx.copy(),
        "bar_ptrs": bar_ptrs.copy(),
    }
    paths_orig = _build_contract_paths(sidecar_orig, n_bars)
    cid = 0
    p_orig = paths_orig[cid]

    # Snapshot path Greeks at every t < n_bars - 1
    for t in range(0, n_bars - 1):
        ref_iv = p_orig.ives[t]
        ref_delta = p_orig.deltas[t]
        ref_ttp = p_orig.theta_to_premiums[t]
        ref_gd = p_orig.gamma_dollars[t]

        # Mutate sidecar rows at bars > t
        mutated = base_features.copy()
        for future_bar in range(t + 1, n_bars):
            mutated[future_bar, :] = rng.uniform(-100.0, 100.0, size=n_features)
            mutated[future_bar, 0] = 1.0  # keep valid

        sidecar_mut = {
            "row_features": mutated,
            "row_contract_idx": row_contract_idx.copy(),
            "bar_ptrs": bar_ptrs.copy(),
        }
        paths_mut = _build_contract_paths(sidecar_mut, n_bars)
        if cid not in paths_mut:
            errors.append(f"H3e t={t}: contract not in mutated paths")
            fails += 1
            continue
        p_mut = paths_mut[cid]

        # Each Greek at bar t must be unchanged (NaN-aware: nan == nan considered equal)
        def equal_or_nan(a, b):
            if np.isnan(a) and np.isnan(b):
                return True
            return a == b

        all_equal = (
            equal_or_nan(p_mut.ives[t], ref_iv)
            and equal_or_nan(p_mut.deltas[t], ref_delta)
            and equal_or_nan(p_mut.theta_to_premiums[t], ref_ttp)
            and equal_or_nan(p_mut.gamma_dollars[t], ref_gd)
        )

        if all_equal:
            passes += 1
        else:
            fails += 1
            if len(errors) < 5:
                errors.append(
                    f"H3e t={t}: iv {p_mut.ives[t]} vs {ref_iv}, delta "
                    f"{p_mut.deltas[t]} vs {ref_delta}, ttp {p_mut.theta_to_premiums[t]} "
                    f"vs {ref_ttp}, gd {p_mut.gamma_dollars[t]} vs {ref_gd}"
                )

    return passes, fails, errors


def main():
    print("=== H3a + H3e Look-ahead Audit ===\n")

    with open("v3/artifacts/research/fw_trade_trajectories.pkl", "rb") as f:
        trajectories = pickle.load(f)
    print(f"H3a: testing 3 trajectory features on {len(trajectories)} trade trajectories\n")

    rng = np.random.default_rng(seed=42)

    all_pass = True
    for name, fn in FEATURES.items():
        passes, fails, errors = audit_feature(name, fn, trajectories, rng, max_trades=50)
        status = "PASS" if fails == 0 else "FAIL"
        print(f"  {name:25} {status}  ({passes:5d} bars audited, {fails} leaks)")
        for err in errors[:5]:
            print(f"    {err}")
        if fails > 0:
            all_pass = False

    print()
    print("H3e: testing 4 per-bar contract Greeks via _build_contract_paths\n")
    passes, fails, errors = audit_h3e_path_greeks(rng)
    status = "PASS" if fails == 0 else "FAIL"
    print(f"  per-bar contract Greeks   {status}  ({passes:5d} bars audited, {fails} leaks)")
    for err in errors[:5]:
        print(f"    {err}")
    if fails > 0:
        all_pass = False

    print()
    if all_pass:
        print("✓ AUDIT PASSED. All H3a + H3e features are causal.")
        print("  v3/layer3/common.py and v3/oracles/opportunity.py are safe to use.")
        return 0
    else:
        print("✗ AUDIT FAILED. Do NOT integrate any feature with leaks. Fix the implementation.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
