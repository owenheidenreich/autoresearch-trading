# Codex Goal — Phase 0b: Unblock the Admission Ledger (2026-08-04)

Governing architecture: [`PATHD_BUILD_ORDER_2026_08_04.md`](../contracts/PATHD_BUILD_ORDER_2026_08_04.md).
Phase 0 is verified complete: **8 features `ADMITTED`, 65 in-scope `BARRED`**, enforcement live, ledger
`fbdf4d12…`. **Still no model training in this phase.** End with `STOP_FOR_CLAUDE_VERIFICATION`.

---

## 1. Where we are

Every admitted feature is calendar or contract geometry — `day_of_week`, `minute_of_session`, `strike`,
`is_call`. **Not one market observable is admitted.** No price, no spread, no volatility, no greeks. A model
on this set can learn a date rule, not a trading strategy, so training is correctly blocked.

The 65 barred in-scope features funnel into a small number of unmet receipts. This phase clears them.

## 2. Three tracks

| Track | Unblocks | Features | Needs |
|---|---|---|---|
| **B — parity estimator** | `implied_spot` → `implied_volatility` → `self_computed_greeks` | **21** | nothing; start now |
| **C — account state** | `causal_account_state` | **6** | nothing; start now |
| **A — multi-session live capture** | `cbbo1m_native` → `cross_section`, plus `cbbo1s_rolling`, `ohlcv1m_sparse` | **38** | **owner authorization** |

Tracks B and C are offline and independent. **Start them immediately.** Track A is the long pole because it
accrues on trading-day wall-clock time, so raise the authorization question first, then work B and C while
it runs.

---

## 3. Track B — shared parity estimator (start now, 21 features)

`entry.opra_implied_spot.v1` is the causal spot parent for both implied volatility and the greeks. Certifying
it is the single highest-leverage offline task in the project.

**Build one estimator, used by both paths.** Spot implied from put-call parity on OPRA quotes. The historical
and live paths must call **byte-identical code**, proven by hash — not two implementations that agree today.

Receipts required:
1. **Shared parity-estimator implementation hash.**
2. **Historical/live candidate-pair identity** — the same call/put pairs selected on both paths.
3. **Comparison to completed official SPX, for measurement only.** Read this precisely: official SPX is a
   *validation reference*, never a feature input. Recording the residual is required; feeding it to a model
   is not permitted and would reintroduce a second live feed.

Then, in dependency order:
- `entry.opra_implied_volatility.v1` — parent receipts, shared solver/constants hash, mutation and
  numerical-stability tests.
- `entry.self_computed_greeks.v1` — causal spot parent receipt, shared solver/constants hash, and
  **historical/live golden-vector identity**: fixed inputs, fixed expected outputs, asserted on both paths.

**A Tier-2 family may not be admitted before its parent.** The ledger already enforces this; do not attempt
to work around it.

## 4. Track C — causal account state (start now, 6 features)

`realized_session_pnl`, `remaining_risk_budget`, `position_occupancy` and similar come from **our own
ledger**, not a vendor feed, so the "live twin" is the account state the runtime maintains.

Receipts: historical/live ledger transition identity · serial replay parity · mutate-future invariance.

The first may need paper-account evidence; the other two are offline. If ledger-transition identity requires
live paper state, say so and stop rather than substituting a simulation and calling it identity.

## 5. Track A — multi-session live OPRA capture (needs authorization)

`cbbo1m_native` is barred on **"multi-session local receipt-latency distribution."** Owned evidence covers
**one** session (2026-08-03, 510 symbols).

**Before connecting anything:**
- Confirm the Databento **live** OPRA entitlement already exists and that streaming capture carries **zero
  marginal cost**. A live subscription and a paid historical download are different things; if capture would
  incur cost, **stop and report** — that is a separate owner authorization.
- Note for the owner that this is a live market-data connection and therefore a CLAUDE.md hard stop
  requiring explicit authorization, even at zero cost.

**Capture design, declared before starting:**
- **≥5 regular sessions**, ideally including at least one elevated-volatility day. Fewer than 5 cannot
  support a session-to-session distribution claim; declare the count in the receipt rather than deciding
  after seeing the data.
- Record **per-message local receipt timestamps** so the latency distribution is measured, not modelled.
- **Measure the clock per family.** Do not reuse the 2,336 ms ThetaData figure, and do not reuse the
  contract-clock 30,603.667 ms figure either — Databento OPRA CBBO p99 was 319.5 ms in the shared receipt,
  and conflating sources is what invalidated Wave 2's H3.
- Capture also serves `cbbo1s_rolling` (same-session value identity, reconnect/no-update mutation tests) and
  `ohlcv1m_sparse` (OHLCV latency, sparse zero-fill invariance under
  `SYNTHESIZE_ZERO_ONLY_AFTER_HEALTHY_FROZEN_CUTOFF_NEVER_CARRY` — never carry a prior bar forward).

## 6. Exit gate

Per track: every required receipt produced and hash-verified, the ledger regenerated, and its
`ledger_sha256` recomputed. A family is `ADMITTED` only when **all** its receipts verify **and** its parent
is admitted. Partial credit is not a status.

**Do not weaken a receipt to reach `ADMITTED`.** Widening a tolerance until identity passes is the same class
of error as rewriting a frozen receipt to make a test go green.

**A family that cannot be certified is a finding.** If put-call parity spot proves too noisy on 0DTE quotes
to certify, that is a real architectural result about whether a one-feed decision plane is achievable — state
it plainly. It would mean the greeks are unreachable without a second feed, which is an owner decision, not a
workaround.

## 7. Hard stops

- **No model training, fitting, or hyperparameter search.**
- Protected 36-session firewall SPENT — never reopen; `holdout_open_count` stays 0.
- Do not modify `FILL_LAW`, the causal t−60s clock, the label law, the OOF firewall, or the enforcement
  path in `pathd_feature_admission_ledger.py`.
- **No paid data.** No broker, orders, paper submit, promotion, default change, or runtime/launchd edits.
- Live capture only after explicit owner authorization, even at zero marginal cost.

## 8. Deliverable

Per-track certification report (receipts, hashes, measured clocks, tolerances, and any family that failed
with its reason), the regenerated ledger with a recomputed hash, an updated `ADMITTED`/`BARRED` count, and an
updated status board. Then `STOP_FOR_CLAUDE_VERIFICATION`.

**Claude will verify:** re-derive the ledger hash; confirm each newly-admitted feature's receipts exist and
verify; confirm no Tier-2 family was admitted before its parent; confirm each family's availability clock was
measured from its own source rather than inherited; confirm official SPX was used only as a validation
reference and never as a feature input; and re-run the enforcement tests to confirm training still fails
closed.

*Prepared by Claude Opus 5 — 2026-08-04.*
