# The drawdown-ordered lifecycle experiment closes UNDERPOWERED at its own preflight

**2026-08-15. No real fit ran; no real outcome was opened.** Authorized by the owner's signed
STOP override and scoped reopening of ledger rows 340/341 after the short-vertical census fired
the program STOP. Kill condition 1 — a binding known-answer preflight — fired first, exactly as
designed.

## What the bot can and cannot do, in one paragraph

The owner's design (enter where predicted post-entry drawdown stays inside −30%, stop at the
−30% touch, otherwise ride 60 minutes) is coherent and cheap to break even: the measured
break-even is **34.2% precision against a 31.8% base rate — a +2.4-point gap**, the smallest this
project has ever priced, and the stop demonstrably controls the tail (loser dispersion $613 →
$239). But on the owned corpus geometry (~150 scorable sessions after a training prefix, ~$1,000
per-trade dispersion), the full evidence gate — serial account, session bootstrap at 0.05/3,
4/5 chronological blocks — **cannot reliably certify even a +20-point precision edge**, an edge
more than eight times break-even. Profitable-but-unprovable is not a pass; the experiment closes
before it can be run.

## The preflight

Declared in `v5/work/drawdown-lifecycle/PREFLIGHT_DECLARATION_V1.json` (self-hash `596f7b5f…`),
run under `v5/ops/run_drawdown_preflight.py` with the declaration, law, implementation and
calibration hashes all verified before execution. Synthetic worlds are calibrated to **32,976
measured owned quote paths** (winners +$641 SD $1,086, losers −$333 SD $239, base rate 31.8%,
measured within-session clustering), the trainer mirrors the production law (standardised clipped
features, AdamW, gradient clipping, best-checkpoint restore, SHA-256 seeds), and "recovery" is a
decision of the complete gate — never a per-minute score diagnostic. The campaign seed bank is
disjoint from every calibration probe, per the capacity-campaign no-reuse rule.

| Arm | Result |
|---|---|
| Planted minimum edge (≈ +20pp selected precision; models learned it — mean selected precision ≈ 52%) | **25/40 recovered = 62%**, Wilson upper **73.9%**, against 80% required |
| 60 null worlds | **0 full-gate false passes** (Wilson upper 4.31%); the corrected bound refused every lucky positive mean, up to +$78/session |

Verdict: **`PREFLIGHT_FAILED_UNDERPOWERED`.** Receipt:
`v4/audit/autoresearch/drawdown_preflight_2026_08_15/receipt.json`.

## Interpretation discipline

- This is **not** "the information does not exist" and **not** "the market is efficient." It is:
  the owned corpus cannot certify the effect at any size worth believing. The certifiable floor
  sits above +20pp of precision; the largest honest entry effect this project ever measured was
  +7.1pp on another label, and it was fold-unstable.
- The failure is the same 1/√n measurement wall the project has hit at every gate (verdict B,
  G1 twice, job 15, row 343) — now measured for this experiment specifically, before any real
  outcome could be opened, which is the entire point of the preflight discipline adopted after
  the 2026-08-15 capacity-campaign corrections.
- The null side is a genuine positive result for the machinery: the full-gate false-pass control
  works under realistic clustering and a measured −$22.8/trade drag.

## Consequences

- Per the signed reopening's kill 1 and the override's §4, **the reopening and the override are
  spent.** The 2026-08-15 program STOP resumes in full force. Ledger rows 340/341 bind again,
  now with this row added.
- The remaining honest lever is arithmetic, recorded here without recommendation: the certifiable
  precision floor scales as 1/√(scored sessions). At roughly ten years of owned SPXW **quote**
  history (~2,500 sessions, ~$1,220 at the recorded $122/session-year estimate), the floor drops
  to roughly **+8pp** — the first corpus size at which an edge of the largest size ever measured
  here would be certifiable. Any such purchase is Tier-1, and this preflight harness (which now
  exists and runs in minutes) must pass at the enlarged geometry **before** money moves.

## Evidence

- Receipt: `v4/audit/autoresearch/drawdown_preflight_2026_08_15/receipt.json`
- Declaration: `v5/work/drawdown-lifecycle/PREFLIGHT_DECLARATION_V1.json`
- Harness: `v5/research/drawdown_preflight.py` (14 tests in `v5/tests/test_drawdown_preflight.py`)
- Calibration: `AR_TRADING_DATA/derived/drawdown_calibration_v1.parquet` (sha `83384d6f…`),
  exported from `quoted_exit_paths.parquet`
- Authorization and spending: `v5/governance/STOP_OVERRIDE_DRAWDOWN_LIFECYCLE_2026_08_15.md`,
  `v5/governance/LEDGER_REOPENING_DRAWDOWN_LIFECYCLE_2026_08_15.md` (signed bytes unedited;
  spent by their own terms on this verdict)
