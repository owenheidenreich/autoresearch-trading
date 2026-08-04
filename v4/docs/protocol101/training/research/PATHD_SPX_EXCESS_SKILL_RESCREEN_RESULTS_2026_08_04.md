# Path-D SPX Excess-Skill Re-Screen (Option 0) — RESULTS (2026-08-04)

**Status: `EXCESS_SKILL_CANDIDATE` — one survivor of sixty. DIAGNOSTIC-ONLY; nothing here is
confirmatory.**

Pre-registration: [`PATHD_SPX_EXCESS_SKILL_RESCREEN_PREREGISTRATION_2026_08_04.md`](PATHD_SPX_EXCESS_SKILL_RESCREEN_PREREGISTRATION_2026_08_04.md)
(frozen and committed as `b29ed007` before the run; not edited).
Evidence: `v4/audit/autoresearch/pathd_spx_excess_skill_rescreen_2026_08_04/`.

---

## 1. The result

213 sessions, 60 members, 200 surrogates per session, 20,000 permutations. Negative control clean, no
member null-dependent.

**Survivors — `omar_clipped_neg3_pos3`, all three horizons, negative sign, and nothing else:**

| Member | raw IC | **excess (wild)** | maxT p | excess (block) | block p | folds |
|---|---|---|---|---|---|---|
| `omar` 60m | +0.522 | **+0.078** | 0.0000 | +0.107 | 0.0000 | 5/5 |
| `omar` 30m | +0.419 | **+0.063** | 0.0002 | +0.084 | 0.0000 | 5/5 |
| `omar` 15m | +0.325 | **+0.052** | 0.0001 | +0.062 | 0.0000 | 5/5 |

`omar` is `(close − session open) / session range`, clipped — **the position of price within the session's
own realized range.** The surviving sign says the index reverts from its session extremes *more* than a
path with identical volatility and no predictability.

**What died, and this is most of the value of the run:**

- **`spx_vwap_gap_*` — dead.** Excess +0.026 to +0.037, but maxT p **0.21–0.99**. The two nulls
  *disagree* here (block bootstrap p 0.001–0.005, wild bootstrap not significant), so it is exactly the
  case the pre-registered two-null rule exists to catch. Its run-1 residual was a block-bootstrap artifact.
- **All momentum members — dead**, confirming run 1.
- **`session_range_bps` — dead.** Excess +0.004 to +0.016, maxT p 0.986–1.000, sign-stable in only 1–2 of
  5 folds. Its run-1 "only 55% explained" was block-bootstrap mis-specification.
- **`minute_of_session` — dead.** Raw IC +0.038 at 60m, maxT p 0.87. **The time-of-session hypothesis is
  not supported.** It was added specifically to test this directly rather than through
  `session_range_bps`, and it does not survive.

## 2. Why the survivor is more credible than run 1's twenty-seven

**The two nulls have complementary weaknesses and both accept `omar`.** The wild bootstrap preserves the
intraday volatility profile exactly (`|e_t|` at every timestamp) but symmetrizes returns, destroying skew.
The block bootstrap preserves skew and the marginal return distribution but shuffles the loud minutes
around. A result that is an artifact of *either* mis-specification should fail one of them — and
`spx_vwap_gap` does exactly that. `omar` survives both.

**Two independent code paths reproduce the raw ICs to machine epsilon.** The re-screen needs a path-based
feature implementation (surrogates have no vendor provenance), which is asserted boundary-by-boundary
against the frozen kernel — max deviation `1.2e-11` — and its raw ICs match run 1's on all 54 shared
members to **1.1e-16**.

**The negative control run 1 lacked now exists and passes.** A held-out surrogate substituted for the real
path produces a null excess; no member clears. The artifact-removal machinery is not manufacturing signal.

## 3. How big is it? — POST-HOC, DESCRIPTIVE

`v4/research/pathd_omar_economic_sizing.py`. An IC is not money. Mean forward 60-minute SPX move by
within-session `omar` decile, 215 sessions, bias-corrected against 100 wild surrogates each:

| decile | real | surrogate | **corrected** |
|---|---|---|---|
| 1 (lowest omar) | +12.218 | +11.347 | **+0.871** |
| 2 | +7.047 | +6.250 | +0.797 |
| 5 | +0.996 | +0.811 | +0.185 |
| 8 | −4.588 | −3.164 | **−1.425** |
| 10 (highest omar) | −10.704 | −9.942 | −0.762 |

- **Raw** bottom-minus-top spread: **+22.92 SPX points**. **93% of it is the artifact.**
- **Bias-corrected** spread: **+1.63 points** → **≈ 0.82 points per trade**.
- On one ES contract that is **≈ $41 per trade**, against an *assumed* round-trip friction of **$17**.

**Read that last line carefully.** It is ~2.4× the friction, and it is the first number in this programme
that has ever been on the right side of that comparison. It is also carrying five caveats:

1. **The correction removes 93% of the raw spread.** +1.63 is a small difference between two numbers near
   +23, and is therefore sensitive to surrogate specification.
2. **The corrected profile is not monotone at the extremes.** Decile 8 (−1.425) is stronger than decile 10
   (−0.762), and deciles 1 and 2 are near-identical. A clean "fade the extremes" story predicts the
   extremes are strongest. They are not. The project's own `decile_monotonicity` rejection test would fail
   this shape.
3. **ES friction is assumed from tick structure, not measured.** That is precisely what Option A buys.
4. **ES is not SPX.** The signal is measured on the index; ES has basis and a near-24-hour session, so
   "session range" is not the same object. **Protocol 028 tested stitched ES VWAP as a feature and rejected
   it.**
5. **The per-trade figure assumes you can harvest the full extreme-decile spread.** Under one-account
   serial occupancy with 60-minute holds, the realizable trade count is far lower than the decile counts
   imply.

## 4. Bottom line

**World B is refuted; World A is not yet established.** After removing the artifact that explained run 1,
something survives — but it is one feature out of ten, at roughly a fifth of the headline, with a
non-monotone decile profile, on the same 213 sessions that generated the hypothesis.

This is a **diagnostic** result by construction. Its entire licensed use is to inform the Option A / B / C
decision, and per the decision key an excess that survives points to **Option A — measure ES friction
before any purchase or training question.** The pre-registration is explicit that the protected holdout is
SPENT, so the only confirmation path for anything here is fresh live-paper observation.

**It does not reopen the 0DTE long-premium class**, which stays closed structurally.

## 5. Evidence and reproduction

```bash
PYTHONPATH=. python -m v4.research.pathd_spx_excess_skill_rescreen   # refuses a second run
PYTHONPATH=. python -m v4.research.pathd_omar_economic_sizing        # post-hoc, descriptive
```

- `receipt.json` — self-sealed (verified MATCH); `diagnostic_only: true`,
  `protected_holdout_opened: false`, `model_trained: false`.
- `results.csv` (60 rows), `results.md`, `omar_economic_sizing.json`.
- Mean surrogate estimation noise **0.0152** per session — independent across sessions, so it inflates the
  across-session standard error and makes the test *harder* to pass, not easier.

*Signed: Claude Opus 5 — 2026-08-04.*
