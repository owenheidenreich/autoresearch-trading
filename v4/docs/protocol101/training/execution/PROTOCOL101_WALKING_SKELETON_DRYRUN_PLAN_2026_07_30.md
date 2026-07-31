# Protocol101 Walking-Skeleton Dry-Run — Staged Plan

**STATUS: PLANNING DOC, NOT AN AUTHORIZATION.** For Codex to build in stages;
each stage stops for owner + Fable review before the next. Not a scientific
campaign. Binding contracts remain the consolidated authority + Graph V2.

Prepared 2026-07-30 (Fable), from the owner's dry-run design.

## What this is (and is not)

An **end-to-end plumbing test** — a "walking skeleton": the thinnest complete
vertical slice through the *entire* real pipeline (train entry → train exit →
combine → replay+visualize → paper-trade live → record → download → 3-way
parity), run with a **real-but-unrefined model**, to prove the pipe connects
before committing to the full data download and the real campaign.

- **"Dumb" means not-smart, not not-doing-anything.** The mock model is built
  *exactly* like the real thing — real feature contract, real composer, real
  serial simulator, real safety rules — using the **gradient-boosted-tree
  baseline** (the honest first rung of the signed model-family ladder). It
  genuinely reasons about trades and tries to be profitable. It is simply not
  escalated to a neural model and not heavily tuned. It is a throwaway
  candidate, not the real champion.
- **It tests PLUMBING, not alpha or scale.** Passing does NOT mean the strategy
  is profitable (the model is unrefined) or that it runs at full-data scale
  (that is a later check). It means every subsystem connects and the
  historical/recorder/live game is truly the same game end-to-end.

## Governance & quarantine (binding for this track)

- **Throwaway everything:** mock model, mock training artifacts, and the small
  data slices are disposable and clearly labeled `walking_skeleton`. They must
  never be promoted, cited as scientific evidence, or confused with the real
  campaign.
- **No protected resources:** the dry-run may NOT touch the protected holdout,
  fresh-confirmation seed, or any sealed evidence. It uses owned
  development-class data only.
- **Sanctioned early-paper exception:** the real graph forbids reaching paper
  before the scientific gates (confirmation/holdout/shadow). This track is an
  explicit, owner-authorized exception **because it is paper-only, plumbing-
  only, and uses a throwaway model** — it makes no scientific claim and never
  touches real money. Real-money paths remain fully closed.
- **Two owner authorizations required before Stages 4-5** (owner will sign when
  ready): (a) re-enable the parity recorder for the test day; (b) IBKR paper
  authorization for the guarded paper-submit path.

## Data (the ~$1 test)

- **Stages 0-3 use ONLY data already owned:** ~250 owned 2025 minute sessions
  (entry) and the 30 owned `cbbo-1s` pilot sessions (exit). Use a small subset
  (e.g. ~20-40 minute sessions for entry, ~10 owned 1s sessions for exit).
  **Zero new download.**
- **The ONLY new download is ONE day** — the live paper-trading day (Stage 4),
  pulled *after the fact* in Stage 5 in both resolutions the trader uses
  (minute for entry + 1-second/tick for exit) for the 3-way comparison.
  **~$1 for one day's 0DTE** (cost-estimate first; hard-cap it).

## Stages (each stops for owner + Fable review)

### Stage 0 — Walking-Skeleton spec
Freeze: the mock model definition (GBT entry + GBT exit, real feature
contract, real composer, real simulator), the exact owned-data slice, the
quarantine labels, the success criteria, and the two pending authorizations.
No build. **Review → proceed.**

### Stage 1 — Entry model (1-minute)
Build the entry tensor pipeline on the owned minute subset; train the quick
GBT entry model; produce entry decisions (WAIT / BUY exact contract) across
the slice. Confirm it is *active* — takes several trades, exercises WAIT and
BUY paths, not a degenerate all-WAIT. **Review → proceed.**

### Stage 2 — Combined entry + exit (1-second)
Build the exit/lifecycle pipeline on the owned 1-second subset; train the
quick GBT exit model; assemble the full trader (minute-entry + 1s-exit +
protective floor). Confirm all action paths fire: WAIT, BUY, HOLD, EXIT,
floor-trigger, forced-flat. **Review → proceed.**

### Stage 3 — Replay & Visualize  *(also formalized into the REAL process — see below)*
Run the assembled full trader across an owned validation slice under the
serial simulator. Emit: historical estimated P&L, the **equity curve**, and
**trades plotted on the SPX price graph**, plus the four-bucket Pickles
outcome distribution. This is the owner's visual read on whether the trader
behaves sanely before it touches a broker. Reuse existing plumbing
(`equity.html` / `trades.csv` from the census sanity path; the trade-chart
exporter). **Review → proceed.**

### Stage 4 — Paper deploy (needs recorder-on + IBKR paper auth)
Freeze the mock trader. Deploy to IBKR **paper** for one live trading day;
take actual guarded paper trades on IB Gateway; log every decision (WAIT /
BUY / HOLD / EXIT / floor) with exact timestamps and the inputs it saw. This
proves model output translates into real executed paper orders. **Review →
proceed.**

### Stage 5 — Record & download
The parity recorder captures the same live day. After close, download that
exact day via Databento in both resolutions the trader uses (minute + tick,
tick downsampled to 1-second per D59). Cost-estimate first; hard-cap ~$5.
**Review → proceed.**

### Stage 6 — Three-way parity (the crown-jewel test)
Run the frozen mock trader three ways on the same day: (a) the live run, (b)
the recorder-replay run, (c) the historical-download run. Assert **identical
decisions at identical timestamps** across all three. Emit a parity report
with any diffs classified. **Target: perfect match; success = a very close
match with every difference explained (no unexplained action/timing
divergence).** **Review → conclude.**

## Success = ready to scale up

If the skeleton passes end-to-end (esp. Stage 6 near/perfect match), the
plumbing is proven. Then, and only then: commit to the full data download
(~$1,100, back to 2023-03-28) and build the **real** refined model for live
paper validation and further testing. If any stage fails, we fix the seam now
— cheaply, on a $1 model — and re-run that stage.

## New permanent step for the REAL process (owner-requested)

**"Replay & Visualize" (Stage 3) is hereby named a formal step in the real
pipeline, not just the dry-run.** Every accepted candidate in the real process
must produce, on its validation replay, the equity curve and the trades-on-SPX
overlay (plus the four-bucket distribution) as a first-class owner-facing
evidence artifact. Fold this into the FT2-21 bundle as an adopted process
addition (proposed **D60**). The visual read is decision-relevant owner
evidence, and the plumbing already exists.
