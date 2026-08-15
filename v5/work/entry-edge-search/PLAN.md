# Goal packet — find an entry signal with positive gross edge

**Job 34. Written 2026-08-13 for an autonomous autoresearch loop.**
Read this file completely before acting. Read [`v5/STATUS.md`](../../STATUS.md) and
[`v5/AGENTS.md`](../../AGENTS.md) first; they bind and this file does not override them.

---

## 1. The goal

> **Find an entry signal whose out-of-fold GROSS profit and loss per trade is positive, on the owned SPXW
> 0DTE corpus, with a session-block bootstrap lower bound above zero after correcting for every experiment
> run — and which beats both a random-selection reference and a shuffled-label null.**

**Gross, not net.** Gross means before the round trip. This is deliberate and it is the whole point.

## 2. Why that is the target, and not "a profitable strategy"

Five things are measured, settled, and must be treated as given:

| Fact | Value | Where |
|---|---:|---|
| Measured aggressive round trip, near-ATM | **$23.00** | [fill quality](../../../v4/audit/autoresearch/fill_quality_2026_08_13/receipt.json) |
| Fees only (unreachable for a taker) | $3.08 | same |
| Gross P&L, hold to a 30-min clock | **−$7.3** | [optimal exit](../../../v4/audit/autoresearch/optimal_exit_2026_08_13/receipt.json) |
| Gross P&L, optimal-stopping exit | **−$0.2** | same |
| Gross P&L, perfect-foresight exit | +$350.8 | same |

The exit is **solved**. Formulated as optimal stopping it removes the option's time decay exactly and stops
— gross goes to zero and no further. Because the round trip is paid once whenever the trade closes, **no
exit can trade around it**. Therefore every dollar of profit must come from the entry, and an entry that
cannot produce positive *gross* can never produce positive *net*.

So the search is over entries, the metric is gross, and $0.00 is the bar to beat first. Net matters only
after gross is positive.

## 3. The loop protocol

One iteration is: **hypothesise → change ONE thing → run → analyse → annotate → keep or revert.**

1. **State the hypothesis in one sentence** before writing code, including what would falsify it.
2. **Change exactly one thing** from the current best configuration — one new feature, one new label, one
   new selection rule, one new population. Not two. If a change requires another change to be meaningful,
   that is two iterations.
3. **Run it** through the existing harness (§6) so the references and controls come for free.
4. **Analyse against the three references**, never against zero alone:
   - the current best configuration,
   - a **random selection matched on trade count**,
   - a **shuffled-label null** retrained the same way.
5. **Annotate** — append one entry to `LOG.md` in this directory, using the format in §8. Every iteration
   gets an entry, including failures. Especially failures.
6. **Keep or revert.** If the change does not beat the current best on gross, with its interval, revert to
   the best configuration and record that you did. Do not carry a change forward because it "seems
   promising".

**Never run two changes and attribute the result to one of them.**

## 4. Hard rules — these are not style preferences

These encode defects that were found the expensive way on 2026-08-13. Each cost real work.

### 4.1 Timestamp-audit every feature, every iteration

For each feature, answer in writing: **could the bot have known this value in the minute it decided?**
A feature whose window ends after the decision minute is refused, even by one minute.

**The shuffled-label null cannot catch this and never will.** Permuting the label removes the thing a
leaked feature would leak *to*, so the control comes back clean while the feature is still poisoned. On
2026-08-13 one such feature — the underlying's move in the minute *after* entry — lifted a headline from
break-even to **+$119/trade with an interval that cleared zero**. Only the audit found it.
See [`test_optimal_exit.py`](../../tests/test_optimal_exit.py) for the encoded version.

### 4.2 Never filter a trade on anything measured after entry

No dropping slots because the underlying "did not move". No dropping contracts that stopped printing. Both
were done in this project, both looked like data hygiene, and both were look-ahead worth **3+ accuracy
points** in the flattering direction. A slot may be skipped only when it cannot be *priced* at the entry
minute.

### 4.3 Chronological validation only

Train on strictly earlier sessions; predict later ones. No random splits, no shuffling of session order,
no fitting a scaler on the full sample. Any threshold or quantile cut must be computed from the **training
fold's** own predictions.

### 4.4 Score the whole declared family

Enumerate and hash the family before running. Report every member. Never select the best member and
present it as the result. Apply the Bonferroni correction for the number of cells actually inspected —
including the ones you inspected and did not report.

### 4.5 Resample from strata, not quantiles

A uniform draw from a quantile grid cannot exceed the top quantile, so its mean is a trimmed mean. On
option payoffs that understated winning trades by 6–19% while leaving the bounded loss side accurate.
Use [`strata_means`](../../ops/measure_hold_occupancy.py).

### 4.6 A wide interval containing zero is not a result

Report the session-block bootstrap interval next to every point estimate and read the interval. A cell
reported as "+$2.80, CI [−33.6, +42.1]" did not reproduce and should have been called noise at the time.

### 4.7 A large or surprising result is a suspected bug

Before reporting anything that looks good, try to break it. State the mechanism that would produce it
honestly, and the mechanism that would produce it by accident, and rule the second out.

## 5. Closed territory — do not retest

These are in [`DO_NOT_RETEST.md`](../../research/history/DO_NOT_RETEST.md) with full evidence. Re-running
any of them is not a new experiment.

| Closed | Result |
|---|---|
| Direction labels — 7 declared rules and a learned ranker | All negative. A rule right 53.52% of the time still lost, because it was right where options were dearest |
| Magnitude / volatility labels | Predictable and **fully priced**: selecting the busiest third raises the move +51% and the premium +42%, P&L moves 30 cents |
| Cheapness / variance-premium timing | The premium is −0.55%/15min and the spread is 2.83%; both sides lose at retail execution |
| Percentage-excursion entry label | Doubles the excursion and **loses more than random** — percentage selects cheap contracts against a fixed cost |
| Exit formulations, both threshold and optimal stopping | Solved; gross ≈ $0 |
| Passive / midpoint execution | Saves ~$10 of spread, costs $60–86 of adverse selection |
| Buying premium and holding to a clock | Negative before a signal is applied |

**Genuinely new** means a label, population, or information source none of the above used. Some candidates
that have never been tried, in no particular order and with no promise attached: chain-wide state (skew,
term structure, put/call volume balance), the underlying's relationship to overnight or prior-day levels,
event-calendar conditioning, cross-strike relative value, order-flow imbalance if a source exists, and
regime conditioning that is calibrated causally.

## 6. Where everything is

### Corpora — read-only, never modify

| What | Path | Contents |
|---|---|---|
| Trade corpus | `/Volumes/AR_TRADING_DATA/spxw_0dte_2022-06-01_2026-07-31/raw/` | 1,045 usable sessions, `ohlcv-1m`, **open/high/low/close** per contract-minute |
| Quote corpus | `~/.autoresearch-trading/pathd_2025-08-01_2026-07-31/aligned/normalized/` | 251 sessions, bid/ask/size, vendor greeks |

### Derived datasets — rebuild these first, they are not committed

Write to `/Volumes/AR_TRADING_DATA/derived/`. Each takes roughly fifteen to twenty-five minutes.

```bash
./.venv/bin/python -m v5.ops.build_magnitude_dataset \
    --out /Volumes/AR_TRADING_DATA/derived/magnitude.parquet

./.venv/bin/python -m v5.ops.build_exit_dataset \
    --out /Volumes/AR_TRADING_DATA/derived/exit_greeks.parquet
```

### The harness — extend these rather than starting over

| Module | Does |
|---|---|
| [`ops/build_exit_dataset.py`](../../ops/build_exit_dataset.py) | Per-minute decision points, greeks attached |
| [`ops/build_magnitude_dataset.py`](../../ops/build_magnitude_dataset.py) | Per-slot entry features and payoffs |
| [`ops/train_optimal_exit.py`](../../ops/train_optimal_exit.py) | The solved exit. **Use it; do not re-derive it** |
| [`ops/train_full_system.py`](../../ops/train_full_system.py) | Entry + exit, with the references and nulls wired in. **This is where most iterations belong** |
| [`research/greeks.py`](../../research/greeks.py) | IV and greeks from price. Live-parity safe |
| [`research/statistics.py`](../../research/statistics.py) | Bootstrap quantiles, Bonferroni, detectable accuracy |
| [`research/autoresearch/budget.py`](../../research/autoresearch/budget.py) | Hash-chained alpha ledger |

Receipts go to `v4/audit/autoresearch/<name>_<date>/receipt.json`. Never overwrite another run's directory.

## 7. Guardrails — authorization and safety

- **No data purchases.** No Databento download, no vendor cost probe, no new schema. If an experiment needs
  data we do not own, stop and write the request down; the owner decides. Tier 1.
- **No broker contact, no order submission, live or paper.** Tier 1.
- **Nothing from 2026-08-06 onward.** Those sessions are reserved by a signed forward-confirmation
  declaration. The corpus ends 2026-07-31; keep it that way.
- **Do not edit** `v4/` evidence, receipts, or any file under `v4/audit/`.
- **Everything must be runnable on a live OPRA feed.** Before adding a feature, state how the live system
  computes it at the decision instant. Greeks are computed from price for this reason — a vendor's greek is
  a train/live divergence. Anything needing data the live path does not have is out of scope.
- **Run before claiming health:** `./.venv/bin/python -m pytest v5/tests -q` and
  `./.venv/bin/python v5/ops/check_project.py`. New code carries tests in proportion to its risk.
- **Do not commit.** Leave the worktree for the owner to review.

## 8. The log — one entry per iteration, in `LOG.md`

```markdown
### Iteration N — <one-line title>

- **Hypothesis:** <one sentence, plus what would falsify it>
- **The one change:** <exactly what differs from the current best>
- **Timestamp audit:** <every new feature, and the minute its value is knowable>
- **Result:** gross $X/trade, CI [lo, hi]; best-so-far $Y; random $Z; shuffled null $W
- **Verdict:** KEPT / REVERTED, and why
- **Receipt:** v4/audit/autoresearch/<dir>/receipt.json
```

Keep a `BEST.md` in this directory naming the current best configuration, its gross figure and interval,
and the receipt that produced it. Update it only when a change is KEPT.

## 9. When to stop

**Stop and report success** the moment a configuration shows positive gross with a bootstrap lower bound
above zero after multiplicity correction, beating both references. Do not keep tuning it. Write it up,
freeze it, and hand it to the owner for a known-answer campaign — that gate has stopped this project twice
and it is not to be skipped.

**Stop and report failure** when any of these is true:

- **40 iterations** have run without a single KEPT change that produced positive gross;
- the hypotheses being generated are variations of §5's closed territory;
- an experiment would need data or authorization the owner has not given.

**A negative result reported honestly is a success.** This project's two most valuable findings this year
were a defect and a null. Do not manufacture a positive.

## 10. Statistical context, so the loop is not calibrated wrongly

Multiplicity is **cheap** here and should not induce timidity. From the alpha ledger, on the corrected
60-minute payoffs:

| Experiments run | True accuracy needed |
|---:|---:|
| 1 | 57.89% |
| 30 | 58.71% |
| 400 | 59.17% |
| 5,000 | 59.55% |

Going from one experiment to five thousand costs **1.66 accuracy points**. Run many hypotheses. The
constraint has never been the number of attempts — it is that no attempt has yet produced positive gross.
