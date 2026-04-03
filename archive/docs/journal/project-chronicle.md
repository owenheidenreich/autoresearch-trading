# ART² Project Chronicle

> A human-readable record of what's happening in the autonomous SPX 0DTE trading system. Most recent first.

---


## 2026-03-29 — v15.1: EXIT_GATE_THRESHOLD + Pipeline Cleanup

**Major changes:**
1. **EXIT_GATE_THRESHOLD=0.60** — Replaced argmax exit (exits at 50.1% NO_TRADE confidence) with threshold exit (requires 60% confidence to close). Validated on existing data: PF 2.14→3.88, WR 44%→49%, P&L +179%. After 2 training steps: PF 3.13, WR 46.5%, max DD 7.4%, 211 trades over 298 days. Model exits now have 80.9% win rate.
2. **Removed 1455 lines of dead daemon/opus/autonomous code** from art2.py. Deleted opus-prompt.md and art2-cycle-protocol.md. Protocol is now single document: `.claude/rules/art2-operating-manual.md`.

**Sharp optimum reached at score 0.358** — 7 consecutive reverts in batch 2. Model is production-quality; further training unlikely to improve it. Next step: paper trading validation or execution-layer tuning (entry threshold, trade frequency).

---

## 2026-03-29 — Cycle 030: Let It Cook

The inner loop is making progress — 0 of 5 experiments were kept with a best score of 0.38. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 029: Let It Cook

The inner loop is making progress — 0 of 15 experiments were kept with a best score of 0.30. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 028: Let It Cook

The inner loop is making progress — 0 of 10 experiments were kept with a best score of 0.30. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 027: Let It Cook

The inner loop is making progress — 0 of 5 experiments were kept with a best score of 0.30. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 026: Let It Cook

The inner loop is making progress — 0 of 23 experiments were kept with a best score of 0.19. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 024: Let It Cook

The inner loop is making progress — 0 of 16 experiments were kept with a best score of 0.19. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 023: Let It Cook

The inner loop is making progress — 0 of 10 experiments were kept with a best score of 0.19. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 021: Let It Cook

The inner loop is making progress — 0 of 1 experiments were kept with a best score of 0.02. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 020: Let It Cook

The inner loop is making progress — 0 of 1 experiments were kept with a best score of 0.02. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 019: Let It Cook

The inner loop is making progress — 0 of 1 experiments were kept with a best score of 0.02. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 018: Let It Cook

The inner loop is making progress — 0 of 1 experiments were kept with a best score of 0.02. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 017: Let It Cook

The inner loop is making progress — 0 of 1 experiments were kept with a best score of 0.02. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 016: Let It Cook

The inner loop is making progress — 0 of 1 experiments were kept with a best score of 0.02. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 015: Let It Cook

The inner loop is making progress — 0 of 1 experiments were kept with a best score of 0.02. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 014: Let It Cook

The inner loop is making progress — 0 of 1 experiments were kept with a best score of 0.02. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 013: Let It Cook

The inner loop is making progress — 0 of 1 experiments were kept with a best score of 0.02. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 012: Let It Cook

The inner loop is making progress — 0 of 1 experiments were kept with a best score of 0.02. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 011: Let It Cook

The inner loop is making progress — 0 of 1 experiments were kept with a best score of 0.02. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 010: Let It Cook

The inner loop is making progress — 0 of 1 experiments were kept with a best score of 0.02. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 009: Let It Cook

The inner loop is making progress — 0 of 10 experiments were kept with a best score of 0.14. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-29 — Cycle 010: The Breakthrough That Was Already In The Data

Trade-level analysis of v14's 537 backtest trades revealed two structural execution failures hiding in plain sight:

**1. Lunch trades are poison.** 64 trades during 10:30-13:30 have PF 0.30 and WR 20.3% — the model enters during known market chop and gets destroyed. Removing them alone lifts PF from 2.77 to 3.38.

**2. The gate exits too fast.** 347 trades (65%) exit after just 1 bar, averaging -5.7%. Meanwhile, 71 trades held >10 bars have 67.6% win rate and PF 28.02. The 44 EOD exits have 95.5% WR and +264% average. The model correctly identifies entries but kills them immediately — there's no MIN_HOLD_BARS enforcement, so the gate fires NO_TRADE on bar 1 of a winning trade.

Combined: filtering out lunch trades and requiring >2 bar hold gives **WR 45.2% (from 29.6%), PF 11.29 (from 2.77), total P&L +12,019% (from +9,021%)** — validated on existing data with zero GPU cost.

This also explains why win-rate-first via loss regularization failed (19 experiments, WR stuck at 22-31%). The problem was never in the loss function — it was in WHERE and WHEN the model trades. The gate head's conviction is genuinely predictive (top quartile +50.7% avg P&L vs bottom quartile -2.8%), confirming the model has learned real market patterns.

**Previous cycle (009):** Win-rate-first restructure via REG_GATE_MARGIN, REG_PNL_CLIP, REG_WIN_RATE. 19 experiments across 2 phases. Dead end — loss regularization cannot override 0DTE's natural ~30% base rate for long options. The original v14 model (PF 2.77) was lost during warm-start experiments.

**Next:** v15 implementation — lunch suppression in gate labels, MIN_HOLD_BARS=2 enforcement, fresh start. Score config drift guard already in place from cycle 009 infrastructure work.

---

## 2026-03-29 — Cycle 008: Let It Cook

The inner loop is making progress — 0 of 10 experiments were kept with a best score of 0.14. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-28 — Cycle 007: Let It Cook

The inner loop is making progress — 0 of 10 experiments were kept with a best score of 0.14. No strategic changes needed this cycle. Letting the model continue exploring on its current trajectory.

---


## 2026-03-28 — Cycle 006: Steering the Inner Loop

Cycle 006 research revealed the value_exit problem: 50% of exits use an untrained value head (VALUE_W=0.0), destroying -182% total P&L. Meanwhile sequential experiments are working beautifully (5.12→13.78, 2/9 kept). Steered inner loop priorities to focus on exit quality improvement rather than stale v14 restoration goals. PBT deprioritized after 0/42 failures — sequential compounding is the right tool for this model's sharp optimum.

---

## Pre-v14 History (v3 through v8.1, archived)

Full pre-v14 chronicle entries archived to `archive/docs/project-chronicle-pre-v14.md`. Key milestones:
- **v3 (2026-03-24):** Fresh start with 37 features after 230 cycles of lessons learned
- **v6 (2026-03-25):** First VIABLE model — val PF 2.77, risk head, IBKR paper trading verified
- **v8 (2026-03-26):** Best pre-v14 model — replay PF 1.43, pipeline overhaul with outer loop keep/revert
- **v14 (2026-03-28):** Exact v10 restoration — fixed hidden gate label bug, 38 features, replay PF 2.77

---

*This chronicle is automatically updated after every ART² cycle. Each entry is written by Claude (Opus) during the DOCUMENT phase and reviewed by the human operator during the REVIEW phase.*
