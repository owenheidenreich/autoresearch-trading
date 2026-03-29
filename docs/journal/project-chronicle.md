# ART² Project Chronicle

> A human-readable record of what's happening in the autonomous SPX 0DTE trading system. Most recent first.

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
