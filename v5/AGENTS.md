# V5 Ground Rules

**Read [`STATUS.md`](STATUS.md) first.** It is the only status, job-register, and gate-chain document.
This file says how to work. `CLAUDE.md` must remain byte-identical to this file.

## 1. The project question

We are building an automated bot that buys SPX 0DTE calls and puts. The one open economic question is
whether SPX/ES direction over 15–60 minutes can be predicted well enough to clear measured round-trip
friction of 0.358 ES points / $17.92.

Every proposal must state how it serves that question. Model training on the option layer is prohibited
until G1 direction passes. The current G1 plan is itself on hold for the independent measurement review.

## 2. Write for the owner

The owner is the decision-maker and is not a full-time quant.

- Use plain English and define a term of art the first time it appears.
- Lead with what changed and what it means for the bot.
- If a finding changes what can be built, say so in one sentence at the top.
- Do not use unexplained protocol numbers, codenames, or status tokens.
- Say directly when something does not work.

## 3. Decision tiers

| Tier | Covered actions | Decision-maker |
|---|---|---|
| **1 — owner always** | Spend money; contact broker or paid vendor; touch real money; delete or move data; install/change/remove unattended jobs; change paper defaults; promote a model; open reserved evidence | Ask, recommend, wait |
| **2 — agent decides and logs** | Research method; null choice; document structure; narrowing authorized scope; test selection; repair design | Agent, with one-line reason at checkpoint |
| **3 — agent acts** | Code, tests, analysis, reading, and local validation that contacts nothing | Agent |

Tier 2 is the default when the owner would say “use your judgment.”

## 4. V5 file policy

- There is no generic `v5/docs/` directory.
- `STATUS.md` is the only current status, roadmap, job register, or gate chain.
- Claude or Codex planning files may exist only at `v5/work/<registered-job>/PLAN.md`.
- A work directory must appear in `STATUS.md` before it gains files.
- Stable rules belong at the v5 root or under `governance/`; findings belong under `research/`.
- Large evidence remains external and must be named in `evidence/INDEX.md`.
- When work closes, move its packet to `v5/history/jobs/<job>/` and record the durable result in STATUS,
  the do-not-retest ledger, or the evidence index.
- Never create a second source-of-truth, current-status, roadmap, general handoff, or gates document.

Run `./.venv/bin/python v5/ops/check_project.py` before claiming the project structure is healthy.

## 5. Research evidence

- Check [`research/history/DO_NOT_RETEST.md`](research/history/DO_NOT_RETEST.md) before proposing work.
- Every completed research finding goes into that ledger with the measured result and a genuinely new
  reopening condition.
- Ground claims in a file, receipt, test, log, or measurement. Missing evidence is `UNKNOWN`.
- Runtime evidence beats documentation. Report conflicts; do not silently resolve them.
- Large or surprising results are suspected bugs until their mechanism is proven.
- Every feature must be computable from data that had actually arrived at the decision time.
- The causal clock, fill law, label law, and out-of-fold firewall are frozen. Fix conflicting documentation;
  never loosen a gate to match it.

## 6. Hard safety rules

Do not run anything that can trade, contact a broker, download paid data, mutate runtime state, install or
remove scheduled jobs, train a model, tune a threshold, promote a candidate, or open reserved evidence
without explicit owner authorization in the current conversation and a fresh read of the relevant safety
file.

This includes IBKR/order/live/paper-submit scripts, Databento or Polygon downloads, model fitting,
threshold search, runtime flag edits, `launchctl` mutations, plist installation, and cleanup outside a
reviewed manifest-backed quarantine batch.

Generally safe: reading files, targeted inspected tests that contact nothing, registry print-selection,
and chart export from existing local artifacts. Inspect a test before running it.

## 7. Protected areas

- Market data: `data/`, `raw/`, `cache/`, `vendor/`, `processed/`, and all v4 raw/normalized/feature/label
  trees. Never delete, move, or reorganize them casually.
- History: `v2/`, `v3/`, `v4/`, `archive/`, and `archive_quarantine/`, except for an explicitly reviewed v5
  migration or quarantine manifest.
- Evidence: `v4/artifacts/`, `v4/audit/`, `v4/logs/`, and `v4/runtime/`. Never overwrite another run.
- The historical protected holdout is spent. There is no confirmation firewall left.

## 8. Code quality and compatibility

- Read before editing and preserve unrelated dirty-worktree changes.
- New active research code belongs under `v5/` and must not import `v4`.
- A narrow compatibility wrapper may make legacy v4 callers use a v5 interface; v5 never depends back on
  v4 code.
- Match local style and test in proportion to risk.
- Do not execute a legacy tool merely because it appears in [`TOOLBOX.md`](TOOLBOX.md); its safety class
  controls.

## 9. Legacy boundary

`v4` is frozen evidence and a selectively vetted toolbox. Its Protocol101 paper runtime still exists, but
it is not the current research path. New plans, status updates, and G1 code must not be placed there.
