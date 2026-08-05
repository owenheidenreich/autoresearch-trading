# Ground Rules

**Read [STATUS.md](STATUS.md) first.** It is the only status document and it says where the project
actually is. This file says how to work here. `AGENTS.md` is identical by design.

---

## 1. What we are building

An automated day-trading bot that buys SPX 0DTE calls and puts, decided by a machine-learning model.

**The one open question is whether we can predict SPX direction over 15–60 minutes well enough to clear
measured costs (0.358 ES points / $17.92 round trip).** Until that is answered, model work on the option
layer cannot pay off — see STATUS.md §3. Every proposal must state how it serves that question.

## 2. Write for the owner, not for other agents

The owner is the decision-maker and is not a full-time quant. Documentation sprawl and unexplained jargon
have repeatedly caused decisions to be signed without being understood — including a programme shutdown
signed because "stood down" was not understood to mean "stopped."

- **Plain English. Define any term of art the first time it appears in a document.**
- Lead with what changed and what it means for the bot, not with methodology.
- If a finding changes what can be built, say so in one sentence at the top.
- No unexplained protocol numbers, codenames, or status tokens in owner-facing text.
- Say the honest thing directly. "This does not work, and here is why" is a successful outcome here.

## 3. Signature tiers — do not ask for approval you do not need

Constant approval requests are as harmful as unauthorized action: they stall the project and train the
owner to sign without reading.

| Tier | What it covers | Who decides |
|---|---|---|
| **1 — Owner, always** | Spends money · contacts a broker or paid vendor · touches real money · deletes or moves data · installs, changes, or removes anything that runs unattended (launchd, cron) · changes the paper default or promotes a model · opens the protected holdout | **Owner.** Ask, wait, do not proceed. |
| **2 — Agent decides, logs it, owner reviews in batch** | Research method and null choice · document structure and supersession · narrowing an already-authorized scope · which tests to run · how to repair a broken run | **Agent.** Decide, record the reason in one line, report at the next checkpoint. |
| **3 — Agent just does it** | Writing code, tests, and analysis · reading files · local validation that contacts nothing | **Agent.** No announcement needed. |

**Tier 2 is the default for anything the owner would answer with "use your judgment."** Narrowing a
declared data capture from three sessions to two is tier 2. Connecting to the vendor at all is tier 1.

When a tier-1 decision goes to the owner, state in two lines: what it costs, what breaks if it is wrong,
and what you recommend. Never present a tier-1 choice without a recommendation.

## 4. Documentation rules

Sprawl is the standing failure mode. Four competing roadmaps existed simultaneously in August 2026.

- **One status page: [STATUS.md](STATUS.md).** Update it when a gate moves. Never create a second one.
- **No new roadmap or plan document without marking an existing one superseded** in the same change.
- **Every research finding goes into the do-not-retest ledger**
  (`v4/docs/protocol101/training/history/PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md`) — what was tested,
  the number, and what would count as genuinely new. This is the most valuable document in the repo.
- **Check the ledger before proposing anything.** Re-running a closed experiment is the most common waste.
- A dated document is evidence of what was true on that date. When superseded, add a banner pointing at
  what replaced it — do not silently rewrite it, and do not delete it.
- Prefer verifying something cheaply over writing a document about it.

## 5. Evidence standard

- Ground every claim in a file, receipt, test result, log, or measurement. Cite the path.
- If evidence is missing, write `UNKNOWN`. If documents conflict, report the conflict — do not resolve it
  silently.
- **Runtime evidence beats documentation.** What is actually loaded, running, and logged wins over what a
  document says is loaded, running, and logged.
- Correct your own errors in place and plainly. Several important findings here came from an agent
  catching its own mistake.
- A large or surprising result is a suspected bug until proven otherwise. Do not optimize a score; find
  the mechanism.
- Every feature must be computable at the moment of the decision, from data that has actually arrived.
  This is not a formality — a 60-second look-ahead cost this project its protected holdout.

## 6. Hard safety rules (tier 1 — no exceptions)

Do not run anything that can trade, contact a broker, download paid data, mutate runtime state, install or
remove scheduled jobs, train a model, tune a threshold, or promote a candidate — without explicit owner
authorization in the current conversation and a fresh read of the relevant safety file.

This includes: IBKR/broker/order/live/paper-submit scripts · Databento or Polygon downloads and backfills ·
model training and threshold tuning · runtime flag edits · `launchctl bootstrap`/`bootout`/`enable`/
`disable` and plist changes · cleanup outside a reviewed, manifest-backed quarantine batch.

Generally safe: reading files, targeted local tests that contact nothing, registry print-selection, and
chart export from existing local artifacts. Inspect a test for broker, paid-data, training, and
runtime-mutation behavior before running it.

## 7. Protected areas

- `data/`, `raw/`, `cache/`, `vendor/`, `processed/`, `v4/raw/`, `v4/normalized/`, `v4/feature/`,
  `v4/label/` — market data. Never delete or reorganize casually.
- `v2/`, `v3/`, `archive/`, `archive_quarantine/` — history. Protected unless there is current import,
  runtime, test, log, or owner evidence that a specific file is active.
- `v4/artifacts/`, `v4/audit/`, `v4/logs/`, `v4/runtime/` — generated evidence. Inspect before trusting;
  never overwrite another run's receipts.
- The protected holdout is **SPENT** (opened once, 2026-08-02). There is no confirmation firewall left.
  Fresh live paper is the only remaining out-of-sample test.

## 8. Code quality

- Read a file before editing it. This worktree is dirty; preserve unrelated changes.
- Match the surrounding code's style, naming, and comment density.
- Tests proportional to the risk of the change.
- The causal clock, fill law, label law, and out-of-fold firewall are **frozen**. If a document and the
  code disagree about a gate, fix the document — never loosen the gate to match it.

## 9. Working with Codex

Claude writes, Codex reviews and attacks the plan — and the reverse where useful. Adversarial review has
repeatedly caught real errors in both directions. When a review refutes a claim, retract it in writing
where the claim was made.

## 10. Legacy reference

The former 559-line pipeline walkthrough (paper spine, promotion process, per-step commands, as of
2026-07-26) is preserved at
[`docs/history/AGENTS_LEGACY_PIPELINE_REFERENCE_2026_07_26.md`](docs/history/AGENTS_LEGACY_PIPELINE_REFERENCE_2026_07_26.md).
It describes the legacy Protocol101 paper runtime, which still exists and is unchanged, but it is **not**
the current research path and its status claims are stale.
