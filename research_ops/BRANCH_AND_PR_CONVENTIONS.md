# Branch And PR Conventions

Purpose: make each GitHub branch and pull request map cleanly to one
research_ops iteration, assumption, and decision.

These conventions are part of the governance layer. They do not authorize
trading behavior changes, runtime flag changes, launchd changes, broker calls,
paid-data downloads, model training, threshold tuning, model promotion, or
protected holdout exploration.

## Branch Names

Use a department-style prefix plus the iteration ID.

Allowed patterns:

```text
research/ITER-001_quote_age_truth
rfc/ITER-002_decision_reconstruction
diag/ITER-003_feature_parity
docs/ITER-004_stale_docs_index
```

Preferred prefix meanings:

| Prefix | Use when |
|---|---|
| `research/` | Cartography, evidence gathering, read-only research packets, or broader research_ops work. |
| `rfc/` | Designing an experiment or diagnostic before implementation. |
| `diag/` | Implementing or running a diagnostic harness that produces artifacts. |
| `docs/` | Governance documentation, stale-doc banners, dashboard updates, or source-of-truth cleanup. |

Do not use vague branch names:

```text
improve-model
fix-bot
try-new-thing
better-architecture
```

If a branch cannot name the iteration, do not start the branch yet. Create the
iteration packet first.

## PR Titles

PR titles must begin with the iteration number in square brackets and name the
artifact or diagnostic being added.

Use:

```text
[ITER-001] Add quote-age truth diagnostic
[ITER-002] Add decision reconstruction matrix
[ITER-003] Add Protocol051/101 feature parity harness
```

Avoid:

```text
Improve model
Fix bot
Try new thing
Better architecture
```

## Required PR Answers

Every PR must answer:

1. What assumption moved?
2. What decision does this unlock?
3. What is still blocked?

These answers should be short, evidence-bound, and linked to the relevant
iteration artifacts. If the answer is "nothing moved," the PR should say that
explicitly and explain why the artifact is still useful.

## Required Links

Every PR should include:

- Iteration ID
- Assumption ID
- RFC path, if applicable
- Implementation summary path, if applicable
- Verifier report path, if applicable
- Decision memo path, if applicable

## Scope Rules

- One branch should normally map to one iteration.
- One PR should normally move one assumption.
- Documentation-only PRs may update multiple governance files if they are all
  part of the same iteration.
- Any PR touching protected paths in
  `research_ops/DO_NOT_TOUCH_WITHOUT_APPROVAL.md` requires explicit human
  approval and a CEO decision memo.
- Challenger promotion, model training, threshold tuning, runtime flag mutation,
  launchd mutation, broker behavior changes, paid-data downloads, and protected
  holdout scoring are blocked by default.

## Review Standard

Reviewers should ask whether the PR made the project more falsifiable. A good
PR reduces ambiguity, preserves operational safety, and leaves behind artifacts
that a future agent can audit without relying on memory.
