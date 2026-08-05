# Work Packet Rules

All committed plans and active handoffs live under `v5/work/<registered-job>/`. A directory may contain
files only after that exact path appears in the job register in [`v5/STATUS.md`](../STATUS.md).

Each active job uses the smallest useful packet:

- `BRIEF.md` states the question and constraints;
- `PLAN.md` states an approved or held execution plan;
- `RESULT.md` is added only when work produces a durable result.

When a job closes, move its packet to `v5/history/jobs/<job>/` in the same change that records its durable
finding in STATUS, the do-not-retest ledger, or the evidence index. Do not create general handoff,
roadmap, source-of-truth, or status files.
