# Protocol101 Shadow-As-Of Week Plan (2026-07-18)

Owner is away 2026-07-20 → 07-24. Goal: while the recorder collects next
week's sessions unattended, also test equivalence link 3 — do decisions
computed DURING the live session (from data arrived so far) match the
end-of-day replay of the recording? By Friday 07-24 the owner returns to:
collected sessions, a readable live-vs-replay comparison for the dev day
(07-20), sealed shadow evidence banked for post-exam analysis, and the
07-20 rehearsal add-on ready to run.

## Design decision: as-of replay, not event-driven shadow (this week)

True event-driven shadow (a real-time decision daemon) is new live code; it
will be built for the pre-paper phase per the existing preregistration. It
is NOT built this week: shipping untested live automation next to the only
evidence collector, unattended, is the one way this week can go badly.

Instead: **shadow-as-of replay** — every 15 minutes during the session, a
job runs the existing, battle-tested deterministic replay
(`v4/scripts/run_protocol101_fair_contract_ibkr_capture_replay.py` code
path) over the capture-so-far and logs the decisions computed at that
wall-clock moment. Post-session, these as-of snapshots are diffed against
the final end-of-day replay. Any divergence measures exactly the live/replay
seam the owner asked about: late-arriving data changing already-decided
minutes, boundary effects, partial-state differences. Sub-minute scheduling
jitter is not tested (minute-boundary decision convention makes it minor);
that residual belongs to the true shadow phase.

Properties: read-only on the capture file, separate process, cannot write
to the recorder's outputs, no broker interaction, no new decision code —
the same frozen pipeline in both roles.

## Separation rules (so nothing gets mixed up)

1. Shadow outputs live INSIDE each session's capture directory under
   `shadow_asof/` (snapshots `asof_HHMM.decision_log.jsonl` + final
   `shadow_vs_final_diff.json`). Consequence: on sealed days the 13:30
   sealing automation moves them into the vault with the capture —
   sealed-day shadow evidence is preserved but unreadable until the exam,
   with zero new rules.
2. Dev day 07-20: shadow outputs openly inspectable; the post-session
   diff report is generated and readable immediately.
3. Sealed days 07-21..24: the 13:30 automation report may state ONLY
   "shadow_asof ran, N snapshots, sealed" — no decision-level fields
   (same rule as the audit reporting fix).
4. Evidence grade: `shadow_asof_diagnostic`. Not part of the confirmation
   battery; never gates the sealed exam; informs the future event-driven
   shadow preregistration.
5. Sealed-day shadow diffs run only AFTER the sealed exam, on days already
   spent by the battery.

## Success criteria for the 07-20 readable diff (preregistered now)

- As-of decisions for completed minutes match the final replay's decisions
  for those minutes >= 0.995 (action + selected slot), with every
  disagreement listed with its arrival-timing explanation.
- Divergences concentrated in the most recent 1–2 minutes of each as-of
  snapshot (data still arriving) are expected and classified separately
  from divergences in settled minutes (a settled-minute flip is the real
  signal to investigate).

## Codex implementation goal (run this weekend; owner authorization below)

OWNER AUTHORIZATION (2026-07-18): install one additional launchd agent for
the shadow-as-of job (session-time schedule, e.g. every 15 min 06:35–13:00
PT weekdays) + one post-session diff step for dev days. No other launchd,
runtime-flag, broker, promotion, or paid-data changes. gate_mode/recorder
deployment untouched.

```text
GOAL: Protocol101 shadow-as-of runner + weekly automation.

1. New script v4/scripts/run_protocol101_shadow_asof_snapshot.py:
   runs the ibkr_capture_replay code path over the current session's
   capture-so-far (read-only), writes decisions to
   <capture_dir>/shadow_asof/asof_HHMM.decision_log.jsonl with wall-clock
   stamp, sequence high-water mark, and rows-decided count. Must not
   modify capture files; must exit cleanly if capture absent/small.
2. New script v4/scripts/run_protocol101_shadow_asof_diff.py:
   post-session, compares all as-of snapshots vs the final replay traces;
   emits shadow_vs_final_diff.json (per-minute action/slot agreement,
   settled-minute vs trailing-edge classification) + a short report.
   Automation runs the diff ONLY for non-sealed sessions.
3. One launchd agent for the snapshot schedule; diff wired into the
   existing 13:30 automation for dev days with count-only mention for
   sealed days.
4. Dry-run the snapshot script against the 2026-07-14 capture (dev day,
   inspectable) to validate before Monday; commit everything.
DO NOT: touch recorder deployment/gate_mode, sealed root, frozen
artifacts; no broker calls; no paid downloads; no decision-code changes —
reuse the existing replay functions verbatim.
```

## Friday 2026-07-24 return checklist

1. Read the 07-20 shadow_vs_final_diff report (link-3 first evidence).
2. Download vendor data for 07-20 (already owner-authorized), build its
   paired replay, rerun the rehearsal battery including 07-20 —
   completes the rehearsal gate (3/3 dev days).
3. Verify sealed captures 07-21..24 green via manifests; sealing intact.
4. Confirm ThetaData month still active; batch-download any remaining
   window dates.
5. Sealed-count check: with 07-15..17 lost, expected sealed = 8 (< 9
   required) unless the extension through 08-04 was authorized — see
   open decision below.

## Open decision (needs owner answer before departure)

The preregistration requires >= 9 sealed sessions; after last week's
losses the expected count is 8. The owner's stated trigger ("any further
sealed-day failure forces the extension") has effectively already fired.
Staged remedy: extend the recorder allowlist through 2026-08-04 (adds
07-31, 08-03, 08-04; same mechanism as the July extension). Recommended:
authorize now, before the away week.

## Implementation status (2026-07-18 evening — implemented by Fable, owner-directed)

- `v4/scripts/run_protocol101_shadow_asof_snapshot.py` — built; full-file
  dry-run on 2026-07-14: 360 rows in ~60s wall. Truncated (60%) dry-run:
  clean partial-read path.
- `v4/scripts/run_protocol101_shadow_asof_diff.py` — built; 07-14 dry-run:
  settled agreement 1.0, trailing-edge 1.0, passes the 0.995 bar; sealed
  sessions produce count-only markers via the sealing rule's classify().
- launchd agents loaded: `com.autoresearch.protocol101.shadowasof.snapshot`
  (StartInterval 900s, self-gated to weekdays 06:35–13:10 PT) and
  `...shadowasof.diff` (daily 13:20 PT, self-gated). Logs:
  `~/Library/Logs/autoresearch-trading/shadowasof.*.log`.
- Recorder allowlist extended through 2026-08-04 (sealed expected 8 → 11).
- First live snapshots: Monday 2026-07-20. First readable diff: same day
  ~13:20 PT (dev day).

## Scope revision (2026-07-18, owner-directed): collection only during the week

The daily 13:20 diff agent is REMOVED. The week runs collection only:
recorder capture + shadow decision logging. ALL tests/diffs/analyses run
at the end of the week after collection completes. The diff script is kept
on disk for the Friday manual run. Sealed-day snapshot counts are visible
from file listings (no decision content) if needed.

## Daily timeline (Mon 07-20 – Fri 07-24, all times PT) — for Codex check-in automations

| Time | What happens | Automated by |
|---|---|---|
| 05:30 | IB Gateway starts | parityrecorder.gateway |
| 05:40 | Preflight | parityrecorder.preflight |
| 05:45 | Recorder capture begins | parityrecorder.recorder |
| 05:50–06:28 | Health checks + watchdog start | parityrecorder.health/watchdog |
| 06:35–13:05 | Shadow decision log written every 15 min into `<capture_dir>/shadow_asof/` | shadowasof.snapshot |
| 13:05 | Recorder shutdown | parityrecorder.shutdown |
| 13:10 | Capture finalize (manifest + checksums) | parityrecorder.finalize |
| 13:15 | Capture-quality audit | parityrecorder.audit |
| 13:30 | Codex post-session check + sealing assign (sealed days move to vault, shadow logs inside them) | Codex automation |

Suggested Codex check-in points (report-only, no tests):
- ~06:45: recorder running? capture file growing? first shadow snapshot
  present? (file existence/size only)
- ~13:40: quality manifest green? session sealed (if 07-21..24)? shadow
  snapshot count for the day? (counts only, no decision content)
- Escalate loudly on: missing capture, failed quality checks, missing
  sealing, zero shadow snapshots.

## End-of-week test batch (Friday 07-24 after close, owner present)

1. Shadow-vs-final diff for dev day 07-20 (manual:
   `run_protocol101_shadow_asof_diff.py --session 2026-07-20 --manual`).
2. Vendor downloads for 07-20 (pre-authorized), paired replay, rehearsal
   battery rerun including 07-20 → completes rehearsal gate 3/3.
3. Sealed manifests review (07-21..24), sealed-count check.
4. Sealed-day shadow diffs remain deferred until after the sealed exam.
