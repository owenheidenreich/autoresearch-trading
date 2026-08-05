# V5 Toolbox

This is the directory to useful machinery. An entry here means the path was inspected and classified; it
does not authorize running it. “Owner-gated” means the owner must explicitly authorize the action in the
current conversation.

| Purpose | Canonical path | Status | Safety | Evidence |
|---|---|---|---|---|
| Current project consistency | [`v5/ops/check_project.py`](ops/check_project.py) | Current v5 | Safe, read-only | [`v5/tests/test_project_structure.py`](tests/test_project_structure.py) |
| Reviewed v4 documentation quarantine | [`v5/ops/quarantine_v4_docs.py`](ops/quarantine_v4_docs.py) | Current v5; dry-run by default | `--apply` is owner-gated and must remain manifest-backed | Quarantine manifest and post-move checks |
| Block repeated failed ideas | [`v5/research/prior_art.py`](research/prior_art.py) | Current v5 | Safe when used read-only | [`v5/tests/test_prior_art.py`](tests/test_prior_art.py) |
| Closed-result ledger | [`v5/research/history/DO_NOT_RETEST.md`](research/history/DO_NOT_RETEST.md) | Current v5 | Read-only except when recording a completed finding | Executable prior-art tests |
| Gate-chain and power audit | [`v5/research/findings/GATE_CHAIN_AUDIT_2026_08_05.md`](research/findings/GATE_CHAIN_AUDIT_2026_08_05.md) | Current evidence | Safe, read-only | Independent arithmetic checks recorded in the finding |
| Paired session statistics | [`v4/research/autoresearch_v2/statistics.py`](../v4/research/autoresearch_v2/statistics.py) | Vetted legacy dependency; promote before new v5 use | Safe local calculation | Existing Path-D tests |
| Prior SPX direction screen | [`v4/research/pathd_spx_directional_skill_screen.py`](../v4/research/pathd_spx_directional_skill_screen.py) | Closed diagnostic, not the held G1 design | Do not rerun unchanged | Ledger and gate-chain audit |
| Matched-surrogate diagnostic | [`v4/research/pathd_spx_screen_surrogate_diagnostic.py`](../v4/research/pathd_spx_screen_surrogate_diagnostic.py) | Legacy reference for the G1 null | Inspect only until the measurement review | Gate-chain audit §3 |
| G5 validation replay | [`v4/research/pathd_phase1_replay.py`](../v4/research/pathd_phase1_replay.py) | Defective; four repairs required | Do not trust a pass | [STATUS.md §7](STATUS.md#7-g5-validation-is-defective) |
| Track-A arrival capture | [`v4/ops/tracka/`](../v4/ops/tracka/) | Frozen and unloaded | Owner-gated vendor contact and unattended execution | [evidence index](evidence/INDEX.md) |
| Legacy paper runtime | [`v4/live/`](../v4/live/) and [`v4/scripts/`](../v4/scripts/) | Frozen legacy capability | Owner-gated; can contact IBKR or submit paper orders | [evidence index](evidence/INDEX.md) |
| Paid-data tools | [`v4/ingest/`](../v4/ingest/) and download scripts under [`v4/scripts/`](../v4/scripts/) | Frozen legacy capability | Owner-gated; may spend money | Paid-data guard and audit receipts |

New G1 implementation belongs under [`v5/research/direction/`](research/direction/), but that directory
must remain implementation-free until the measurement review moves its registered job out of HOLD.
