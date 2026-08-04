# Path-D Wave 1 Exit Repair — Results (2026-08-03)

**Bottom line: `NO_EDGE`.** The eight-member family is exhausted. Do not extend it or rerun these
mechanisms unchanged. The protected holdout remained closed (`holdout_open_count = 0` for this wave).
No paid download, broker connection, live/paper order, promotion/default change, runtime-flag edit, or
scheduling change occurred.

## Evidence and claim boundary

- Frozen preregistration:
  `PATHD_WAVE1_EXIT_REPAIR_PREREGISTRATION_2026_08_03.md`, SHA-256
  `063e9ced30d95bda013e05334a25c13469baa1f0dc2c5d693ecc048730e40f0e`.
- Machine packet:
  `/Volumes/AR_TRADING_DATA/artifacts/pathd_wave1_exit_repair_2026_08_03_retry2/results.json`, file
  SHA-256 `b6f0e273dfb878dbe78b4bbfa715d0e1004cb6861f76fea75ed883d80fa69b1e`.
- Full generated report:
  `/Volumes/AR_TRADING_DATA/artifacts/pathd_wave1_exit_repair_2026_08_03_retry2/report.md`, file SHA-256
  `e893d611bc783d21a1c314d65271d59e140411e29c9f84ee31b739191d9e8570`.
- Semantic report SHA-256:
  `c751d1eebdecb2e9a15aa28a95feb23eabffc172d9fe28ad80794187af2984c9`.
- Gate population: all 1,031 OOF exit trajectories, 166 sessions, five outer folds. The legacy
  learned-entry/control-entry four-box cells are also recorded in every candidate packet.
- Claim boundary: `NO_EDGE` is a successful bounded research outcome. No Stage-2, paper, deployment,
  promotion, or real-money claim follows.

Two preserved failed attempts precede the authoritative `retry2` packet. The first stopped before the
first fit on a read-only NumPy weight view. The second was stopped after H03 when review showed that the
legacy 153-row learned-entry headline subset spans only three folds and conflicts with the frozen
five-fold exit gate. The correction changed no hypothesis, parameter, label, fill law, causal clock, or
OOF split: it scored the exit repair over the handoff's stated 1,031-trajectory exit population and kept
the four-box cells as diagnostics.

## Prior-art ruling

The executable search covered both canonical sources:

- `history/PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md`, especially §4.
- `history/PROTOCOL101_PROTOCOL_FARM_LINEAGE_2026_07_19.md`, especially Protocol065 at line 133.

`recovery penalty` had one non-blocking lineage hit: Protocol065 passed and Protocol066 validated the
mechanism. `rebalanced exit label` and `risk lower bound calibration` had no canonical hit. None of the
eight hypotheses had a blocking hit. The family did not reopen H1 relaxed suffix-max, H2 generic regret
weighting, H3e profitable-only targets, H3f regret regression, side-blind L3, or Protocol029/030/031
generic early-exit rules.

## Family results

H01/H02 failed the required pre-fit balance check at an aggregate 19.21% positive labels. They were not
trained; their rejection tests, controls, and charter diagnostics are correctly `NOT_EVALUATED` except
for `label_balance`. H03-H08 used the frozen deterministic 50/50 sample.

| ID | Result | Policy PnL | Best comparator | Delta | Positive folds | Bootstrap LCB | maxT p |
|---|---|---:|---:|---:|---:|---:|---:|
| H01 | `NO_EDGE` / preflight reject | N/E | N/E | N/E | N/E | N/E | N/E |
| H02 | `NO_EDGE` / preflight reject | N/E | N/E | N/E | N/E | N/E | N/E |
| H03 | `NO_EDGE` | -$54,138 | -$49,508 (`matched_random_4`) | -$4,630 | 2/5 | -$88.65 | 0.9913 |
| H04 | `NO_EDGE` | -$56,418 | -$41,958 (`matched_random_6`) | -$14,460 | 3/5 | -$270.47 | 0.9939 |
| H05 | `NO_EDGE` | -$54,088 | -$49,618 (`matched_random_4`) | -$4,470 | 2/5 | -$88.07 | 0.9903 |
| H06 | `NO_EDGE` | -$54,193 | -$52,868 (`matched_random_6`) | -$1,325 | 1/5 | -$43.86 | 0.9676 |
| H07 | `NO_EDGE` | -$53,648 | -$52,298 (`matched_random_6`) | -$1,350 | 2/5 | -$45.06 | 0.9672 |
| H08 | `NO_EDGE` | -$55,203 | -$48,258 (`matched_random_7`) | -$6,945 | 1/5 | -$119.73 | 0.9941 |

Per-fold paired deltas:

| ID | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|---:|
| H03 | -$3,345 | +$2,625 | -$4,600 | +$1,295 | -$605 |
| H04 | -$18,360 | -$500 | +$425 | +$1,490 | +$2,485 |
| H05 | -$2,620 | +$1,085 | -$4,660 | +$1,865 | -$140 |
| H06 | -$1,050 | -$35 | -$340 | -$10 | +$110 |
| H07 | +$140 | -$740 | -$750 | -$120 | +$120 |
| H08 | -$1,945 | -$4,720 | -$770 | +$1,710 | -$1,220 |

No trained member beat its comparator, no bootstrap LCB was positive, no member had four positive
folds, and no member survived the 9,999-draw session-blocked maxT family correction (seed 1065).

## Negative controls

All three required controls were run for every trained candidate, with the catastrophic floor and each
candidate's fallback arm unchanged. Every control remained below that candidate's best comparator, so
none was accepted and the family is not `INVALID`.

| ID | Sign-reversed PnL | Session-shuffled PnL | Constant PnL | Any accepted? |
|---|---:|---:|---:|---:|
| H03 | -$155,443 | -$58,333 | -$55,518 | no |
| H04 | -$131,868 | -$65,003 | -$55,518 | no |
| H05 | -$156,293 | -$58,128 | -$55,518 | no |
| H06 | -$71,528 | -$56,543 | -$55,518 | no |
| H07 | -$72,508 | -$57,553 | -$55,518 | no |
| H08 | -$61,088 | -$56,208 | -$55,518 | no |

H01/H02 controls are `NOT_EVALUATED_PREFLIGHT`: no model exists to transform after the required
pre-fit balance rejection.

## Rejection tests and acceptance components

`LB` is label balance; `FS` first-step exit; `TP` tail preservation; `DM` decile monotonicity; `BL` the
charter big-loss-share gate. `pass` below means the fixed gate passed, not that the candidate was good.

| ID | LB | FS rate (≤50%) | TP p99 / top-decile (≥80% each) | DM best decile | BL share (≤2%) |
|---|---:|---:|---:|---:|---:|
| H01 | fail, 19.21% | N/E | N/E | N/E | N/E |
| H02 | fail, 19.21% | N/E | N/E | N/E | N/E |
| H03 | pass, 50% | fail, 91.85% | fail, 8.12% / 3.00% | fail, 5 | fail, 3.39% |
| H04 | pass, 50% | fail, 82.74% | fail, 14.64% / 7.26% | fail, 5 | fail, 8.63% |
| H05 | pass, 50% | fail, 91.37% | fail, 8.80% / 2.76% | fail, 0 | fail, 3.30% |
| H06 | pass, 50% | fail, 91.85% | fail, 7.42% / 1.99% | fail, 5 | fail, 2.52% |
| H07 | pass, 50% | fail, 91.37% | fail, 5.46% / 2.01% | fail, 0 | fail, 2.42% |
| H08 | pass, 50% | fail, 76.82% | fail, 12.19% / 6.24% | fail, 0 | fail, 7.18% |

Every trained candidate therefore failed `beats_comparator_pooled`, `bootstrap_lcb > 0`,
`positive_fold_deltas >= 4`, `rejection_tests_pass`, and `maxt_survived`. All passed
`negative_controls_clean` and the separate session/weekday concentration gate. The latter is not a PnL
concentration objective.

## Charter diagnostics

Four-bucket shares (`big win / scratch / small loss / big loss`):

| ID | Big win | Scratch | Small loss | Big loss |
|---|---:|---:|---:|---:|
| H03 | 2.13% | 20.37% | 74.10% | 3.39% |
| H04 | 3.88% | 18.62% | 68.87% | 8.63% |
| H05 | 2.33% | 19.98% | 74.39% | 3.30% |
| H06 | 2.23% | 20.37% | 74.88% | 2.52% |
| H07 | 2.52% | 19.98% | 75.07% | 2.42% |
| H08 | 4.95% | 16.00% | 71.87% | 7.18% |

The remaining three mandated diagnostics are report-only:

| ID | Aggregate harvest ratio | Longest underwater steps | Max drawdown | Top-5 / top-20 positive-PnL share |
|---|---:|---:|---:|---:|
| H03 | -5.10% | 1,025 | $54,060 | 47.72% / 97.39% |
| H04 | -5.05% | 1,030 | $56,155 | 38.74% / 82.81% |
| H05 | -5.10% | 1,025 | $53,890 | 47.28% / 95.97% |
| H06 | -5.11% | 1,027 | $54,181 | 41.73% / 95.69% |
| H07 | -5.05% | 1,027 | $53,636 | 43.23% / 93.04% |
| H08 | -4.88% | 1,027 | $55,191 | 26.11% / 71.29% |

No one session or weekday supplied more than 50% of positive paired delta in any trained member. Win
rate and PnL concentration were not optimized.

## Durable conclusion

The fixed 50/50 sampler repairs the pre-fit balance metric, and the fallback arms reduce the big-loss
share close to the 2% boundary, but none of the changes repairs the exit head. The best complete known
repair (H07) still exits immediately 91.37% of the time, preserves only 5.46% of hold-to-close p99 and
2.01% of top-decile contribution, has a non-monotone score, trails its comparator by $1,350, and has a
negative bootstrap LCB. Reduced LCB weight (H08) lowers first-step exits to 76.82% but worsens pooled PnL
and big-loss share. These exact mechanisms are closed for this Phase-1 game.

*Signed: Codex — 2026-08-03 — `NO_EDGE`; STOP_FOR_CLAUDE_VERIFICATION.*
