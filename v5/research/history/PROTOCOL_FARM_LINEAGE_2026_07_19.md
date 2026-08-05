# Protocol101 Protocol-Farm Lineage From Codex Transcripts

**Prepared:** 2026-07-19

**Scope:** creation lineage for the Protocol101 farm, especially the Git gap from 2026-04-28 through 2026-05-13

**Mode:** transcript recovery only; no model training, downloads, broker/runtime changes, sealed-vault reads, or frozen-artifact changes

## Executive finding

The missing Git lineage is not lost conversationally. One long Codex session opened on April 28 and continued through May 24. It contains the contemporaneous birth sequence, hypotheses, commands, results, reversals, and ledger patches for Protocol002 through Protocol160, then continues into later research. It explains the previously opaque ranges 001-050, 055-065, and 082-100.

This does **not** restore missing Git commits. The repository still has the direct Apr-27 to May-14 parent edge described in [the prior distillation](DO_NOT_RETEST.md). What is recovered is a new evidence tier:

| Tier | Meaning | Strength |
|---|---|---|
| Committed source/artifact | A hashable repository object or surviving audit output | Strongest for what code/artifact existed |
| **Contemporaneous transcript** | The live research conversation, tool calls, output, and decisions while the farm was being built | Stronger than a later retrospective ledger for birth order and intent; weaker than a committed artifact for exact executable state |
| Retrospective ledger | Later consolidated narrative in `v4/ledger/RESEARCH_LEDGER.md` | Useful corroboration; weaker for creation provenance |

The farm did not begin with a Protocol001 implementation. The first exact numbered creation found is **Protocol002**. `001` appears to be an unused/reserved number, not a missing model. A side branch was named `003C`, and several later apparent gaps are audits, persistence attempts, or superseded infrastructure rather than unrecorded trading models.

## Transcript inventory and backup

Citation shorthand below uses `S1:Lx-Ly`, where `S1` is:

`/Users/gduby/.codex/sessions/2026/04/28/rollout-2026-04-28T14-39-19-019dd608-46dc-7463-97fb-d0a0f17e0ee9.jsonl`

| Session file | Timestamp span | Protocol relevance | Disposition |
|---|---|---|---|
| `S1` | 2026-04-28T21:39:42Z to 2026-05-24T03:22:12Z; 58,057 JSONL lines; 179,174,140 bytes | **Primary farm session.** Exact births from Protocol002 onward, including 051/054/066/081/101 and runtime work through 160; later continues past 160. | Copied to local history backup and hashed. |
| `~/.codex/sessions/2026/05/11/rollout-2026-05-11T20-24-47-019e1a37-39d2-7a10-9787-c56a79558744.jsonl` | 2026-05-12T03:26:00Z to 03:33:45Z; 242 lines | No protocol identifiers; unrelated. | Not copied. |
| `~/.codex/sessions/2026/05/14/rollout-2026-05-14T14-00-03-019e284a-111e-7fe3-b48e-aae87ba2cc1c.jsonl` | 2026-05-14T21:00:11Z to 2026-05-17T04:20:49Z; 34,735 lines | No numbered farm lineage; unrelated despite date overlap. | Not copied. |
| `~/.codex/archived_sessions/rollout-2026-05-10T11-20-23-019e131e-7297-7011-b1b5-32b2cbb03c06.jsonl` | 2026-05-10T18:20:56Z to 18:21:10Z; 18 lines | No protocol identifiers. | Not copied. |
| `~/.codex/archived_sessions/rollout-2026-05-24T15-22-11-019e5c14-db69-7002-b00e-1f46e252eeba.jsonl` | 2026-05-24T22:22:12Z to 2026-05-25T02:43:03Z; 3,202 lines | Later audit/handoff context referring back to many protocols; not their creation session. | Not copied. |
| Apr-8/Apr-17 and Jul-8/Jul-9 archives | Outside requested creation window | Either no protocol farm or later copied/handoff context. | Not copied. |

No named session files existed under `2026/04/26`, `04/27`, `04/29`, `05/12`, or `05/13`. The long `04/28` session spans those dates, which explains the apparently empty daily directories.

Backup:

- Local untracked copy: `_history_backup/codex_sessions/rollout-2026-04-28T14-39-19-019dd608-46dc-7463-97fb-d0a0f17e0ee9.jsonl`
- Original SHA256: `0e8db86dacd85fb13986a3c81206de686d014121bfb57a5002196a6267c01b00`
- Copy SHA256: `0e8db86dacd85fb13986a3c81206de686d014121bfb57a5002196a6267c01b00`
- `_history_backup/` is ignored by Git. The transcript is not committed.

## Primary session and direction changes

The session began as a v4 feasibility review, not as an instruction to generate 160 protocols: “review all files and all code changes created in v4 ... [and] a more economically feasible approach” (`S1:L6-L12`). The numbered farm emerged as successive owner-directed autoresearch loops.

1. **Generalization screen (002).** The first numbered harness preregistered bounded variants and frozen March/Q4 audits. Its conclusion was “no broad data-purchase signal yet,” with zero passing combinations (`S1:L5279-L5817`).
2. **A+ pattern and contract economics (003-028).** The working idea changed from generic surface features to “surface + structure + A+ contract value + pattern/value” (`S1:L6659-L7468`). Protocol007 briefly earned a narrow fresh-data audit, then failed fresh Q3 mostly by not trading (`S1:L8254-L8871`). The one-change loop then exposed fixed-threshold, ensemble, loss-mask, teacher, edge, and activity-cap effects (`S1:L8881-L9508`). Protocol024 became the robust branch; 025-027 failed the three-strike loop (`S1:L12368-L12975`).
3. **Entry/exit separation and broader walk-forward (029-054).** Three generic early-exit overlays destroyed convex winners (`S1:L14328-L14581`). The loop returned to entry-side contract quality, learned that hard quality vetoes rejected `+14,140` of March trades, and moved to broader expanding walk-forward (`S1:L14992-L16039`). Official context and intra-minute same-side ranking produced Protocol051 (`S1:L18131-L20090`), then the first full same-contract lifecycle candidate 054 (`S1:L20761-L20806`).
4. **Sequence lifecycle and reproducibility (055-087).** Failure attribution showed early residual exits were vulnerable to recovery. Protocol065 added a recovery penalty; 066 validated it over ten seeds; 068-075 froze/reproduced the modular stack. Q4-2024 prehistory then produced deterministic Protocol081 and 082 reproduction (`S1:L20986-L25964`).
5. **Strict one-account entry arbitration (088-101).** Shadow/serial infrastructure exposed overlap. Protocol092 modeled slot opportunity cost; 094-096 failed to solve Q4; 097 introduced explicit wait/take; 100 showed a seed-3 history failure; 101 added causal recent-event memory and cleared the registered serial gate by only `+$140` in Q4 (`S1:L26266-L28204`).
6. **Falsification, parity, risk, and runtime (102-160).** The direction stopped being “find more backtest PnL” and became freeze, external stress, high-resolution replay, live-schema parity, guarded paper risk, logging, and operations. Protocol114 called the curve `fragile_needs_more_data` because delay stress was severe (`S1:L30572-L30851`); Protocol117 achieved full high-resolution replay coverage (`S1:L31758-L32041`); Protocol126-130 hardened timing/schema/risk (`S1:L34074-L34451`); Protocol155 required live timing evidence (`S1:L38667-L38900`); and Protocol160 replaced the pulse runner with a persistent connection loop (`S1:L42613-L42740`).

## Granular decoder: Protocol001-054

“Verdict at birth” is the contemporaneous conclusion, not current promotion status. Dollars are replay PnL unless stated otherwise.

| P | Born | Hypothesis / role | Verdict and contemporaneous number | Evidence |
|---:|---|---|---|---|
| 001 | Not found | No exact Protocol001 creation, audit directory, or decision was found. | **Unused/reserved**, not evidence of a lost model. | Session starts at review (`S1:L6-L12`); first named harness is 002 (`S1:L5279`). |
| 002 | Apr 30 | Six surface/structure variants x 3 exit policies x 6 fixed trials; frozen March/Q4. | **Reject.** 0/108 combos passed; best: selection `$100`, PF `1.215`; March `-$2,385`, PF `0.338`; Q4 `$2,966`, PF `1.594`. | `S1:L5279-L5817`; `v4/audit/autoresearch/v4_generalization_protocol_002/report.md`. |
| 003 | Apr 30 | Add causal A+ timing-pattern and contract-value features. `003C` separately added learned entry permission. | **Keep narrow clue.** P003 March `$2,396`, PF `1.217`; Q4 `$7,291`, PF `1.377`. 003C March `$1,724`, PF `1.368`; Q4 `$370`, PF `1.193`; friction fragile. | `S1:L6659-L7896`. |
| 004 | Apr 30/May 1 | Seven permission objectives under the broad-purchase stress gate. | **Reject.** Best `bce_cost50`: March `$1,607`, PF `1.197`, +50 `-$1,593`; Q4 `$3,794`, PF `1.230`, +50 `$144`. | `S1:L7901-L7965`; `v4/audit/autoresearch/v4_aplus_permission_protocol_004/report.md`. |
| 005 | May 1 | Select the permission threshold as if each trade already paid +$50. | **Reject/no change.** It selected the same behavior; March still failed +50. | `S1:L7976-L8056`; `v4/audit/autoresearch/v4_aplus_permission_protocol_005_stress50_select/report.md`. |
| 006 | May 1 | Deterministic Greek/value veto chosen by raw selection +50 score. | **Reject locked winner.** `bce_cost50 + theta<=0.25`; March `$1,526`, PF `1.291`, +50 `$926`; Q4 `$3,544`, PF `1.227`, +50 only `$144`, PF `1.008`. | `S1:L8098-L8186`; ledger patch `S1:L8217`. |
| 007 | May 1 | Choose veto by domain priority: permission plus gamma/theta economics. | **Provisional pass, then fresh-audit fail.** Initial March `$2,816`, PF `1.440`; Q4 `$1,222`, PF `1.771`. Fresh Q3 median PnL/trades `0/0`, positive seed fraction `.33`. | `S1:L8254-L8871`. |
| 008 | May 1 | Locked fresh-Q3 audit/data block for 007. | **Reject 007 / stop buying.** Q3 clean, but two seeds did not trade. | `S1:L8828-L8871`; `v4/audit/autoresearch/v4_aplus_value_veto_protocol_008_q3_fresh_locked/`. |
| 009 | May 1 | Fixed absolute permission thresholds to restore Q3 coverage. | **Reject.** Best Q3 `0` PnL on 1 trade; March +50 `-$887`. | `S1:L8881-L8933`; `v4/audit/autoresearch/v4_aplus_hypothesis_009_fixed_threshold_q3/report.md`. |
| 010 | May 1 | Remove permission layer; use base A+ surface plus gamma/theta veto. | **Reject.** Q3 median `0`, PF `0`, 1 trade; March +50 `-$1,913`. | `S1:L8933-L8943`; H010 report. |
| 011 | May 1 | Average the three base surface seeds before selection. | **Reject.** Q3 took 0 trades; March `-$194`, PF `0.971`, +50 `-$1,544`. | `S1:L8943-L8953`; H011 report. |
| 012 | May 1 | Correct masked-Huber math, then rerun old A+ model. | **Bug fix kept; model rejected.** Selection `-$59`, PF `0.638`; March/Q3 0 trades. | `S1:L9153-L9189`; H012 report. |
| 013 | May 1 | Auxiliary BCE teacher head for the transparent A+ rule. | **Reject.** Zero trades everywhere: auxiliary classifier did not move the utility head. | `S1:L9121-L9212`; H013 report. |
| 014 | May 1 | Direct teacher-margin loss on profitable causal A+ tokens. | **Keep.** Locked-seed Q3 `$14,350`, PF `1.239`; real Q4 `$40,542`, PF `1.642`; March `$18,738`, PF `1.982`. | `S1:L9275-L9350`; H014 locked-seed reports. |
| 015 | May 1 | Raise entry edge from 25 to 50 for friction cushion. | **Reject.** March collapsed to 0 trades; selection floor failed. | `S1:L9379-L9408`; H015 report. |
| 016 | May 1 | Keep edge25 but cap at 2 trades/day. | **Keep conservative expression.** Q3 `$11,394`, PF `1.339`; Q4 `$20,432`, PF `1.582`; March `$3,962`, PF `1.320`. | `S1:L9415-L9456`; H016 Q3/Q4 reports. |
| 017 | May 1 | Cap at 1 trade/day. | **Reject.** Q3 friction robust, but selection had only 10 trades and Q4 +100 failed two seeds; Q4 `$5,312`, PF `1.280`. | `S1:L9465-L9496`; H017 reports. |
| 018 | May 1 | Formalize teacher-margin + policy1 + `post_open_late_edge25_max2` as the economic candidate. | **Conditional data-purchase signal.** March `$3,962`, PF `1.320`; Q2 `$13,492`, PF `1.462`; Q3 `$11,394`, PF `1.339`; Q4 `$20,432`, PF `1.582`. | `S1:L9508-L10357`; ledger patch `S1:L9537`. |
| 019 | May 1 | Audit frozen 018 on the partial Q1 block. | **Fail partial block.** `-$7,802`, PF `0.593`, 46 trades. No retuning permitted. | `S1:L10626-L10727`. |
| 020 | May 1 | Complete Q1 and rerun frozen 018 plus existing 1s slices. | **Thin recovery.** Full Q1 `$2,190`, PF `1.057`, 117 trades, but +50 `-$3,810`; 25/119 March trades, 0 sign flips. | `S1:L11502-L11674`. |
| 021 | May 1 | Locate Q1 fragility by side/time/contract economics. | **Diagnostic.** Q1 post-open puts `-$2,790`, PF `0.880`; calls `$4,124`, PF `1.275`. Do not hard-filter puts. | `S1:L11947-L12017`. |
| 022 | May 3 | Learned side-aware proposal-level quality gate. | **Reject.** Q1 `$9,812`, PF `1.383`, but Q4 +50 `-$344`. | `S1:L12041-L12368`; ledger patch `S1:L12218`. |
| 023 | May 3 | Direct in-network put-quality margin. | **Reject over-rotation.** Q1 puts improved, but Q4 fell to `$2,754`, PF `1.073`; March only 14 trades. | `S1:L12192-L12368`. |
| 024 | May 3 | Softer side-weighted value multitask objective. | **Keep conditional branch.** March `$9,854`, PF `2.847`; Q1/Q2/Q3/Q4 PF `1.311/1.271/1.551/1.308`; +50 positive everywhere. | `S1:L12240-L12472`. |
| 025 | May 3 | Lighter balanced value loss to recover Q4 without losing Q1. | **Reject.** Q4 recovered, Q1 +50 stress failed. | `S1:L12585-L12731`. |
| 026 | May 3 | Keep 024 loss; add explicit Greek/timing interactions. | **Reject.** Q1 cleaned even under +100, but Q4 fell below 024. | `S1:L12781-L12873`. |
| 027 | May 3 | Blend 025 loss with 026 interactions. | **Reject / loop stop.** Improved March/Q1/Q4, but Q2 +50 broke and only 2/3 seeds were positive. | `S1:L12883-L12975`; ledger patch `S1:L12954`. |
| 028 | May 10 | Replace proxy VWAP wholesale with stitched ES futures VWAP. | **Reject direct replacement.** March `-$80`, PF `.928`; Q2 `-$2,130`, PF `.552`; Q4 `-$2,790`, PF `.890`. | `S1:L13979-L14031`; P028 report. |
| 029 | May 10 | Generic causal position-state/headroom exit on frozen 024 entries. | **Reject.** Q1 improved to `$16,370`, PF `3.009`, but Q4 fell to `$3,490`, PF `1.163`, +50 `-$2,660`. | `S1:L14279-L14328`. |
| 030 | May 10 | Permit early exit only on loss or meaningful giveback. | **Reject.** Q2 `-$4,090`, PF `.768`; Q4 `-$10,570`, PF `.623`. | `S1:L14498-L14581`. |
| 031 | May 10 | Loss-only damage-control exit. | **Reject / exit-loop stop.** Q1 `-$4,850`, Q2 `-$6,710`, Q4 `-$11,520`; convex winners often went underwater first. | `S1:L14544-L14581`. |
| 032 | May 10 | Ladder-relative quality features inside entry model. | **Reject replacement; keep signal.** Q1 `$13,700`, PF `2.333`; March only 5 trades, Q2 below 024. | `S1:L14846-L14992`. |
| 033 | May 10 | Learned quality gate on frozen 024 proposals. | **Reject.** Helped Q2/Q3/Q4 but damaged March/Q1. | `S1:L14857-L14992`; grouped ledger patch `S1:L14963`. |
| 034 | May 10 | Require quality gate to retain at least 80% of February trades. | **Best challenger, not replacement.** Beat 024 in Q1/Q2/Q3/Q4 but March `$7,040 vs $11,300`. Attribution: removed 18 trades worth `+$14,140`. | `S1:L14911-L15111`. |
| 035 | May 10 | Require 100% February retention. | **Reject.** Recovered some March, gave back Q1, still did not beat 024 in March. | `S1:L14911-L14992`; P033-036 ledger entry. |
| 036 | May 10 | Re-rank near-tied same-side contracts by static economics. | **Reject/no-op.** Selection chose baseline, so simple reranking added no lift. | `S1:L14925-L14992`. |
| 037 | May 10/11 | Freeze minute/side/count; learn only same-side strike choice. | **Reject/no-op.** `baseline_original_contract` selected; exact 024 replay. | `S1:L15121-L15194`. |
| 038 | May 11 | Soft quality objective inside expanding walk-forward. | **Reject soft objective; keep methodology.** Candidate Q2/Q3/Q4/Q1 `-$480/$0/$0/$1,460` vs baseline `$11,330/$12,660/$19,020/$31,380`. | `S1:L15203-L15760`. |
| 039 | May 11 | No-new-knob broader Protocol024-family expanding baseline. | **Freeze research baseline.** 10-seed Q2/Q3/Q4/Q1 PnL `$9,100/$13,595/$18,230/$27,095`, PF `1.341/1.414/1.577/1.765`. | `S1:L15760-L16039`. |
| 040 | May 11 | Promotion-readiness checks on 039. | **Blocked.** Q1 1s replay 173/1,125 and PnL `$470 -> -$2,700`; official SPX/VIX fraction `0.000`; March slice encouraging. | `S1:L16123-L16142`. |
| 041 | May 11 | Capped official-context/targeted-1s request, then approved acquisition/build evidence. | **Infrastructure pass, not model.** 27 targeted sessions/198 pairs/4,603,814 rows; estimated `$0.685970`; later SPX/VIX file coverage complete. | `S1:L16214-L17004`. |
| 042 | May 11 | Rerun frozen 039 with official context and available 1s promotion checks. | **Supportive but blocked.** 417/4,064 trades; `$114,430 -> $108,870`, 0 sign flips; only 10.26% coverage, none in Q3. | `S1:L18096-L18640`. |
| 043 | May 11 | Diagnose official-context post-open put suppression. | **Diagnostic.** Missed winners tracked pattern/liquidity interactions, not one static Greek/value threshold. | `S1:L18609-L18640`. |
| 044 | May 11 | Soft global put-pattern recall margin. | **Reject.** Beat 039 in 2/4 folds; March delta `-$4,020`. | `S1:L18609-L18640`. |
| 045 | May 11 | Add joint pattern/value/Greek/structure interaction tokens. | **Pass 3-seed screen.** Beat baseline in 3/4 folds; March `+$6,660`. | `S1:L19039`; P045 ledger entry. |
| 046 | May 11 | Ten-seed validation of 045. | **Validated challenger, no replacement.** Q2/Q3/Q4/Q1 `$11,770/$8,745/$10,930/$34,050`; weaker medians in Q2/Q3. | `S1:L19039`; P046 ledger entry. |
| 047 | May 11 | Balanced value loss on interaction tokens. | **Reject.** Beat baseline 1/4; Q2 +50 `-$500`. | `S1:L19213`; P047 ledger entry. |
| 048 | May 11 | Official-context ladder-relative quality 3-seed screen. | **Pass screen.** Advanced to ten seeds. | `S1:L19943-L19986`; grouped 048-051 ledger entry. |
| 049 | May 11 | Ten-seed validation/1s audit of 048. | **No replacement.** Q2/Q3/Q4/Q1 `$11,615/$14,558/$13,225/$27,165`; improved Q3/Q4, lost Q2/Q1; 439 1s trades, 0 sign flips. | `S1:L19943-L19986`. |
| 050 | May 11 | Intra-minute same-side contract-ranking margin. | **Pass 3-seed screen.** Beat baseline in 3/4 folds and improved March. | `S1:L19943-L19986`. |
| 051 | May 11/12 | Ten-seed validation of same-side rank model. | **Promote frozen entry research baseline.** Q2 `$14,140 vs $13,330`; Q3 `$11,620 vs $10,200`; Q4 `$10,970 vs $10,525`; Q1 `$37,845 vs $32,810`; 429 1s trades, 0 sign flips. | `S1:L19939-L20090`. |
| 052 | May 12 | Full same-contract causal lifecycle screen over frozen 051 entries. | **Pass 3-seed screen.** Improved Q2/Q3/Q4/March, roughly matched Q1. | `S1:L20761-L20806`. |
| 053 | May 12 | Loss-only/early lifecycle path attempt using the insufficient processed decision-row source. | **Superseded.** Selected contracts could leave the ATM ladder; not decision-grade. | `S1:L20761-L20806`; grouped 052-054 ledger entry. |
| 054 | May 12 | Ten-seed full-path lifecycle validation with mandatory stop/target/flat. | **Keep lifecycle candidate.** Q2 `$13,460 vs $14,140`; Q3 `$15,455 vs $11,620`; Q4 `$15,715 vs $10,970`; Q1 `$37,920 vs $37,845`; March `$19,510 vs $16,730`; +100 positive. | `S1:L20761-L20806`. |

## Granular decoder: Protocol055-101

| P | Born | Hypothesis / role | Verdict and contemporaneous number | Evidence |
|---:|---|---|---|---|
| 055 | May 12 | Path-level replay and attribution of frozen 054 exits. | **Keep 054, not paper-ready.** 1s drag and Q2 clipped-winner/model-exit-loss cohorts required diagnosis. | `S1:L20986`; ledger P055. |
| 056 | May 12 | Reconstruct post-exit recovery, continued decay, clipped winners, and Q2 exit losses. | **Diagnostic.** Recovery after early exits was the next target; no model change. | `S1:L21102`; ledger P056. |
| 057 | May 12 | Learned recovery-confirmation layer atop frozen 054. | **Reject.** Q2/Q3 unchanged, Q4 `$19,430 vs $19,540`, March `$20,210 vs $20,280`; no broad lift. | `S1:L21392`; ledger P057. |
| 058 | May 12 | Train directly on regret versus the frozen baseline exit. | **Reject.** Helped Q3 but damaged Q2/Q4/Q1/March; March `$11,010 vs $20,280`. | `S1:L21600`; ledger P058. |
| 059 | May 12 | Require one-bar confirmation before a giveback exit. | **Reject / third lifecycle miss.** Every fold weaker; Q3 +100 remained negative. | `S1:L21701`; ledger P059. |
| 060 | May 12 | Build causal trade-level and step-level lifecycle sequence dataset, retaining post-054 rows as labels. | **Keep foundation; no performance claim.** | `S1:L21991`; ledger P060. |
| 061 | May 12 | First GRU continuation model on 060. | **Reject replacement; keep architecture clue.** Aggregate Q3/Q4/Q1 improved, but March `$166,810 vs $185,510`. | `S1:L22086`; ledger P061. |
| 062 | May 12 | GRU residual override relative to 054. | **Reject.** Q3/Q4 improved; Q1/March damaged. | `S1:L22189`; ledger P062. |
| 063 | May 12 | Choose residual threshold on validation only. | **Strong challenger, reject replacement.** Q3 `$169,290 vs $120,800`; Q4 `$169,070 vs $157,500`; Q1 `$395,200 vs $355,120`; March `$180,360 vs $185,510`. | `S1:L22189`; ledger P063. |
| 064 | May 12 | Attribute March false residual overrides. | **Diagnostic.** March negative overrides cost `-$399,960` row-level, usually at step 0; median later recovery headroom about `$1,080`. | `S1:L22321`; ledger P064. |
| 065 | May 12 | Up-weight early negative-residual states with high future recovery/baseline regret. | **Pass screen.** Q3 `$133,975`, Q4 `$275,710`, Q1 `$375,130`, March `$189,250`, all above 054. | `S1:L22478`; ledger P065. |
| 066 | May 12 | Ten-seed validation, attribution, and partial 1s replay of 065. | **Keep validated lifecycle challenger.** Q3 `$139,842.5`, Q4 `$270,260`, Q1 `$366,115`, March `$188,740`; delta-positive seeds 9/10, 10/10, 10/10, 8/10. | `S1:L22577`; ledger P066. |
| 067 | May 12 | Diagnose the three negative-delta seed/split cases. | **No new knob.** Misses limited to March seeds 4/10 and Q3 seed 3; judged too small to chase. | `S1:L22618`; ledger P067. |
| 068 | May 12 | Freeze hashes, exact ask/bid replay, slippage/order-state accounting. | **Infrastructure pass.** 32,490 rows matched with p99 PnL difference 0; +$0.25/side remained positive; 5,000 order records all `exit_filled`. | `S1:L22814`; ledger P068. |
| 069 | May 12 | Persist Protocol066 model/scaler/threshold artifacts. | **Pass infrastructure.** 30 bundles/120 files; metrics reproduced. | `S1:L23029`; ledger P069. |
| 070 | May 12 | Define no-order live shadow parity contract. | **Correctly blocked.** Template built; no real live JSONL yet; focused tests 30. | `S1:L23029`; ledger P070. |
| 071 | May 12 | Offline artifact-load/inference rehearsal. | **Pass offline.** 50 trades/1,250 rows, 0 failures; action mix represented. | `S1:L23285`; ledger P071. |
| 072 | May 12 | Reload every artifact and reproduce research exits. | **Pass exact.** 30 artifacts/32,490 rows, 0 mismatches; 26.7% of exits depended on 054 fallback. | `S1:L23285`; ledger P072. |
| 073 | May 12 | Remove the 054 fallback from 066. | **Reject simplification.** Q3/Q4/March and PF deteriorated; fallback remained required. | `S1:L23285`; ledger P073. |
| 074 | May 12 | First/unforced persistence attempt for the upstream 051/054 live stack. | **Superseded.** Q3 selected configuration drifted, so the attempt was stopped rather than accepted. | P075 ledger says the unforced rerun drifted; P074 audit directory `v4_aplus_hypothesis_074_protocol054_live_stack_artifacts/`. |
| 075 | May 12 | Force frozen 054 configs and persist the full 051-entry/054-fallback stack. | **Keep modular stack candidate.** 40 bundles/200 files; Q2/Q3/Q4/Q1/March `$14,420/$11,880/$16,895/$38,820/$20,890`; not byte-identical to original 054. | `S1:L23908`; ledger P075. |
| 076 | May 13 | Retrain the 051/054 modular baseline with Q4 2024 as earlier history. | **Positive, reject replacement.** Dynamic exits lost median edge in Q2 and Q1. | `S1:L25638-L25685`; grouped 076-082 ledger entry. |
| 077 | May 13 | Build Q4-start causal lifecycle sequence data from 076. | **Keep dataset foundation.** | `S1:L25638-L25685`. |
| 078 | May 13 | Train Q4-start residual recovery sequence model. | **Promising precursor**, advanced to persistence/reproduction. | `S1:L25638-L25685`. |
| 079 | May 13 | Persist 078 artifacts. | **Infrastructure produced**, then reproduction exposed a boundary mismatch. | `S1:L25638-L25685`. |
| 080 | May 13 | Reproduce 079 decisions. | **One floating-point threshold mismatch found.** Did not accept as exact. | `S1:L25638-L25685`. |
| 081 | May 13 | Add deterministic `1e-4` threshold margin and repersist. | **Keep best lifecycle challenger.** Q2 `$201,585 vs $200,635`; Q3 `$218,765 vs $156,410`; Q4 `$346,880 vs $158,410`; Q1 `$397,815 vs $319,560`; March `$206,065 vs $165,330`. | `S1:L25638-L25964`. |
| 082 | May 13 | Exact artifact reproduction of 081. | **Pass.** 40 artifacts/42,170 rows, 0 mismatches. | `S1:L25638-L25964`. |
| 083 | May 13 | One-second path audit for 081. | **Supportive.** 4,160 trades, 0 sign flips, p95 absolute difference `$10`; 276 mandatory 1s stop/target events. | `S1:L25964-L26002`; grouped 083-087 ledger entry. |
| 084 | May 13 | Freeze/slippage/order-state replay for 081. | **Pass infrastructure.** 42,170 rows exact; +$0.25/side positive; 5,000 sampled records all one contract/exit-filled. | `S1:L25964-L26002`. |
| 085 | May 13 | Protocol081 no-order shadow schema/template. | **Blocked by no real live JSONL.** | `S1:L25964-L26002`. |
| 086 | May 13 | Offline shadow rehearsal for 081. | **Pass.** 1,250 rows, 0 failures. | `S1:L25964-L26002`. |
| 087 | May 13 | Remove upstream fallback/router complexity. | **Reject simplification.** No-fallback damaged Q3/Q4 and PF. | `S1:L25964-L26002`. |
| 088 | May 12/13 | Combine 051 entry + 054 fallback + 081 residual in a no-order router; then attempt market-hours capture. | **Offline pass; live blocked.** 1,250 rows/0 failures offline. Gateway live capture returned subscription errors; delayed 24-row plumbing check passed but was non-promotion-grade. | `S1:L26266-L26671`. |
| 089 | May 13 | Convert router JSONL into a shadow-paper ledger. | **Pass schema/accounting, not performance.** 50 closed paths, `$36,150`; concurrency warning because input was selected-path rehearsal. | `S1:L27155`; ledger P089. |
| 090 | May 13 | Enforce strict one-position lifecycle over 089. | **Pass.** 1,250 input rows -> 276 rows/13 serial trades; `$6,210`; max concurrency 1. | `S1:L27278`; ledger P090. |
| 091 | May 13 | Broader strict offline router replay. | **Pass infrastructure.** 10,425 rows/417 paths -> 2,102 rows/87 trades; `$27,130`; max concurrency 1. | `S1:L27369`; ledger P091. |
| 092 | May 13 | Learn which entry deserves the single occupied slot, keeping 081 exits fixed. | **Reject promotion.** Q3 `$57,640 vs $57,790`; Q4 `$90,140 vs $92,320`; Q1/March beat baseline. | `S1:L27663`; ledger P092. |
| 093 | May 13 | Attribute 092 vs first-available serial baseline. | **Diagnostic.** Q3 delta `-$8,970`, mainly no-entry `-$7,660` and occupancy `-$3,050`; same-minute swaps `+$1,740`. | `S1:L27785`; ledger P093. |
| 094 | May 13 | Pick threshold by validation delta vs strict baseline. | **No-op/reject.** Exact 092 results. | `S1:L27954`; grouped 094-096 ledger entry. |
| 095 | May 13 | Train target as candidate PnL minus best positive opportunity blocked before exit. | **Mixed/reject.** Q3 `$60,570 vs $57,790`, but Q4 `$89,520` and Q1 `$86,780` deteriorated. | `S1:L27954`. |
| 096 | May 13 | Validation-selected blend of profit and slot scorers. | **Research clue, reject promotion.** Preserved Q3 `$60,570`, reverted later splits toward 092, still lost Q4. | `S1:L27954`. |
| 097 | May 13 | Candidate-set network with explicit wait/take and train-only DP oracle. | **Keep architecture, reject promotion.** Q3 `$59,210`; Q4 `$91,150`; Q1 `$95,290`; March `$42,290`; Q4 still `-$1,170` vs strict baseline. | `S1:L28087`; grouped 097-099 ledger entry. |
| 098 | May 13 | Add candidate-PnL utility auxiliary loss. | **Reject.** Improved Q3/Q4 vs 092, damaged Q1/March, still lost Q4 baseline. | `S1:L28087`. |
| 099 | May 13 | Lower/alter recall threshold to reduce timid waiting. | **Reject.** Worsened Q3/Q4. | `S1:L28087`. |
| 100 | May 13 | Attribute 097's Q4 seed failures. | **Diagnostic.** Seed 3 was `-$18,610` vs baseline, 59 fewer trades, and missed `$62,830` of positive baseline-only winners. | `S1:L28186-L28204`. |
| 101 | May 13 | Add causal previous-event and rolling-three history to wait/take policy. | **Freeze research promotion candidate.** Q3 `$59,130 vs $57,790`; Q4 `$92,460 vs $92,320`; Q1 `$95,010 vs $90,000`; March `$41,790 vs $39,100`; Q4 margin only `+$140`. | `S1:L28186-L28412`. |

## Granular decoder: Protocol102-125

| P | Born | Role | Verdict and contemporaneous number | Evidence |
|---:|---|---|---|---|
| 102 | May 13 | Freeze 101 and run seed/month/concentration readiness diagnostics. | **Research candidate; broader validation required.** Positive seed-margin fractions Q3/Q4/March `.60`, Q1 `.80`; July `-$2,020`, December `-$7,500` vs baseline. | `S1:L28245-L28412`. |
| 103 | May 13 | Check Q4 2024 external-audit readiness. | **Blocked by missing candidate/lifecycle outcomes, not data quality.** | `S1:L28511`; ledger P103. |
| 104 | May 13 | Build frozen Q4-2024 candidate stream. | **Infrastructure pass**, enabling external stress. | `S1:L28726-L28786`. |
| 105 | May 13 | Build Q4-2024 lifecycle sequence table. | **Infrastructure pass.** | `S1:L28726-L28786`. |
| 106 | May 13 | Apply frozen 081 lifecycle exits to Q4-2024 candidates. | **Infrastructure pass.** | `S1:L28726-L28786`. |
| 107 | May 13 | Score frozen 101 on Q4 2024. | **Research-only pass.** Median `$58,280 vs $54,590`, margin `+$1,290`, PF `3.460`, but positive seed-margin fraction `.60`; backward temporal stress, not chronological promotion. | `S1:L28726-L28786`. |
| 108 | May 13 | Attribute Q4-2024 weakness. | **Diagnostic.** Mostly missed/abstained post-open convex call winners; same-minute swaps net positive. | `S1:L28851-L29032`. |
| 109 | May 13 | Average five frozen 101 seed logits without retraining. | **Reject.** Helped worst external seed but changed registered medians Q3/Q4/Q1/March by `-$910/-$900/-$3,580/-$2,710`. | `S1:L28994-L29032`. |
| 110 | May 13 | Preflight missing Q3-2024 batch. | **Ready for approval only.** Existing continuous coverage 375 sessions; missing block 64 sessions. | `S1:L29106-L29125`. |
| 111 | May 13 | Make paid-data approval an executable guard. | **Pass guardrail.** Dry-run/estimate allowed; actual endpoints require exact manifest approval. | `S1:L29213-L29243`. |
| 112 | May 13 | Break down Protocol101 money/training/validation/test and recover premium fields. | **Diagnostic only.** Replay had no starting balance/compounding initially; entry-quote preservation gap was identified for later backfill. | `S1:L29289-L30086`; P112 audit directory. |
| 113 | May 13/14 | Export trade overlay/equity/replay CSV and paper-account inspection. | **Inspection only.** 6,989 replay trades, 369 sessions, 145,728 SPX bars; later one-seed headline `$10k -> $329,050`, explicitly not promotion evidence. | `S1:L29379-L30488`. |
| 114 | May 13/14 | Try to falsify the frozen 101 curve via concentration, serial/affordability, slippage, and delay stress. | **`fragile_needs_more_data`.** Paper-account checks passed, but one-minute combined delay turned several split medians negative. | `S1:L30572-L30851`. |
| 115 | May 14 | Replay on already-held CBBO-1s. | **Partial support.** 524/5,493 trades (9.5%); `$233,640 -> $232,760`, 0 sign flips; no Q3/external coverage. | `S1:L31098`. |
| 116 | May 14 | Prepare targeted high-resolution request for exact selected sessions/symbols. | **Approval-ready.** Estimated `$4.7930`, hard cap `$10`; CMBP-1 before 2025-02-20 and CBBO-1s after. | `S1:L31758`. |
| 117 | May 14 | Download approved targeted batch and replay frozen 101. | **Supportive timing pass.** 239/239 files, 5.6GB, 5,493/5,493 trades, 0 sign flips; high-res delta `-$12,290` on `$1,648,340`. | `S1:L32041`. |
| 118 | May 14 | Historical live-schema no-order shadow rehearsal. | **Pass historical rehearsal.** 1,028 trades/2,056 observations, concurrency 1, no unaffordable trades; `$10k -> $325,130`; no orders. | `S1:L32232-L32440`. |
| 119 | May 14 | Live-readiness/entitlement gate. | **Initially blocked.** Gateway eventually connected; delayed 24-row plumbing passed, but live SPX/VIX/SPXW subscriptions were absent. | `S1:L32592-L33248`. |
| 120 | May 14 | Prove frozen 051 surface scorer is portable to live code. | **Pass offline.** 1,000 decisions, finite edge fraction 1.0; 561 calls/439 puts. | `S1:L32777`; P120 report. |
| 121 | May 14 | Wire 051 scorer into actual 101 entry inference. | **Pass.** 266 events/1,360 candidate rows/0 nonfinite features; cleared zero-filled-edge blocker. | `S1:L32881-L33078`. |
| 122 | May 14 | Define capital assumption and affordability. | **Pass `$10,000` paper baseline.** `$500` is access reserve, not bankroll. | `S1:L33308-L33438`. |
| 123 | May 14 | Rehearse frozen trades through order-state machine. | **Pass.** 1,028 trades, 0 skips, max concurrency 1, all final state `exit_filled`; no broker call. | `S1:L33498-L33572`. |
| 124 | May 14 | Consolidate live-data parity state. | **Blocked live subscriptions; delayed plumbing passed.** | `S1:L33520-L33572`. |
| 125 | May 14 | Package visual inspection, shadow schema, and Tuesday checklist. | **Ready for no-order shadow only.** 2,056 shadow events, max open positions 1; top 20 trades 19.1% of net PnL. | `S1:L34037`. |

## Granular decoder: Protocol126-160

| P | Born | Role | Verdict and contemporaneous number | Evidence |
|---:|---|---|---|---|
| 126 | May 14 | Reprice frozen 101 under sub-minute entry/exit delays and define a live quote-expiry budget. | **Blocked: incomplete timing coverage.** 32,958 delay rows existed, but not enough critical-path coverage to clear paper readiness. | `S1:L34074-L34451`; ledger P126; P126 report. |
| 127 | May 14 | Expand the live-shadow schema to include decisions, risk blocks, market snapshots, and account state. | **Pass schema hardening.** Ready for live capture, not proof of market parity. | `S1:L34074-L34451`; ledger P127. |
| 128 | May 14 | Put deterministic stale-data, premium, overlap, daily-loss, and settlement guards around a `$10,000` one-contract paper account. | **Pass overlay.** Ready for no-order shadow; paper submission still gated. | `S1:L34074-L34451`; ledger P128. |
| 129 | May 14 | Test adaptive one-to-three-contract sizing with 101 entries/exits frozen. | **Reject.** Loss clustering worsened; worst day was about `-$5,520` versus `-$2,080` for one contract. | `S1:L34074-L34451`; ledger P129; P129 report. |
| 130 | May 14 | Package the Tuesday no-order shadow runbook and preflight contract. | **Ready for no-order shadow only.** No trading verdict. | `S1:L34074-L34451`; ledger P130. |
| 131 | May 14 | Track account PnL and scale only after profits using conservative multi-contract ladders. | **Reject promotion.** More PnL did not improve risk enough; initial paper remained one contract. | `S1:L34604`; ledger P131. |
| 132 | May 14 | Search confidence/account-aware sizing policies around frozen 101. | **Keep offline research candidate.** Three sizing hypotheses passed the internal screen; none were live-approved. | `S1:L34737`; ledger P132. |
| 133 | May 14 | Stress the sizing candidates by split, month, day, and slippage. | **Pass research screen.** `lower_two_contract_threshold` survived; still pre-parity. | `S1:L34737`; ledger P133. |
| 134 | May 14 | Attribute the surviving sizer trade by trade against one contract. | **Pass research attribution.** Incremental PnL `$34,950`; 184 scaled trades and 85.3% of trades remained one contract. | `S1:L34737`; ledger P134. |
| 135 | May 14 | Test an always-on account-aware sizer across starting-cash and slippage assumptions. | **Pass research sensitivity.** Not approved for paper/live scaling. | Account-aware sizing sequence in `S1:L34737-L34971`; ledger P135. |
| 136 | May 14 | Replace a fixed daily stop with an equity-scaled stop. | **Pass research screen.** Two stop hypotheses accepted; no live verdict. | `S1:L34737-L34971`; ledger P136. |
| 137 | May 14 | Consolidate confidence, account state, premium limits, drawdown, and stops into Account-Aware Sizer V1. | **Keep offline candidate.** `$10,000 -> $379,820` versus `$319,050` for one contract in the later consolidated replay. | `S1:L34971`; ledger P137; P151 report. |
| 138 | May 14 | Exempt the base one-contract trade from the premium exposure cap. | **Reject.** External block remained negative. | `S1:L34971`; ledger P138. |
| 139 | May 14 | Retune the sizer's daily stop to avoid blocking recovery winners. | **Pass research screen.** Best rule `-$1,500 + 0.500%` of equity; still not live. | `S1:L34971`; ledger P139. |
| 140 | May 14 | Prepare IB Gateway paper-mode morning autostart and API preflight assets. | **Infrastructure ready to install.** No strategy evidence. | `S1:L34971-L35617`; ledger P140. |
| 141 | May 14 | Add a paper-account-only permission and account probe guard. | **Pass probe; orders disabled.** Connected to the paper account without opening submission. | `S1:L34971-L35617`; ledger P141. |
| 142 | May 14 | Add a guarded executor and exercise only its non-submitting path. | **Pass dry-run validation.** No paper order was sent. | `S1:L34971-L35617`; ledger P142. |
| 143 | May 14 | Install and verify morning LaunchAgents. | **Infrastructure pass at birth.** Later operational state changed; this was not parity evidence. | `S1:L34971-L35617`; ledger P143. |
| 144 | May 14 | Define append-only JSONL/CSV paper trade logging. | **Pass logging contract.** Three synthetic events; no market or performance result. | `S1:L35617`; ledger P144. |
| 145 | May 14/15 | Rehearse cold start and paper API availability. | **Initially blocked, then pass.** Final rehearsal exposed paper API port `4002`; no order path exercised. | `S1:L35737-L36246`; ledger P145. |
| 146 | May 14/15 | Check credential-backed unattended IB Gateway readiness without printing secrets. | **Pass readiness check.** Credentials were present locally; this report does not read or reproduce them. | `S1:L36108-L36246`; ledger P146. |
| 147 | May 15 | Orchestrate the morning Protocol101 session: startup, preflight, pulse-based shadow/paper-dry-run, and evidence capture. | **Keep operations harness, later superseded by 160.** It produced no-order/dry-run evidence but depended on repeated short connections. | `S1:L38777`, `S1:L39804-L39818`; P147 audit tree. |
| 148 | May 15 | Analyze a completed session and fail closed on missing evidence. | **Pass analyzer plumbing.** Operational report only. | `S1:L37644`; P148 audit tree. |
| 149 | May 15 | Render the live paper log as an inspection visual. | **Pass visualization plumbing.** No edge or parity claim. | `S1:L37644`; P149 audit tree. |
| 150 | May 15 | Require explicit preflight, shadow, timing, risk, and authorization evidence before paper-order enablement. | **Fail-closed gate built.** It did not itself enable orders. | `S1:L37644`; P150 audit tree. |
| 151 | May 15 | Consolidate the best account-aware sizing policy against one contract. | **Pass research-only economics.** `$379,820 vs $319,050`, incremental `$60,770`; top day was 6.6% of positive incremental PnL. | `S1:L37969`; P151 report. |
| 152 | May 15 | Run a formal multi-contract promotion gauntlet including timing coverage. | **Blocked.** High-resolution coverage was 74.9%, below the 95% promotion requirement; economic delta remained `$60,770`. | `S1:L38211`; P152 report. |
| 153 | May 15 | Check that quantity-aware intents remain compatible with the live schema/risk stack while defaults still reject quantity two. | **Pass research compatibility.** 1,020 rows, no failures; default one-contract guards still blocked `qty=2`. | `S1:L38484`; P153 report. |
| 154 | May 15 | Aggregate 151-153 into a multi-contract promotion decision. | **Blocked.** Missing timing evidence; one-contract remained the operational baseline. | `S1:L38657`; ledger P154. |
| 155 | May 15 | Define the one-contract live-paper timing/fill evidence needed to unblock later decisions. | **Blocked at birth.** Zero closed one-contract paper trades were available. | `S1:L38667-L38900`; ledger P155. |
| 156 | May 18 | Consolidate launchd state, log tails, API probes, and failure signals into one autostart observability report. | **Blocked at birth.** LaunchAgents were not loaded in that snapshot; later operations continued. | `S1:L39065`; ledger P156. |
| 157 | May 19 | Add the daily Protocol101 operations monitor over startup, live rows, guards, orders, fills, and closeout. | **Keep operational monitor.** It reports evidence; it does not establish edge. | `S1:L39818`; `v4/scripts/run_protocol157_protocol101_daily_ops_monitor.py`. |
| 158 | May 19 | Bridge live 051 surface decisions and 101 entry inference into guarded paper-order intents. | **Pass no-order bridge verification.** 42 contracts, 2 decisions, 0 submitted orders. | `S1:L39874-L40042`; P158 tests/audit output. |
| 159 | May 20 | Audit historical/live feature parity and repair persistent official context plus one-minute resampling. | **Pass with warnings after fixes.** Corrected stale/missing context plumbing; did not prove profitable live fills. | `S1:L41833`; Protocol159 audit/report sources. |
| 160 | May 20 | Replace repeated pulse connections with one persistent IBKR connection that carries model state and guarded order lifecycle through the session. | **Keep/current runtime architecture.** Persistent cadence separated fast market polling from slower entry evaluation; creation itself was not live-parity proof. | `S1:L42613-L42740`; `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py`. |

## Number-gap audit

The transcript changes the interpretation of the old Git gap:

- **001:** no exact birth, implementation, audit directory, or verdict. It is best labeled reserved/unused, not reconstructed.
- **002-008:** the first generalization and A+ permission/value-veto campaign. `003C` is a named side branch, not a decimal protocol and not Protocol004.
- **009-020:** a continuous one-change hypothesis chain and its locked/fresh audits. The early artifacts often use `hypothesis_009` rather than “Protocol009,” but the transcript explicitly advances these numbers in order.
- **021-050:** no missing protocols. They cover Q1 attribution, learned quality/value objectives, VWAP replacement, exit falsifications, walk-forward methodology, official context, and contract ranking.
- **055-065:** no missing protocols. They are the failure-analysis and sequence-lifecycle bridge between 054 and validated 066.
- **074:** a real but superseded persistence attempt. Its forced/frozen successor is 075.
- **076-087:** a second lifecycle build starting earlier in history, ending with deterministic Protocol081 and exact Protocol082 reproduction.
- **082-100:** no missing protocols. 083-087 are validation/parity/simplification checks; 088-091 build serial shadow accounting; 092-100 are entry-arbitration experiments and attribution.
- **112:** a money/accounting and field-provenance investigation, not another model.
- **140-150:** mostly guarded paper operations. They should not be cited as independent strategies or evidence that 101 survived live trading.
- **147:** the pulse-based session harness. **160** later superseded its connection model, though supporting analyzers and logs remained useful.

Therefore the protocol numbers are an experiment-and-operations journal, not a sequence of 160 independently trained strategies.

## Direct contemporaneous statements

These short extracts anchor the most consequential direction changes. They are quoted from `S1`, not from the later ledger.

| Event | Contemporaneous wording | Evidence |
|---|---|---|
| Protocol002 generalization result | “no broad data-purchase signal yet” | `S1:L5817` |
| Protocol007 provisional status | “narrow broad-data-purchase approval” | `S1:L8254` |
| Protocol007 fresh reversal | “failed the fresh Q3 audit” | `S1:L8828` |
| Fixed-threshold coverage failure | “Q3 median trades stayed at 1” | `S1:L8933` |
| Protocol024 branch | “first one that looks meaningfully better balanced” | `S1:L12312` |
| Hard quality veto problem | “rejected convex put winners” | `S1:L15111` |
| Broader-methodology turn | “the broader walk-forward edge is real” | `S1:L16039` |
| Protocol051 selection | “strongest v4 research baseline candidate” | `S1:L20090` |
| Protocol101 serial gate | “first model to clear the strict serial gate” | `S1:L28204` |
| Protocol114 verdict | `fragile_needs_more_data` | `S1:L30851` |
| Protocol160 redesign | “persistent IBKR connection model” | `S1:L42613` |

## Corrections and additions to the prior distillation

This report supplements [PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md](DO_NOT_RETEST.md); it does not alter the April edge ledger.

1. **The Git lineage remains absent, but the conversational lineage is recovered.** The prior report accurately found no intervening commits in the Apr-28..May-13 parent edge. `S1` now supplies contemporaneous birth order, intent, outputs, reversals, and file-writing context. It cannot supply missing commit object IDs or prove that every transient working-tree state was committed.
2. **Protocol051 “edge” is not an executable-dollar edge.** In the live scorer, `edge = surface_action_score - surface_flat_score`; it is an internal model score margin, not dollars, expected PnL, probability, or a bid/ask execution cushion (`S1:L42383`). The `edge25` label was an upstream candidate/filter convention, not a learned 25-dollar threshold. Any wording that calls 051's output an executable-dollar edge should be read as incorrect.
3. **Protocol001 is not recovered.** The numbering begins with 002 in the primary transcript. The cleanest evidence-based status is `UNUSED/RESERVED`, not `UNKNOWN STRATEGY`.
4. **Protocols055-100 were not omitted research.** They form the lifecycle, reproducibility, shadow-accounting, and strict-entry-arbitration chain that leads from 054 to 066, 081, and 101.
5. **Protocol160's birth is now dated and motivated.** It was created on May 20 to replace repeated connect/disconnect pulses with a persistent broker connection and decoupled polling/evaluation cadence (`S1:L42613-L42740`). The current registry's use of 160 is thus lineage-consistent.
6. **The protocol farm is still pre-parity evidence.** Protocol114 explicitly found timing fragility; 119/124 lacked live subscriptions; 126 had incomplete timing coverage; 152/154 blocked multi-contract promotion; and 155 began with no closed paper trades. Later runtime wiring cannot retroactively upgrade the replay curve to live parity.

## Canonical decoder and current status

This table connects the historically important modules to the present stack. “Current” is grounded in `v4/promotion/PAPER_TRADING_DEFAULT.json` and the surviving source tree as of 2026-07-19; it is not a new promotion decision.

| Protocol | What it actually is | Born / lineage | Current status |
|---:|---|---|---|
| 051 | Official-context surface scorer and same-side contract ranker; outputs an internal action-vs-flat score margin. | May 11/12; 048-050 screens -> ten-seed 051. | Upstream scorer in `v4/live/protocol051_surface_edge.py`; frozen component of the registered stack. |
| 054 | First validated full same-contract lifecycle policy over frozen 051 entries; later retained as fallback. | May 12; 052 screen, 053 source failure, 054 corrected full-path validation. | Historical fallback component; not the registered headline entry policy. |
| 066 | Recovery-aware GRU residual lifecycle challenger layered over 054. | May 12; 055-065 attribution/model chain -> ten-seed 066. | Important predecessor; superseded in the current lifecycle role by 081. |
| 081 | Deterministic residual lifecycle model with a `1e-4` decision margin. | May 13; 076-080 earlier-history rebuild/reproduction issue -> 081; 082 exact reproduction. | Registered lifecycle/exit model, loaded through `v4/live/protocol066_inference.py`. |
| 101 | Candidate-set wait/take entry policy using causal previous-event and rolling-three history under strict serial occupancy. | May 13; 088-100 serial shadow and arbitration chain -> 101. | Current paper-default entry policy (`PAPER_DEFAULT_PROTOCOL101`). |
| 113 | Trade/equity/overlay export plus replay/account inspection. | May 13/14; built after freeze and money/accounting diagnosis. | Evidence/visualization tooling, not a trading model; current exporter is `v4/scripts/export_protocol101_trade_charts.py`. |
| 155 | One-contract live timing/fill evidence analyzer. | May 15; created after multi-contract timing promotion was blocked. | Evidence gate/analyzer; born blocked with no closed paper trades. |
| 160 | Persistent paper trader integrating live context, 051 scoring, 101 entries, 081 exits, guards, logging, and order lifecycle. | May 20; superseded 147's short pulse connection model. | Current registered paper runtime entrypoint: `v4.scripts.run_protocol160_protocol101_persistent_paper_trader`. |

## Evidence limits and bottom line

- Transcript line numbers refer to physical JSONL lines in the immutable original and the matching ignored backup. Individual lines can be very long because tool results are serialized inside one JSON object.
- A transcript statement is strongest for contemporaneous intent, sequencing, and what the agent observed. Exact repository artifacts and reports remain preferable for numeric verification when they survive.
- Some grouped ledger patches summarize multiple protocol numbers in one message. Where a protocol had no standalone report, the table says so through grouped line ranges rather than inventing an artifact.
- No sealed vault, credential value, frozen model artifact, broker endpoint, paid-data endpoint, training job, or runtime configuration was accessed or changed for this reconstruction.

The recovered lineage explains how an April strategy family became Protocol051, then acquired 054/066/081 lifecycle logic, strict serial Protocol101 entry arbitration, and finally Protocol160 operations. It also preserves the critical negative result: the attractive replay curve entered operations as **fragile, pre-parity evidence**, not as a proven live edge.
