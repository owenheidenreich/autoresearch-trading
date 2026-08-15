# What it would cost to settle the seven blocking divergence axes

**Dated 2026-08-12. This is a costing study, not a plan and not a schedule.**
[`STATUS.md`](../../STATUS.md) remains the only job register; nothing here is a job until it has a row
there. No axis was settled by writing this, and no axis status was changed.

## The headline

**Four of the seven blocking axes need no new market data and no spend.** Three of those four could be
started immediately with nothing but local files that are already on this machine; the fourth needs only
an owner signature that is already drafted.

That does not contradict "the project is waiting on calendar time" — settling these unblocks no gate,
because [G4](../../STATUS.md#8-gate-chain) sits behind a G1 pass the owned corpus cannot produce. It does
mean the honest count of blocking axes can fall from seven to three without waiting for a single new
session, and that the work which *does* need calendar time is a much shorter list than it looks.

| Axis | Binds at | What settles it | Tier | Startable today |
|---|---|---|---|---|
| `clock_and_dst` | G4 | Fixtures over owned bars either side of both DST transitions | 3 | **Yes — free** |
| `slippage_accounting` | G5 | Decompose the friction figure; make the packet refuse a double count | 3 | **Yes — free** |
| `reconnects_and_gaps` | G6 | Four boundary fixtures showing the live path fails closed | 3 | **Yes — free** |
| `emission_lag` | G4 | Owner signature, then one local derivation from banked receipts | 1 then 3 | Blocked only on signature |
| `definitions_survivorship` | **G3** | Banked live definitions vs a historical definitions request | **1 — paid** | Needs a purchase decision |
| `revisions_vs_first_print` | G4 | Banked live rows vs a *delayed* historical re-request | **1 — paid** | Needs a purchase decision |
| `partial_fills_and_rejects` | G7 | Twenty consecutive no-order shadow sessions | **1 — broker** | No. Needs G1–G6 first |

## The three that are free and startable now

### `clock_and_dst` — the corpus already contains both transitions

Needs fixtures proving the same wall-clock minute maps to the same bar either side of a daylight-saving
change. **Verified 2026-08-12 that the required sessions are on disk**, in
`~/.autoresearch-trading/pathd_2025-08-01_2026-07-31/raw/databento/glbx_es_ohlcv_1m/`:

| Transition | Session before | Session after | Both present |
|---|---|---|---|
| Fall back, 2025-11-02 | 2025-10-31 | 2025-11-03 | yes |
| Spring forward, 2026-03-08 | 2026-03-06 | 2026-03-09 | yes |

The corpus spans 2025-08-01 to 2026-07-31, so it straddles both. No download, no vendor, no network. This
is the cheapest of the seven and the one whose failure mode is nastiest — the register's own note is that
an off-by-one-hour error *"looks exactly like a real regime change."*

### `slippage_accounting` — the double-count is real and currently latent

Confirmed by reading both sides:

- `es_round_trip_friction_points = 0.358` is `FROZEN`, and its unfreeze condition describes it as a
  **"spread, fee and slippage study"** — so slippage is already inside the 0.358.
- [`candidate_packet.py:166`](../validation/candidate_packet.py) takes a separate
  `slippage_per_side_points`, computes `2 x slippage x multiplier x quantity`, and adds it to commission to
  form `total_cost_usd`. The CLI exposes `--slippage-per-side-points`.

The default is `0.0`, so **nothing is double-counted today**. But the composition of 0.358 is written down
nowhere — the gate-chain audit uses it as a single scalar throughout — and a flag that invites a nonzero
value sits next to a bar that already contains it. Settling this is local: decompose the figure if the
underlying study survives, and otherwise make the packet refuse a nonzero argument rather than documenting
a convention nobody will read.

### `reconnects_and_gaps` — specified, never exercised

The behaviour is already specified in the gate-chain audit §8, and `StreamReadiness` discards the first
interval after a reconnect. What is missing is any test that the live path *fails closed* on a reconnect, a
duplicate, a correction, or a sequence gap. Four boundary fixtures, no data, no network.

## The one that needs only a signature

`emission_lag` is now the subject of a drafted rule at
[`governance/EMISSION_LAG_LOWERING_RULE_2026_08_12.md`](../../governance/EMISSION_LAG_LOWERING_RULE_2026_08_12.md).
Once signed, the derivation runs against receipt bytes already on disk. Nothing further is bought or
captured. The reason it is Tier 1 first is that the choice of statistic decides whether the number rises or
falls, which is a judgement about safety rather than a computation.

## The two that need a paid request

Both are small, well-scoped Databento Historical requests. Neither may proceed without an explicit owner
purchase decision (Tier 1).

### `revisions_vs_first_print` — the delay is the whole point

This asks whether the Historical API serves corrected bars the live feed never showed. There is already
**one session of partial evidence**: the 2026-08-03 twin comparison matched all 914 live CBBO-1m rows
against historical after decode, with identical columns and dtypes
([comparison](../../../v4/audit/autoresearch/databento_live_opra_training_twin_2026_08_03/comparison_same_session_attempt002/comparison_result.json)).

**That test was weaker than it looks, in a way worth stating plainly.** The historical file was requested
the *same day* as the live capture. A revision applied hours or days later is invisible to a same-day
comparison, so what was actually proven is same-day value identity, not the absence of revisions.

The Track-A capture makes a genuinely stronger test available for the first time: its live rows for
2026-08-10, 08-11 and 08-12 are banked and hashed, and are now days old. Re-requesting those three sessions
today compares a first print against a historical record that has had time to be revised. This is the
cheapest way to convert a known-weak result into a real one, and it decays — the sooner it is requested,
the less it proves, so there is a genuine argument for waiting rather than rushing.

### `definitions_survivorship` — and a gate-ordering problem it exposes

This asks whether historical definitions list contracts that did not exist at decision time. The live half
is already banked: every Track-A window wrote a `definitions/` directory alongside its `market/` data.

There is also a **positive signal already on disk**: the 08-03 comparison recorded
`same_day_added_symbol_count: 18` — eighteen SPXW strikes came into existence *during* that session. So
intra-session growth is not hypothetical. Whether the historical definitions file backfills those 18 to the
session start is exactly the unchecked question.

**This axis binds at G3, and G3 passed on 2026-08-12 while it was still `UNKNOWN`.** That is not a claim
the certification is wrong — the issuance rests on latency and freshness receipts, which are a different
question — but the register names G3 as this axis's gate and G3 closed without it. Either the axis binds at
G4 in truth, or the G3 issuance should have consulted the register. That is an owner-visible discrepancy
and is reported rather than resolved here.

## The one that cannot start

`partial_fills_and_rejects` needs twenty consecutive no-order shadow sessions. G7 is not built, and it sits
behind G1–G6. This is genuinely calendar- and broker-bound and there is no way to shorten it.

## Two things found while scoping

**The divergence guard has no production caller.** `assert_no_unknown_on_path` is invoked in exactly one
place in the repository: [`v5/tests/test_divergence.py`](../../tests/test_divergence.py). No fit path, gate
driver, or certification command calls it. This is harmless *today* — there is no fit to guard, because G4
is blocked — but it means the register currently binds only whoever remembers to call it. The register was
built because "look harder is not a control"; the same objection applies one level up, to a guard that
nothing invokes. The fix is to call it from the fit entry point at the time that entry point is written,
and the useful thing to record now is that this is a known gap rather than an oversight to rediscover.

**`/Volumes/AR_TRADING_DATA` is not mounted.** The evidence index names it as holding exit features,
labels, and option payoff reports. Only `Macintosh HD` is mounted as of 2026-08-12. Nothing scoped above
needs it — all four free axes read the internal corpus or v5 source — but any work that does will stop
immediately until the drive is attached.

## What this study did not do

Settle any axis, change any status, compute any economics, contact any vendor, or spend anything. Every
figure above was read from files already on this machine.
