# Autoresearch-Trading Architecture

Detailed ASCII architecture diagrams for the autonomous SPX 0DTE options trading platform.

---

## 1. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          AUTORESEARCH-TRADING SYSTEM                                │
│                    Autonomous SPX 0DTE Options Trading Platform                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌──────────────┐   ┌───────────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │  DATA LAYER  │──▶│  TRAINING LAYER   │──▶│  VALIDATION  │──▶│  LIVE TRADING  │  │
│  │  prepare.py  │   │  inner_loop.py    │   │  replay.py   │   │  live/         │  │
│  │              │   │  train.py         │   │              │   │  service.py    │  │
│  └──────┬───────┘   └────────┬──────────┘   └──────┬───────┘   └───────┬────────┘  │
│         │                    │                     │                    │           │
│         ▼                    ▼                     ▼                    ▼           │
│  ┌──────────────┐   ┌───────────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │ Polygon S3   │   │ Akash H100 GPU    │   │ Historical   │   │ IBKR Gateway   │  │
│  │ IBKR Gateway │   │ Sonnet Agent      │   │ data.pt      │   │ Paper Trading  │  │
│  └──────────────┘   └───────────────────┘   └──────────────┘   └────────────────┘  │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                         INFRASTRUCTURE (infra/)                             │    │
│  │  deploy.sh ─── boot/start/sync/stop/ssh/logs/status ─── watchdog.sh       │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Pipeline (`prepare.py`)

```
                              DATA PIPELINE
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  EXTERNAL DATA SOURCES                                              │
│  ════════════════════                                               │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  IBKR API    │  │  IBKR API    │  │  Polygon S3 Bulk Data   │  │
│  │  (Gateway)   │  │  (Gateway)   │  │  (4-year rolling)       │  │
│  ├──────────────┤  ├──────────────┤  ├──────────────────────────┤  │
│  │ SPX Index    │  │ VIX Index    │  │ SPXW Option Chains      │  │
│  │ SPY ETF bars │  │ (CBOE)       │  │ (0DTE strikes, greeks)  │  │
│  │ (volume src) │  │              │  │                          │  │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘  │
│         │                 │                        │                │
│         ▼                 ▼                        ▼                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │      INCREMENTAL CACHE LAYER                                │   │
│  │      ~/.cache/autoresearch-trading/data/                    │   │
│  │                                                              │   │
│  │  ┌──── APPEND-ONLY (monolithic pkl) ────────────────────┐   │   │
│  │  │  spy_1min.pkl  │ spx_1min.pkl  │ vix_1min.pkl       │   │   │
│  │  │  _incremental_update(): load → find max → download   │   │   │
│  │  │  new → concat → deduplicate → save                   │   │   │
│  │  └───────────────────────────────────────────────────────┘   │   │
│  │                                                              │   │
│  │  ┌──── PER-DAY (already incremental) ───────────────────┐   │   │
│  │  │  spxw/{date}.pkl      │ spxw_chain/{date}.pkl        │   │   │
│  │  │  ~989 files each      │ Skips existing days           │   │   │
│  │  └───────────────────────────────────────────────────────┘   │   │
│  │                                                              │   │
│  │  ┌──── AGGREGATE (rebuilt each run, ~1 sec) ──────────────┐  │   │
│  │  │  spxw_full.pkl        │ spxw_chain_full.pkl          │   │   │
│  │  └───────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  FEATURE ENGINEERING                         │   │
│  │  compute_features() → 37 features per 1-min bar             │   │
│  │                                                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │   │
│  │  │ Price/Volume │  │ Session/Time │  │ Options/Greeks    │  │   │
│  │  │ ─ VWAP bands │  │ ─ bar_of_day │  │ ─ ATM IV         │  │   │
│  │  │ ─ returns    │  │ ─ session %  │  │ ─ delta/gamma    │  │   │
│  │  │ ─ volume z   │  │ ─ is_power   │  │ ─ charm/vanna   │  │   │
│  │  │ ─ momentum   │  │ ─ day_of_wk  │  │ ─ put/call skew │  │   │
│  │  ├──────────────┤  ├──────────────┤  ├───────────────────┤  │   │
│  │  │ Trend/Vol    │  │ Key Levels   │  │ Extended (10)    │  │   │
│  │  │ ─ Bollinger  │  │ ─ prev close │  │ ─ RSI            │  │   │
│  │  │ ─ ATR ratio  │  │ ─ VWAP cross │  │ ─ ATR ratio      │  │   │
│  │  │ ─ VIX regime │  │ ─ gap        │  │ ─ Bollinger %B   │  │   │
│  │  ├──────────────┤  ├──────────────┤  ├───────────────────┤  │   │
│  │  │ Mkt Struct(5)│  │              │  │                   │  │   │
│  │  │ ─ poc_dist   │  │              │  │                   │  │   │
│  │  │ ─ va_position│  │              │  │                   │  │   │
│  │  │ ─ vwap_band  │  │              │  │                   │  │   │
│  │  │ ─ ib_break   │  │              │  │                   │  │   │
│  │  │ ─ theta_pres │  │              │  │                   │  │   │
│  │  └──────────────┘  └──────────────┘  └───────────────────┘  │   │
│  └──────────────────────┬──────────────────────────────────────┘   │
│                          ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      data.pt (PyTorch tensor)               │   │
│  │  features:  (N_bars, 37) float32                            │   │
│  │  targets:   call_pnl, put_pnl, exit labels, stopped P&L    │   │
│  │  prices:    atm/otm5/otm10 call/put price arrays           │   │
│  │  metadata:  dates, timestamps, day_boundaries               │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Autoresearch Training Loop (`inner_loop.py` + `train.py`)

```
                    AUTORESEARCH LOOP (on Akash H100 GPU)
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  ┌──────────────────────────────────────────────┐                        │
│  │  PROMPT CONTROL PLANE                        │                        │
│  │  ┌──────────────┐  ┌──────────────────────┐  │                        │
│  │  │ program.md   │  │ lab_notebook.md      │  │                        │
│  │  │ (strict      │  │ (persistent context: │  │                        │
│  │  │  contract)   │  │  wins + dead ends)   │  │                        │
│  │  └──────┬───────┘  └──────────┬───────────┘  │                        │
│  └─────────┼─────────────────────┼──────────────┘                        │
│            └──────────┬──────────┘                                        │
│                       ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────┐        │
│  │                  EXPERIMENT ITERATION                         │        │
│  │                                                               │        │
│  │   ┌─────────────┐    1. Read current train.py + history      │        │
│  │   │  Sonnet Agent │◀── 2. System: program.md + lab_notebook   │        │
│  │   │  (Code Writer)│    3. User: metrics + code + prompt      │        │
│  │   └──────┬───────┘                                            │        │
│  │          │ Returns modified train.py                          │        │
│  │          ▼                                                    │        │
│  │   ┌──────────────┐                                            │        │
│  │   │  VALIDATION  │  ─ AST syntax check                       │        │
│  │   │  GATES       │  ─ Contract compliance (4-head, 37 feat)  │        │
│  │   │              │  ─ Safety: no torch.compile/DDP            │        │
│  │   └──────┬───────┘                                            │        │
│  │          │ Pass                                                │        │
│  │          ▼                                                    │        │
│  │   ┌──────────────────────────────────────────┐                │        │
│  │   │  TRAINING (subprocess, ~6 min budget)     │                │        │
│  │   │  train.py on H100 + data.pt               │                │        │
│  │   │  ─ Four-head transformer                  │                │        │
│  │   │    (gate + direction + value + risk)       │                │        │
│  │   │  ─ Position-aware simulation               │                │        │
│  │   │  ─ Validation early stopping               │                │        │
│  │   └──────┬───────────────────────────────────┘                │        │
│  │          │ stdout → metrics JSON                              │        │
│  │          ▼                                                    │        │
│  │   ┌──────────────────────────────────────┐                    │        │
│  │   │  SCORING → PROMOTE / REVERT          │                    │        │
│  │   │  score > best? → keep (model+code)   │                    │        │
│  │   │  score ≤ best? → revert code          │                    │        │
│  │   └──────────────────────────────────────┘                    │        │
│  └───────────────────────┬───────────────────────────────────────┘        │
│                          │ Loop continues                                 │
│                          ▼                                                │
│  ARTIFACTS: experiments.v2.jsonl, status.json, artifacts/exp-*/          │
│                                                                          │
│  ══════ PBT MODE (Population-Based Training) ══════                      │
│                                                                          │
│  pbt-init → N members (generalists + specialists)                        │
│  pbt-run  → each member trains with env-var overrides                    │
│  Selection: elite carry-forward, exploit top-25%, explore top-50%        │
│  Anti-stagnation: 2 random members after 2 stalled generations           │
│  State: training/.pbt_state.json (resumable)                             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Live Paper Trading Stack (`training/live/`)

```
                    IBKR LIVE PAPER TRADING STACK
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   tools/paper_live.py (CLI entry point)                                  │
│          │                                                               │
│          ▼                                                               │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │  PaperTradingService (live/service.py)                       │      │
│   │  Orchestrates the full trading session                       │      │
│   └──┬──────────┬───────────┬──────────────┬─────────────────────┘      │
│      │          │           │              │                             │
│      ▼          ▼           ▼              ▼                             │
│  ┌────────┐ ┌──────────┐ ┌────────────┐ ┌──────────────────────┐       │
│  │Context │ │Decision  │ │ Contract   │ │  Execution Engine    │       │
│  │Engine  │ │Engine    │ │ Resolver   │ │  (OCO Orders)        │       │
│  └───┬────┘ └────┬─────┘ └─────┬──────┘ └──────────┬───────────┘       │
│      │           │             │                    │                    │
│  ┌───┴──────────────────────────────────────────────┴───────────┐      │
│  │                   COMPONENT DETAIL                            │      │
│  │                                                               │      │
│  │  context.py    — 30-day historical context + normalization    │      │
│  │  features.py   — 5s bars → 1m bars, same 37 features         │      │
│  │  decision.py   — Model inference → gate + dir + value + risk  │      │
│  │  resolver.py   — SPXW 0DTE contract resolution (ATM/OTM)     │      │
│  │  execution.py  — LMT orders, dynamic stop, OCO brackets      │      │
│  │  contracts.py  — Typed data contracts between components      │      │
│  └───────────────────────────────────────────────────────────────┘      │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │  IBKRMarketStream (service.py)                               │      │
│   │  ─ 5-sec real-time bars: SPX, SPY, VIX                      │      │
│   │  ─ Option top-of-book + Greeks for 6 SPXW contracts         │      │
│   │  ─ Aggregates to 1-min bars via FiveSecondMinuteAggregator  │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                          │
│   Session Lifecycle:                                                     │
│   1. Startup cleanup: cancel all orphaned orders                         │
│   2. Bar loop: aggregate → features → inference → entry/exit             │
│   3. Cooldown: 5-bar (5 min) block after stop-loss exits                 │
│   4. Shutdown: flatten open positions, session summary                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Infrastructure & Deployment (`infra/`)

```
              AKASH GPU DEPLOYMENT LIFECYCLE
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   LOCAL MACHINE                          AKASH H100 CONTAINER   │
│   ══════════════                         ════════════════════    │
│                                                                  │
│   deploy.sh boot ──▶ Create SDL deployment ──▶ H100/A100 GPU    │
│                      (bid → lease)             PyTorch 2.5.1     │
│                                                                  │
│   deploy.sh start ──▶ SCP upload: train.py, run_loop.py,        │
│                       prepare.py, data.pt, program.md,           │
│                       lab_notebook.md, best_model.pt             │
│                                                                  │
│   deploy.sh sync ──▶ rsync download results/ (auto, every 30s)  │
│                                                                  │
│   deploy.sh stop ──▶ Kill loop, download final results,          │
│                      close Akash lease                            │
│                                                                  │
│   Other: ssh, logs, status                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Feature Parity Verification

```
              DATA FLOW PARITY: TRAINING = REPLAY = LIVE
┌──────────────────────────────────────────────────────────────────┐
│                                                                    │
│  All three stages use the SAME feature pipeline from prepare.py:   │
│  ─ compute_features()                  → 37 FEATURE_NAMES          │
│  ─ normalize_features_with_context()   → rolling z-score           │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐    │
│  │  TRAINING    │  │  REPLAY      │  │  LIVE IBKR            │    │
│  │  prepare.py  │  │  replay.py   │  │  live/features.py     │    │
│  │  → data.pt   │  │  → csv       │  │  → real-time bars     │    │
│  ├──────────────┤  ├──────────────┤  ├───────────────────────┤    │
│  │ compute_     │  │ compute_     │  │ compute_              │    │
│  │  features()  │  │  features()  │  │  features()           │    │
│  │ normalize_   │  │ normalize_   │  │ normalize_            │    │
│  │  features_   │  │  features_   │  │  features_            │    │
│  │  with_       │  │  with_       │  │  with_                │    │
│  │  context()   │  │  context()   │  │  context()            │    │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬────────────┘    │
│         └──────────────────┴─────────────────────┘                 │
│                              │                                      │
│              Same 37 features, same normalization                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 7. Model Architecture (v6 Four-Head)

```
                    TradingModel — v6 Four-Head Architecture
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  INPUT: (batch, lookback=120, 37 features)                               │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────┐                                                        │
│  │  input_proj   │  Linear(37 → d_model=64) + GELU + Dropout            │
│  │  input_norm   │  LayerNorm(64)                                        │
│  │  pos_embed    │  Learned positional embedding (1, 120, 64)            │
│  └──────┬───────┘                                                        │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────────────────────────────────┐                        │
│  │  TRANSFORMER ENCODER (shared backbone)        │                        │
│  │  ─ depth=3 layers                             │                        │
│  │  ─ n_heads=4                                  │                        │
│  │  ─ ff_mult=3 (feedforward = 192)              │                        │
│  │  ─ norm_first=True (Pre-LN)                   │                        │
│  │  ─ dropout=0.3, causal masking                │                        │
│  │  Output: (batch, 120, 64) → last bar → (64,)  │                        │
│  └──────┬───────────────────────────────────────┘                        │
│         │                                                                │
│         │ ┌─────────────────────────────────────────────┐                │
│         │ │  POSITION STATE (7 dims, runtime-built)      │                │
│         │ │  [0] in_trade       (0 or 1)                 │                │
│         │ │  [1] bars_held      (normalized /390)        │                │
│         │ │  [2] unrealized_pnl (tanh-scaled)            │                │
│         │ │  [3] account_health (balance / capital)      │                │
│         │ │  [4] loss_streak    (consecutive / 3.0)      │                │
│         │ │  [5] best_pnl_since_entry (tanh-scaled)      │                │
│         │ │  [6] bars_since_pnl_high (/390)              │                │
│         │ └──────────┬──────────────────────────────────┘                │
│         │            │                                                    │
│         │     ┌──────┴──────┐                                            │
│         │     │ position_proj│  Linear(7 → d_model//4=16)                │
│         │     └──────┬──────┘                                            │
│         │            │                                                    │
│         │            │  pos_emb (16)                                     │
│         │            │                                                    │
│    ┌────┼────────────┼────────────────────────────────────────┐          │
│    │    │            │                                        │          │
│    │    └────┬───────┤                                        │          │
│    │         │       │                                        │          │
│    │  ┌──────┴────┐  │                                        │          │
│    │  │gate_proj  │  │  Linear(64+16=80 → 64)                │          │
│    │  └──────┬────┘  │                                        │          │
│    │         │       │                                        │          │
│    ▼         ▼       ▼                                        ▼          │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌──────────────────────────────┐ ┌─────┐     │
│  │  A  │ │  B  │ │  D  │ │           E                   │ │ ToD │     │
│  │Gate │ │ Dir │ │Value│ │         Risk                  │ │Weight│     │
│  │Head │ │Head │ │Head │ │         Head                  │ │Logit│     │
│  ├─────┤ ├─────┤ ├─────┤ ├──────────────────────────────┤ ├─────┤     │
│  │LN   │ │LN   │ │value│ │                              │ │390  │     │
│  │→32  │ │→32  │ │_proj│ │ ┌────────────────────────┐   │ │learn│     │
│  │GELU │ │GELU │ │(80→ │ │ │ ACCOUNT STATE (4 dims) │   │ │able │     │
│  │Drop │ │Drop │ │ 64) │ │ │ [0] growth_ratio       │   │ │     │     │
│  │→2   │ │→6   │ │LN   │ │ │ [1] log_account_size   │   │ │     │     │
│  ├─────┤ ├─────┤ │→32  │ │ │ [2] daily_pnl_frac     │   │ │     │     │
│  │NO_  │ │CALL │ │GELU │ │ │ [3] win_rate_20        │   │ │     │     │
│  │TRADE│ │_ATM │ │Drop │ │ └──────────┬─────────────┘   │ │     │     │
│  │TRADE│ │CALL │ │→1   │ │            │                  │ │     │     │
│  │     │ │_OTM5│ ├─────┤ │  risk_account_proj (4→16)    │ │     │     │
│  │soft │ │CALL │ │MSE  │ │  risk_proj (64+16+16=96→64)  │ │     │     │
│  │max  │ │_OTM │ │on   │ │  risk_head: LN→32→GELU→3     │ │     │     │
│  │→prob│ │ 10  │ │rem  │ ├──────────────────────────────┤ │     │     │
│  │     │ │PUT_ │ │P&L  │ │ stop_pct:  sigmoid [0.15-0.6]│ │     │     │
│  │     │ │ATM  │ │     │ │ size_frac: sigmoid [0-1]     │ │     │     │
│  │     │ │PUT_ │ │     │ │ conviction: tanh [-1, +1]    │ │     │     │
│  │     │ │OTM5 │ │     │ │                              │ │     │     │
│  │     │ │PUT_ │ │     │ │ Account-aware: sees balance, │ │     │     │
│  │     │ │OTM10│ │     │ │ daily P&L, win rate          │ │     │     │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──────────────┬───────────────┘ └──┬──┘     │
│     │       │       │                    │                     │        │
│  Entry/  Strike   Exit                Learned              Time-of-    │
│  Exit    +Dir    Signal              Stop+Size             Day Loss    │
│  Decis.  Select  (val<θ)            +Conviction           Weights     │
│                                                                          │
│  Phase C: Dynamic stop-loss from risk head OR formula fallback          │
│           stop = risk_head[0] mapped to [0.15, 0.60]                    │
│           Fallback: BASE * confidence * iv_factor * vix_factor          │
└──────────────────────────────────────────────────────────────────────────┘

EXIT PRIORITY (identical across train/replay/IBKR):
  1. STOP_LOSS    — price hits dynamic stop level (risk head or formula)
  2. MODEL_EXIT   — gate head says NO_TRADE while holding (bars_held ≥ 2)
  3. VALUE_EXIT   — value head predicts remaining P&L < threshold (0.02)
  4. MAX_HOLD     — held for 390 bars (EOD)
  5. EOD          — end of trading day

TRAILING STOP TIERS (live only):
  +120% unrealized → lock +80%
  +80%  unrealized → lock +50%
  +50%  unrealized → lock +25%
  +30%  unrealized → lock breakeven
```

### How the Phases Work Together

| Phase | Head/Module | What It Decides | Training Signal |
|-------|-------------|-----------------|-----------------|
| **A** | Gate head | Enter trade? Exit trade? | Cross-entropy vs EV-weighted soft labels |
| **B** | Direction head | Which strike + direction? | Cross-entropy vs best-performing option |
| **C** | Dynamic stop (formula/risk) | Where to place stop-loss? | Risk head: MSE on optimal stop; Formula: gate confidence + IV + VIX |
| **D** | Value head | Is remaining upside worth holding? | MSE regression vs actual remaining P&L (clamped [-1, 1]) |
| **E** | Risk head | How much to risk? How confident? | MSE on stop distance + size fraction + conviction signal |

All four heads share the **same transformer backbone** — they see the same 120 bars of 37 features. The position state (7 dims) gives them trade context. The risk head additionally receives account state (4 dims). During a single forward pass, the model produces all outputs simultaneously:

- Gate (A) opens positions. Value (D) can close them early if upside is gone.
- Direction (B) picks the strike. Risk (E) sets the stop distance based on account health.
- Value (D) learns from realized P&L, creating a feedback loop with stops — trades that get stopped teach the value head what "bad remaining P&L" looks like.
- Risk (E) learns position sizing and conviction, adapting to account drawdowns.

### Loss Weights

| Weight | Default | Env Var | What It Controls |
|--------|---------|---------|------------------|
| Gate loss | 0.5 | `TRAIN_GATE_W` | Entry/exit signal quality |
| Direction loss | 2.5 | `TRAIN_DIR_W` | Strike selection accuracy |
| PnL alignment | 0.5 | `TRAIN_PNL_W` | P&L-weighted supervision |
| Exit loss | 0.15 | `TRAIN_EXIT_W` | Exit timing quality |
| Confidence loss | 0.10 | `TRAIN_CONF_W` | Calibration of gate probability |
| Value loss | 0.3 | `TRAIN_VALUE_W` | Remaining P&L prediction |
| Risk loss | 0.2 | `TRAIN_RISK_W` | Stop + size + conviction |

### PBT-Evolvable Parameters

All hyperparameters are readable via `_env_float`/`_env_int` and can be overridden by PBT env vars. Three tiers: loss weights, optimizer params, regularization/sampling. See `tools/inner_loop.py` PBT section for full parameter space.

---

## 8. End-to-End Data & Signal Flow

```
┌─────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌───────────┐
│ Polygon  │   │   IBKR   │   │           │   │          │   │           │
│ S3 Bulk  │   │ Gateway  │   │ prepare.py│   │ data.pt  │   │ train.py  │
│ (SPXW    │──▶│ (SPX,SPY │──▶│ Features  │──▶│ Tensor   │──▶│ on H100   │
│  chains) │   │  VIX)    │   │ + Labels  │   │ Bundle   │   │ GPU       │
└──────────┘   └──────────┘   └───────────┘   └──────────┘   └─────┬─────┘
                                                                     │
                  ┌──────────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────┐   ┌───────────────┐   ┌──────────────────────┐
│                      │   │               │   │                      │
│  inner_loop.py       │──▶│ Sonnet Agent  │──▶│  Mutated train.py   │
│  (experiment runner) │   │ (Code Writer) │   │  (experiment N+1)    │
│                      │   │               │   │                      │
└──────────┬───────────┘   └───────────────┘   └──────────────────────┘
           │
           │  After experiments: best_model.pt + best_train.py
           ▼
┌──────────────────────┐
│  replay.py           │   Validates model on unseen days
│  ─ Bar-by-bar sim    │
│  ─ Trades CSV output │
└──────────┬───────────┘
           │  Model passes replay validation
           ▼
┌──────────────────────┐   ┌───────────────┐   ┌──────────────────────┐
│  live/context.py     │──▶│ live/         │──▶│  live/execution.py   │
│  30-day feature      │   │ decision.py   │   │  OCO orders → IBKR  │
│  context bundle      │   │ 4-head infer  │   │  Paper account       │
└──────────────────────┘   └───────────────┘   └──────────────────────┘
```

---

## 9. ART² Meta-Loop (`tools/art2.py`)

ART² is a two-loop autonomous system. The **outer loop** (Opus) makes strategic decisions grounded in domain knowledge. The **inner loop** (Sonnet agents) optimizes train.py within constraints. See the [operating manual](../../.claude/rules/art2-operating-manual.md) for full lifecycle details.

```
                    ART² TWO-LOOP ARCHITECTURE
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   OUTER LOOP (Claude Opus — strategic brain)                         │
│   ════════════════════════════════════════                            │
│   Owns: ALL files in project                                         │
│   Decides: what to change, when to rebuild, when to fresh-start      │
│   Prohibited: score_config, run_loop.py safety checks                │
│                                                                      │
│   9-PHASE LIFECYCLE                                                  │
│   SETUP → TRAIN → TEARDOWN → ANALYZE → RESEARCH                    │
│   → IMPROVE → DOCUMENT → REVIEW → REPEAT                           │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   INNER LOOP (Claude Sonnet — tactical optimizer)                    │
│   ═══════════════════════════════════════════════                     │
│   Owns: train.py hyperparameters + training dynamics                 │
│   Prohibited: new nn.Module subclasses, architecture changes,        │
│              score formula, new loss terms                           │
│   Each experiment: ~6 min (propose → validate → train → score)       │
│                                                                      │
│   PBT MODE: Population of N members competing per generation         │
│   ─ pbt-init: create population (generalists + specialists)          │
│   ─ pbt-run: train all members, select winners, mutate losers        │
│   ─ pbt-status: inspect current PBT state                            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

```
  art2.py cycle (or daemon --max-cycles N)
       │
       ├── SETUP: preflight, IBKR gate, duration sizing
       ├── TRAIN ───► Akash H100 ───► Sonnet agents via inner_loop.py
       ├── TEARDOWN: deploy.sh stop, download results
       ├── ANALYZE: experiments.v2.jsonl → analysis.json
       ├── REPLAY: backtest best_model.pt on val dates
       ├── RESEARCH: trade-level analysis vs domain knowledge
       ├── VIABILITY: profitability assessment → verdict
       ├── REPORT: briefing.md with research + viability findings
       ├── IMPROVE: Opus reads briefing, applies strategic changes
       ├── DOCUMENT: Update chronicle, notebooks
       └── REVIEW: Human approval gate (REVIEW sentinel file)
```
