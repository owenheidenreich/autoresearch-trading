=# Autoresearch-Trading Architecture

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
│  │  PROMPT CONTROL PLANE (sequential mode only) │                        │
│  │  ┌──────────────┐  ┌──────────────────────┐  │                        │
│  │  │ program.md   │  │ lab_notebook.md      │  │                        │
│  │  │ (strict      │  │ (persistent context: │  │                        │
│  │  │  contract)   │  │  wins + dead ends)   │  │                        │
│  │  └──────┬───────┘  └──────────┬───────────┘  │                        │
│  └─────────┼─────────────────────┼──────────────┘                        │
│            └──────────┬──────────┘                                        │
│                       ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────┐        │
│  │  EXPERIMENT ITERATION (sequential: AI-guided / PBT: blind)   │        │
│  │                                                               │        │
│  │   ┌─────────────┐    1. Read current train.py + history      │        │
│  │   │  Sonnet Agent │◀── 2. System: program.md + lab_notebook   │        │
│  │   │  (Code Writer)│    3. User: metrics + code + prompt      │        │
│  │   │  (SEQ ONLY)  │    (PBT skips this — uses env-var overrides)       │
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
│  ══════ TWO TRAINING MODES ══════                                        │
│                                                                          │
│  MODE 1: SEQUENTIAL WARM-START (default)                                 │
│  ─────────────────────────────────────                                   │
│  inner_loop.py experiment --summary "hypothesis"                         │
│  Each experiment builds on the last kept model. Opus (or Sonnet agent)   │
│  proposes targeted edits to train.py. Score > best → KEEP, else REVERT. │
│  Best for: early convergence, architecture changes, fresh starts,        │
│           debugging, when the training signal itself is being fixed.     │
│                                                                          │
│  MODE 2: PBT (Population-Based Training)                                 │
│  ─────────────────────────────────────                                   │
│  pbt-init → N members (generalists + specialists)                        │
│  pbt-run  → each member trains with env-var overrides                    │
│  Selection: elite carry-forward, exploit top-25%, explore top-50%        │
│  Anti-stagnation: 2 random members after 2 stalled generations           │
│  State: training/.pbt_state.json (resumable)                             │
│  Best for: multi-parameter exploration once the model has converged      │
│           on a stable baseline. Explores loss weights, LR, dropout,     │
│           etc. simultaneously via evolutionary competition.              │
│                                                                          │
│  WHEN TO USE WHICH:                                                      │
│  ─ Fresh start / new architecture → Sequential (model needs to learn    │
│    basic behavior before hyperparameter variants are meaningful)          │
│  ─ Stable baseline, optimizing → PBT (explores parameter space that     │
│    single experiments can't cover efficiently)                            │
│  ─ Score stuck after 5+ sequential experiments → consider PBT            │
│  ─ PBT stagnation (all members similar) → back to sequential with       │
│    architectural or label changes from Opus                              │
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

## 7. Model Architecture (v7 Four-Head)

```
                    TradingModel — v7 Four-Head Architecture
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
│  │  A  │ │  B  │ │  D  │ │           E                  │ │ ToD │     │
│  │Gate │ │ Dir │ │Value│ │         Risk                 │ │Weight│    │
│  │Head │ │Head │ │Head │ │         Head                 │ │Logit│     │
│  ├─────┤ ├─────┤ ├─────┤ ├──────────────────────────────┤ ├─────┤     │
│  │LN   │ │LN   │ │value│ │                              │ │390  │     │
│  │→32  │ │→32  │ │_proj│ │ ┌────────────────────────┐   │ │learn│     │
│  │GELU │ │GELU │ │(80→ │ │ │ ACCOUNT STATE (4 dims) │   │ │able │     │
│  │Drop │ │Drop │ │ 64) │ │ │ [0] growth_ratio       │   │ │     │     │
│  │→2   │ │→6   │ │LN   │ │ │ [1] log_account_size   │   │ │     │     │
│  ├─────┤ ├─────┤ │→32  │ │ │ [2] daily_pnl_frac     │   │ │     │     │
│  │NO_  │ │CALL │ │GELU │ │ │ [3] win_rate_20        │   │ │     │     │
│  │TRADE│ │_ATM │ │Drop │ │ └──────────┬─────────────┘   │ │     │     │
│  │TRADE│ │CALL │ │→1   │ │            │                 │ │     │     │
│  │     │ │_OTM5│ ├─────┤ │  risk_account_proj (4→16)    │ │     │     │
│  │soft │ │CALL │ │BCE  │ │  risk_proj (64+16+16=96→64)  │ │     │     │
│  │max  │ │_OTM │ │on   │ │  risk_head: LN→32→GELU→3     │ │     │     │
│  │→prob│ │ 10  │ │exit │ ├──────────────────────────────┤ │     │     │
│  │     │ │PUT_ │ │label│ │ stop_pct:  sigmoid [0.15-0.6]│ │     │     │
│  │     │ │ATM  │ │     │ │ size_frac: sigmoid [0-1]     │ │     │     │
│  │     │ │PUT_ │ │     │ │ conviction: tanh [-1, +1]    │ │     │     │
│  │     │ │OTM5 │ │     │ │                              │ │     │     │
│  │     │ │PUT_ │ │     │ │ Account-aware: sees balance, │ │     │     │
│  │     │ │OTM10│ │     │ │ daily P&L, win rate          │ │     │     │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──────────────┬───────────────┘ └──┬──┘     │
│     │       │       │                    │                     │        │
│  Entry/  Strike   Exit                Learned              Time-of-    │
│  Exit    +Dir    Signal              Stop+Size             Day Loss    │
│  Decis.  Select  sigmoid(p)>θ      +Conviction           Weights     │
│                                                                          │
│  Phase C: Dynamic stop-loss from risk head OR formula fallback          │
│           stop = risk_head[0] mapped to [0.15, 0.60]                    │
│           Fallback: BASE * confidence * iv_factor * vix_factor          │
└──────────────────────────────────────────────────────────────────────────┘

EXIT PRIORITY (identical across train/replay/IBKR):
  1. STOP_LOSS    — price hits dynamic stop level (risk head or formula)
  2. MODEL_EXIT   — gate head says NO_TRADE while holding (bars_held ≥ 2)
  3. VALUE_EXIT   — v7: sigmoid(value_logit) > conviction-adjusted threshold (0.5-0.7)
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
| **D** | Value head | Should I exit now? (v7: binary classifier) | BCE on sparse exit labels (trailing stop, stall, take-profit signals) |
| **E** | Risk head | How much to risk? How confident? | MSE on stop distance + size fraction + conviction signal |

All four heads share the **same transformer backbone** — they see the same 120 bars of 37 features. The position state (7 dims) gives them trade context. The risk head additionally receives account state (4 dims). During a single forward pass, the model produces all outputs simultaneously:

- Gate (A) opens positions. Value (D) can close them early via sigmoid exit probability.
- Direction (B) picks the strike. Risk (E) sets the stop distance based on account health.
- Value (D) learns from sparse exit labels (v7): trailing stop drop, momentum stall, and take-profit signals. It outputs a logit → sigmoid → exit probability, compared against a conviction-adjusted threshold (0.5 + conviction × 0.2).
- Risk (E) learns position sizing and conviction, adapting to account drawdowns.

### Loss Weights

| Weight | Default | Env Var | What It Controls |
|--------|---------|---------|------------------|
| Gate loss | 0.5 | `TRAIN_GATE_W` | Entry/exit signal quality |
| Direction loss | 2.5 | `TRAIN_DIR_W` | Strike selection accuracy |
| PnL alignment | 0.5 | `TRAIN_PNL_W` | P&L-weighted supervision |
| Exit loss | 0.40 | `TRAIN_EXIT_W` | Exit timing quality (v7: raised, labels now sparse) |
| Confidence loss | 0.10 | `TRAIN_CONF_W` | Calibration of gate probability |
| Value loss | 0.5 | `TRAIN_VALUE_W` | Binary exit classifier (v7: BCE on exit labels) |
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

ART² is a three-tier autonomous system. **Opus** (strategist) makes high-level decisions. **Sonnet agents** (code writers) propose targeted train.py edits. **inner_loop.py** (mechanical layer) executes experiments without AI reasoning. See the [operating manual](../../.claude/rules/art2-operating-manual.md) for full lifecycle details.

```
                    ART² THREE-TIER ARCHITECTURE
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   TIER 1: STRATEGIST (Claude Opus)                                   │
│   ════════════════════════════════                                    │
│   Owns: ALL files in project                                         │
│   Reads: lab_notebook.md, program.md, domain knowledge, briefings    │
│   Decides: what to change, when to rebuild, which training mode,     │
│           when to fresh-start, architectural changes                 │
│   Prohibited: score_config, run_loop.py safety checks                │
│                                                                      │
│   9-PHASE LIFECYCLE                                                  │
│   SETUP → TRAIN → TEARDOWN → ANALYZE → RESEARCH                    │
│   → IMPROVE → DOCUMENT → REVIEW → REPEAT                           │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   TIER 2: CODE WRITER (Claude Sonnet agents)                         │
│   ══════════════════════════════════════════                          │
│   Reads: program.md (contract) + lab_notebook.md (dead ends/wins)    │
│   Receives: specific hypothesis from Opus + current train.py         │
│   Outputs: targeted edits to train.py                                │
│   Prohibited: new nn.Module subclasses, architecture changes,        │
│              score formula, new loss terms                           │
│   Used in: SEQUENTIAL mode only. NOT used in PBT mode.               │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   TIER 3: MECHANICAL LAYER (inner_loop.py)                           │
│   ════════════════════════════════════════                            │
│   No AI reasoning. Executes experiments mechanically:                │
│   validate → upload → train on GPU → parse metrics → score →        │
│   keep/revert. Does NOT read lab_notebook.md or program.md.          │
│                                                                      │
│   SEQUENTIAL MODE: Opus/Sonnet edit train.py, then inner_loop.py     │
│   runs it. Intelligence comes from the tiers above.                  │
│                                                                      │
│   PBT MODE: Purely evolutionary. No AI in the loop.                  │
│   inner_loop.py generates env-var overrides, trains population       │
│   members, selects winners, mutates losers. No Sonnet agent,         │
│   no lab_notebook.md, no program.md — just math.                     │
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

### 9.1 The Daemon (`art2.py daemon`)

The daemon is the top-level orchestrator. It runs as a persistent process (typically in a terminal or background), polling every 5 minutes, and automatically transitions between monitoring mode and training mode based on market state.

**Invocation:**
```bash
python3 tools/art2.py daemon --max-cycles 5 --minutes 100
```

**Startup sequence:**
1. Time check (prints current PT/ET with market status)
2. PID lock — prevents two daemons from running simultaneously (`results/art2/daemon.pid`)
3. SIGTERM handler — allows graceful shutdown
4. Claude CLI resolution — finds the `claude` binary for invoking Opus. If unavailable, runs in "auto-only" mode (can only execute Action A).

#### 9.1.1 Main Loop State Machine

Each iteration of the daemon's `while` loop follows this priority chain. The daemon stops at the first matching condition:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DAEMON MAIN LOOP                                 │
│                     (polls every 5 min idle)                            │
│                                                                         │
│  ┌──── 1. SENTINEL CHECK ────────────────────────────────────────────┐  │
│  │  STOP file exists?    → exit daemon                               │  │
│  │  PAUSED file exists?  → sleep 300s, retry                         │  │
│  │  REVIEW file exists?  → sleep 60s, retry (human must approve)     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              ↓ (no sentinels)                           │
│  ┌──── 2. BUDGET CHECK ─────────────────────────────────────────────┐  │
│  │  Monthly remaining < $20?  → PAUSE (budget_exhausted)             │  │
│  │  Daily spend >= $80?       → PAUSE (daily_spend_cap)              │  │
│  │  Projected cost > 1.5× remaining? → auto-shrink minutes          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              ↓ (budget OK)                              │
│  ┌──── 3. MARKET CHECK ─────────────────────────────────────────────┐  │
│  │  Market OPEN (9:30-16:00 ET)?                                     │  │
│  │    → monitoring mode: check paper trading, sleep 300s             │  │
│  │  Market opens within 120 min?                                     │  │
│  │    → pre-market readiness check, sleep until closer to open       │  │
│  │  Market opens later today?                                        │  │
│  │    → auto-shrink minutes to finish before market - 2hr buffer     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              ↓ (market closed, budget OK)               │
│  ┌──── 4. RUN CYCLE ────────────────────────────────────────────────┐  │
│  │  cmd_cycle(minutes=effective_minutes)                             │  │
│  │    TRAIN → ANALYZE → REPLAY → DIAGNOSE → RESEARCH → VIABILITY   │  │
│  │    → REPORT                                                       │  │
│  │  Failed? → increment consecutive_failures (3 → PAUSE)            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              ↓ (cycle succeeded)                        │
│  ┌──── 5. STRATEGIC DECISION ───────────────────────────────────────┐  │
│  │  compute_recommended_action(analysis.json) → Action A-F           │  │
│  │                                                                   │  │
│  │  High-confidence A ("let it cook")                                │  │
│  │    → auto-decide: no Opus needed, continue immediately            │  │
│  │                                                                   │  │
│  │  Medium A, or any B/C/D/E                                         │  │
│  │    → invoke Opus via Claude CLI with briefing.md + domain docs    │  │
│  │    → Opus returns: action + file edits + rationale                │  │
│  │    → daemon applies the decision automatically                    │  │
│  │                                                                   │  │
│  │  Action F, or Opus defers (needs_human=true)                      │  │
│  │    → PAUSE: alert user, write REVIEW sentinel                     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              ↓                                          │
│           sleep 30s → back to top of loop                               │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 9.1.2 Sentinel Files

Sentinel files in `results/art2/` provide external control over the daemon without killing the process:

| File | Effect | How to trigger | How to resume |
|------|--------|----------------|---------------|
| `STOP` | Graceful exit after current phase | `touch results/art2/STOP` | Restart daemon |
| `PAUSED` | Pause loop, poll every 5 min | Auto-created on budget/failure, or `touch results/art2/PAUSED` | `rm results/art2/PAUSED` |
| `REVIEW` | Pause for human decision review | Auto-created after Opus defers or risky decisions | `rm results/art2/REVIEW` |

The daemon also writes `results/art2/heartbeat.json` on every loop iteration with current state (phase, cycle count, budget remaining). The monitor dashboard (`tools/monitor.py`) reads this file to display daemon status.

#### 9.1.3 Budget Management

The daemon tracks API spending to prevent runaway costs:

- **Monthly budget** (default: tier budget, typically $1000). If remaining < $20, daemon pauses.
- **Daily cap** ($80/day). Prevents a single long session from consuming the month's budget.
- **Auto-sizing**: If the projected cost of `N` minutes of experiments exceeds 1.5× remaining budget, the daemon shrinks the training window (minimum 30 min).
- **Market-aware sizing**: If training would overlap with market hours, minutes are shrunk to finish 2 hours before open.

Cost model: ~$0.20/experiment (Sonnet agent calls). Each experiment takes ~6 min. So 100 minutes ≈ 16 experiments ≈ $3.20 API cost per cycle.

#### 9.1.4 The Decision Tree (`compute_recommended_action`)

After each training cycle, the daemon analyzes `analysis.json` and recommends one of these actions:

| Action | Label | Trigger | Confidence |
|--------|-------|---------|------------|
| **A** | Let it cook | Accept rate ≥ 33%, or ≥ 15% | high / medium |
| **B** | Steer inner loop | Accept rate < 5% after 10+ experiments, or scores trending but not kept | medium / low |
| **C** | Architectural change | (Reserved for Opus — not auto-triggered) | — |
| **D** | Fresh start | (Reserved for Opus — not auto-triggered) | — |
| **E** | Rebuild data | Paper trading P&L diverges significantly from backtest | medium |
| **F** | Fix infrastructure | No experiments, all crashed, all safety-blocked, or gaming detected | high |

The decision tree walks these steps in order:
1. **No data?** → F (infrastructure broken)
2. **All experiments crashed?** → F
3. **Accept rate ≥ 33%?** → A-high (productive, don't intervene)
4. **Gaming detected?** (score-PF divergence) → F
5. **>80% safety-blocked?** → F
6. **Paper trading divergence?** → E
7. **Stalled (<5% accept, 10+ experiments)?** → B (check if scores are trending up)
8. **Moderate (15%+ accept)?** → A-medium
9. **Default** → B-low

#### 9.1.5 Opus Invocation

When the daemon needs strategic intelligence beyond "let it cook," it invokes Opus via the Claude CLI:

```
_invoke_opus(briefing_path, cycle_dir, claude_bin)
```

This call:
1. Reads the system prompt from `tools/opus-prompt.md`
2. Injects domain knowledge from `docs/domain/0dte-domain-knowledge.md` and `docs/domain/pickles-trading-knowledge.md`
3. Includes the cycle's `briefing.md` (generated by REPORT phase)
4. Runs `claude` CLI with the assembled prompt
5. Parses the response for: action taken, files modified, rationale, and whether human review is needed

Opus can make any strategic decision: edit train.py, update lab_notebook.md priorities, switch training modes, trigger a fresh start, or defer to the human. The daemon applies whatever Opus decides, unless `needs_human=true`, in which case it pauses with a REVIEW sentinel.

#### 9.1.6 Failure Handling

- **Cycle failure**: Increments `consecutive_failures`. After 3 consecutive failures, daemon pauses and alerts the user. A successful cycle resets the counter to 0.
- **Opus failure**: If Opus CLI fails (timeout, parse error) but the recommended action was A, the daemon falls back to auto-decide. For non-A actions, it pauses.
- **PID lock**: Prevents two daemons from competing. Lock is always cleaned up in the `finally` block.

#### 9.1.7 Monitoring Mode

When the market is open (9:30-16:00 ET), the daemon does not train. Instead it enters monitoring mode:
- Calls `_monitor_paper_trading()` to check if the paper trader is running and healthy
- Writes heartbeat as "monitoring"
- Sleeps 5 minutes, then re-checks

This ensures training never interferes with live paper trading, and the daemon automatically resumes training after market close.

### 9.2 The Cycle (`cmd_cycle`)

A single cycle is one complete iteration of the research loop. The daemon calls `cmd_cycle()` for each training round.

**Phases executed in order:**

| Phase | Function | Purpose | Output |
|-------|----------|---------|--------|
| TRAIN | `cmd_train()` | Boot Akash GPU, run N experiments via inner_loop.py | `experiments.v2.jsonl`, updated `best_model.pt` |
| ANALYZE | `cmd_analyze()` | Parse experiment results, compute accept rates, score trends | `analysis.json` |
| REPLAY | `cmd_replay()` | Backtest `best_model.pt` on 3 validation dates | `replay/` dir with trade logs |
| DIAGNOSE | `cmd_diagnose()` | Compare train metrics vs replay metrics, detect overfitting | Diagnosis section in briefing |
| RESEARCH | `cmd_research()` | Analyze individual trades against domain knowledge | Research findings |
| VIABILITY | `cmd_viability()` | Profitability verdict (VIABLE / NOT VIABLE) | Verdict + reasoning |
| REPORT | `cmd_report()` | Assemble all findings into `briefing.md` | `briefing.md` |

Each phase can fail independently. If TRAIN fails, the cycle returns `False` to the daemon (which counts it as a failure). Later phases degrade gracefully — if REPLAY fails, the briefing simply notes "no replay data available."

All cycle artifacts are stored in `results/art2/cycle-NNN/` with logs in `cycle-NNN/logs/`.
