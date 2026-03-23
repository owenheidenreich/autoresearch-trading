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
│  │  prepare.py  │   │  run_loop.py      │   │  replay.py   │   │  live/         │  │
│  │              │   │  train.py         │   │              │   │  service.py    │  │
│  └──────┬───────┘   └────────┬──────────┘   └──────┬───────┘   └───────┬────────┘  │
│         │                    │                     │                    │           │
│         ▼                    ▼                     ▼                    ▼           │
│  ┌──────────────┐   ┌───────────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │ Polygon S3   │   │ Akash H100 GPU    │   │ Historical   │   │ IBKR Gateway   │  │
│  │ IBKR Gateway │   │ Claude Sonnet API │   │ data.pt      │   │ Paper Trading  │  │
│  └──────────────┘   └───────────────────┘   └──────────────┘   └────────────────┘  │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                         INFRASTRUCTURE (infra/)                             │    │
│  │  deploy.sh ─── boot/start/sync/stop/ssh/logs/status ─── watchdog.sh       │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                         TOOLING (tools/)                                    │    │
│  │  monitor.py │ paper_live.py │ replay_battery.py │ data_quality_report.py   │    │
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
│  │  │                                                       │   │   │
│  │  │  _incremental_update():                               │   │   │
│  │  │  1. Load existing cache                               │   │   │
│  │  │  2. Find max date in cache                            │   │   │
│  │  │  3. Download only (max_date - 1d)..end (IBKR)        │   │   │
│  │  │  4. Concat + deduplicate + save                       │   │   │
│  │  │                                                       │   │   │
│  │  │  Cold start: ~30 min (full 3yr IBKR download)        │   │   │
│  │  │  Daily update: ~30 sec (1 week of new bars)           │   │   │
│  │  └───────────────────────────────────────────────────────┘   │   │
│  │                                                              │   │
│  │  ┌──── PER-DAY (already incremental) ───────────────────┐   │   │
│  │  │  spxw/{date}.pkl      │ spxw_chain/{date}.pkl        │   │   │
│  │  │  ~989 files each      │ per-day option bars           │   │   │
│  │  │                       │                               │   │   │
│  │  │  prefetch_spxw_from_flatfiles():                      │   │   │
│  │  │  Skips days that already have caches. Only downloads  │   │   │
│  │  │  missing days from Polygon S3 flat files.             │   │   │
│  │  └───────────────────────────────────────────────────────┘   │   │
│  │                                                              │   │
│  │  ┌──── AGGREGATE (rebuilt each run) ────────────────────┐   │   │
│  │  │  spxw_full.pkl        │ spxw_chain_full.pkl          │   │   │
│  │  │  Built from per-day caches by load_spxw_caches().    │   │   │
│  │  │  Deleted + rebuilt each prepare.py run (~1 sec).     │   │   │
│  │  └───────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  FEATURE ENGINEERING                         │   │
│  │  compute_features() → 32 features per 1-min bar             │   │
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
│  │  └──────────────┘  └──────────────┘  └───────────────────┘  │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      data.pt (PyTorch tensor)               │   │
│  │  features:  (N_bars, 32) float32                            │   │
│  │  targets:   call_pnl, put_pnl, exit labels, stopped P&L    │   │
│  │  prices:    atm/otm5/otm10 call/put price arrays           │   │
│  │  metadata:  dates, timestamps, day_boundaries               │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Autoresearch Training Loop (`run_loop.py` + `train.py`)

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
│  │   │  Claude API  │◀── 2. System: program.md + lab_notebook   │        │
│  │   │  (Sonnet)    │    3. User: metrics + code + prompt       │        │
│  │   └──────┬───────┘                                            │        │
│  │          │ Returns modified train.py                          │        │
│  │          ▼                                                    │        │
│  │   ┌──────────────┐                                            │        │
│  │   │  VALIDATION  │  ─ AST syntax check                       │        │
│  │   │  GATES       │  ─ Contract compliance (2-head, 32 feat)  │        │
│  │   │              │  ─ Safety: no torch.compile/DDP            │        │
│  │   └──────┬───────┘                                            │        │
│  │          │ Pass                                                │        │
│  │          ▼                                                    │        │
│  │   ┌──────────────────────────────────────────┐                │        │
│  │   │  TRAINING (subprocess, ~4 min budget)     │                │        │
│  │   │  ┌────────────────────────────────────┐   │                │        │
│  │   │  │  train.py on H100                  │   │ ◀── data.pt   │        │
│  │   │  │  ─ Two-head transformer model      │   │                │        │
│  │   │  │  ─ Gate head: [NO_TRADE, TRADE]    │   │                │        │
│  │   │  │  ─ Dir head: [6 option actions]    │   │                │        │
│  │   │  │  ─ Position-aware simulation       │   │                │        │
│  │   │  │  ─ Validation early stopping       │   │                │        │
│  │   │  └────────────────────────────────────┘   │                │        │
│  │   └──────┬───────────────────────────────────┘                │        │
│  │          │ stdout → metrics JSON                              │        │
│  │          ▼                                                    │        │
│  │   ┌──────────────────────────────────────┐                    │        │
│  │   │  SCORING ENGINE                       │                    │        │
│  │   │                                       │                    │        │
│  │   │  score = PF × TradeSharpe × freq     │                    │        │
│  │   │          × penalties × bonuses        │                    │        │
│  │   │                                       │                    │        │
│  │   │  ┌─────────┐ ┌──────────┐ ┌────────┐ │                    │        │
│  │   │  │Profit   │ │Frequency │ │Penalty │ │                    │        │
│  │   │  │Factor   │ │Band      │ │Gates   │ │                    │        │
│  │   │  │         │ │(sweet    │ │─ draws │ │                    │        │
│  │   │  │Trade    │ │ spot)    │ │─ stops │ │                    │        │
│  │   │  │Sharpe   │ │          │ │─ ruin  │ │                    │        │
│  │   │  └─────────┘ └──────────┘ └────────┘ │                    │        │
│  │   └──────┬───────────────────────────────┘                    │        │
│  │          │                                                    │        │
│  │          ▼                                                    │        │
│  │   ┌──────────────────────────┐                                │        │
│  │   │  PROMOTE / REVERT        │                                │        │
│  │   │                          │                                │        │
│  │   │  score > best_score?     │                                │        │
│  │   │  ┌─YES──▶ Keep train.py │──▶ best_model.pt               │        │
│  │   │  │       Update notebook │──▶ best_train.py               │        │
│  │   │  │                       │──▶ promoted/history.jsonl       │        │
│  │   │  └─NO───▶ Revert code   │                                │        │
│  │   └──────────────────────────┘                                │        │
│  │                                                               │        │
│  └───────────────────────┬───────────────────────────────────────┘        │
│                          │ Loop continues (--hours / --max-experiments)    │
│                          ▼                                                │
│  ┌──────────────────────────────────────────────────────────────┐        │
│  │  PREFETCH PIPELINE                                           │        │
│  │  Speculative Claude API call during training → throughput    │        │
│  └──────────────────────────────────────────────────────────────┘        │
│                                                                          │
│  ARTIFACTS PER EXPERIMENT:                                               │
│  results/run-YYYY-MM-DD-HHMMSS/artifacts/exp-N/                         │
│    reasoning.txt │ train_before.py │ train_after_candidate.py            │
│    decision.json │ sanitize_actions.json │ train_output.log              │
│    prompt_system.txt │ prompt_user.txt                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Model Architecture (`train.py`)

```
                         TWO-HEAD TRANSFORMER MODEL
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  INPUT: (batch, seq_len, 32) features                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  32 Features                                                   │  │
│  │  ─ SPY/SPX price features (VWAP, returns, Bollinger, etc.)    │  │
│  │  ─ Volume profile (z-score, relative volume)                  │  │
│  │  ─ VIX/Regime (level, slope, term structure)                  │  │
│  │  ─ Session structure (bar_of_day, time_to_close, power hour)  │  │
│  │  ─ Options greeks (IV, delta, gamma, charm, vanna)            │  │
│  │  ─ Position state (P&L, hold time, bars held)                 │  │
│  └────────────────────────┬───────────────────────────────────────┘  │
│                           ▼                                          │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Input Projection: Linear(32 → D_MODEL=64)                    │  │
│  └────────────────────────┬───────────────────────────────────────┘  │
│                           ▼                                          │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Transformer Encoder (DEPTH=3 layers)                          │  │
│  │  ┌──────────────────────────────────────────────────────────┐  │  │
│  │  │  Layer 1: MultiHeadAttn(N_HEADS) + FFN + LayerNorm      │  │  │
│  │  │           Dropout(0.15)                                  │  │  │
│  │  ├──────────────────────────────────────────────────────────┤  │  │
│  │  │  Layer 2: MultiHeadAttn(N_HEADS) + FFN + LayerNorm      │  │  │
│  │  │           Dropout(0.15)                                  │  │  │
│  │  ├──────────────────────────────────────────────────────────┤  │  │
│  │  │  Layer 3: MultiHeadAttn(N_HEADS) + FFN + LayerNorm      │  │  │
│  │  │           Dropout(0.15)                                  │  │  │
│  │  └──────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────┬───────────────────────────────────────┘  │
│                           │                                          │
│              ┌────────────┴────────────┐                             │
│              ▼                         ▼                              │
│  ┌─────────────────────┐  ┌─────────────────────────┐               │
│  │    GATE HEAD        │  │    DIRECTION HEAD        │               │
│  │  Linear → (batch,2) │  │  Linear → (batch,6)      │               │
│  │  [NO_TRADE, TRADE]  │  │  [CALL_ATM, CALL_OTM5,  │               │
│  │                     │  │   CALL_OTM10, PUT_ATM,   │               │
│  │  Asymmetric loss:   │  │   PUT_OTM5, PUT_OTM10]   │               │
│  │  false entry 2x     │  │                           │               │
│  └──────────┬──────────┘  └────────────┬──────────────┘               │
│             └──────────┬───────────────┘                              │
│                        ▼                                              │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  COMBINED 8 EFFECTIVE ACTIONS                                  │  │
│  │                                                                │  │
│  │  Gate=NO_TRADE + no position  → DO_NOTHING (0)                │  │
│  │  Gate=TRADE    + dir=0..5     → BUY_CALL_ATM (1)              │  │
│  │                                 BUY_CALL_OTM5 (2)             │  │
│  │                                 BUY_CALL_OTM10 (3)            │  │
│  │                                 BUY_PUT_ATM (4)               │  │
│  │                                 BUY_PUT_OTM5 (5)              │  │
│  │                                 BUY_PUT_OTM10 (6)             │  │
│  │  Gate=NO_TRADE + has position → EXIT (7)                      │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. Live Paper Trading Stack (`training/live/`)

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
│      │           │             │                    │                    │
│  ┌───┴──────────────────────────────────────────────┴───────────┐      │
│  │                   COMPONENT DETAIL                            │      │
│  │                                                               │      │
│  │  ┌──────────────────────────────────────────────────────┐    │      │
│  │  │  context.py — LiveContextBundle                      │    │      │
│  │  │  ─ Downloads 30-day historical context (Polygon+IBKR)│    │      │
│  │  │  ─ Computes normalization stats for features         │    │      │
│  │  │  ─ Caches to ~/.cache/autoresearch-trading/          │    │      │
│  │  └──────────────────────────────────────────────────────┘    │      │
│  │                                                               │      │
│  │  ┌──────────────────────────────────────────────────────┐    │      │
│  │  │  features.py — LiveFeatureEngine                     │    │      │
│  │  │  ─ FiveSecondMinuteAggregator (5s bars → 1m bars)   │    │      │
│  │  │  ─ Computes same 32 features as prepare.py           │    │      │
│  │  │  ─ Normalizes using context bundle stats             │    │      │
│  │  │  ─ Returns LiveFeatureSnapshot each minute           │    │      │
│  │  └──────────────────────────────────────────────────────┘    │      │
│  │                                                               │      │
│  │  ┌──────────────────────────────────────────────────────┐    │      │
│  │  │  decision.py — ModelDecisionEngine                   │    │      │
│  │  │  ─ Loads best_model.pt via replay.load_model()       │    │      │
│  │  │  ─ Runs inference: features → gate + direction probs │    │      │
│  │  │  ─ Emits DecisionIntent (entry) or RiskUpdateIntent  │    │      │
│  │  │  ─ Position-aware: tracks P&L, hold time, bars held  │    │      │
│  │  │  ─ Min trade probability gate (default 0.55)         │    │      │
│  │  └──────────────────────────────────────────────────────┘    │      │
│  │                                                               │      │
│  │  ┌──────────────────────────────────────────────────────┐    │      │
│  │  │  resolver.py — SPXWContractResolver                  │    │      │
│  │  │  ─ Resolves SPXW 0DTE option contracts from IBKR    │    │      │
│  │  │  ─ Maps actions → (right, strike_offset)             │    │      │
│  │  │  ─ ATM, OTM+5, OTM+10 for calls & puts             │    │      │
│  │  └──────────────────────────────────────────────────────┘    │      │
│  │                                                               │      │
│  │  ┌──────────────────────────────────────────────────────┐    │      │
│  │  │  execution.py — OCOExecutionEngine                   │    │      │
│  │  │  ─ Translates DecisionIntent → IBKR LimitOrder      │    │      │
│  │  │  ─ Max 1 SPX contract position                       │    │      │
│  │  │  ─ Dynamic stop loss (15-60%, confidence+IV+VIX)    │    │      │
│  │  │  ─ Kill switch file support                          │    │      │
│  │  │  ─ Daily loss limit check                            │    │      │
│  │  │  ─ Audit trail → results/live/audit.jsonl            │    │      │
│  │  │  ─ Dry-run mode (no actual orders)                   │    │      │
│  │  └──────────────────────────────────────────────────────┘    │      │
│  │                                                               │      │
│  │  ┌──────────────────────────────────────────────────────┐    │      │
│  │  │  contracts.py — Typed Contracts                      │    │      │
│  │  │  ─ DecisionIntent    (model → execution)             │    │      │
│  │  │  ─ RiskUpdateIntent  (stop-loss updates)             │    │      │
│  │  │  ─ ExecutionState    (position tracking)             │    │      │
│  │  │  ─ LiveContextBundle (feature normalization)         │    │      │
│  │  │  ─ FeatureContractVersion (schema validation)        │    │      │
│  │  └──────────────────────────────────────────────────────┘    │      │
│  └───────────────────────────────────────────────────────────────┘      │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │  IBKRMarketStream (service.py)                               │      │
│   │  ─ 5-sec real-time bars: SPX, SPY, VIX                      │      │
│   │  ─ Option top-of-book + Greeks for 6 SPXW contracts         │      │
│   │  ─ Aggregates to 1-min bars via FiveSecondMinuteAggregator  │      │
│   └──────────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Infrastructure & Deployment (`infra/`)

```
              AKASH GPU DEPLOYMENT LIFECYCLE
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   LOCAL MACHINE                          AKASH H100 CONTAINER   │
│   ══════════════                         ════════════════════    │
│                                                                  │
│   deploy.sh boot                                                │
│   ┌──────────────┐    Akash blockchain    ┌──────────────────┐  │
│   │ Create SDL   │───────────────────────▶│ H100/A100 GPU    │  │
│   │ deployment   │    (bid → lease)       │ 64GB RAM         │  │
│   │ Wait for SSH │◀──────────────────────│ PyTorch 2.5.1    │  │
│   └──────┬───────┘                        │ Ubuntu + SSH     │  │
│          │                                └──────────────────┘  │
│          ▼                                                       │
│   deploy.sh start                                                │
│   ┌──────────────┐     SCP upload          ┌─────────────────┐  │
│   │ Upload:      │────────────────────────▶│ /root/          │  │
│   │ ─ train.py   │  training/*.py          │ ├── train.py    │  │
│   │ ─ run_loop.py│  + data.pt              │ ├── run_loop.py │  │
│   │ ─ prepare.py │  + program.md           │ ├── prepare.py  │  │
│   │ ─ data.pt    │  + lab_notebook.md      │ ├── data.pt     │  │
│   │ ─ program.md │  + best_model.pt        │ ├── program.md  │  │
│   └──────┬───────┘                         │ └── results/    │  │
│          │                                 └────────┬────────┘  │
│          │          SSH: start_loop.sh               │           │
│          │────────────────────────────────▶ run_loop.py starts  │
│          │                                           │           │
│          ▼                                           │           │
│   deploy.sh sync (auto, every 30s)                   │           │
│   ┌──────────────┐     rsync download     ┌─────────┴────────┐ │
│   │ results/     │◀───────────────────────│ results/run-*/   │  │
│   │ run-YYYY-*/  │  experiments.v2.jsonl  │ experiments.jsonl │  │
│   │              │  artifacts/            │ artifacts/exp-*/  │  │
│   └──────┬───────┘                        │ status.json      │  │
│          │                                └──────────────────┘  │
│          ▼                                                       │
│   ┌──────────────┐                                               │
│   │ monitor.py   │  Rich terminal dashboard                     │
│   │ (local)      │  ─ GPU utilization, memory, temp             │
│   │              │  ─ Experiment progress, scores                │
│   │              │  ─ Best model tracker                         │
│   └──────────────┘                                               │
│                                                                  │
│   deploy.sh stop                                                │
│   ┌──────────────┐     SSH kill + final    ┌─────────────────┐  │
│   │ Download     │◀───sync + close lease──│ Kill run_loop   │  │
│   │ final results│                        │ Close deployment │  │
│   └──────────────┘                        └─────────────────┘  │
│                                                                  │
│   OTHER COMMANDS:                                                │
│   ─ deploy.sh ssh    → Drop into remote shell                   │
│   ─ deploy.sh logs   → Tail loop stdout                         │
│   ─ deploy.sh status → GPU stats + run progress                 │
│   ─ watchdog.sh      → GPU health + process monitor             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Feature Parity Verification (confirmed 2026-03-20)

```
              DATA FLOW PARITY: TRAINING = REPLAY = LIVE
┌──────────────────────────────────────────────────────────────────┐
│                                                                    │
│  All three stages use the SAME feature pipeline from prepare.py:   │
│  ─ compute_features()                  → 32 FEATURE_NAMES          │
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
│              Same 32 features, same normalization                   │
│                                                                    │
│  LIVE VERIFICATION RESULTS (paper account DUP440540):              │
│  ─ 29/32 features present at session start                         │
│  ─ ret_6, ret_12, volume_at_price_pctile fill after ~30 min       │
│  ─ Option data staleness: <1 second                                │
│  ─ All 6 SPXW 0DTE contracts resolved with conIds                 │
│  ─ Order fill: SPXW 6510C bought @ $7.30 LMT, sold @ $4.50 MKT   │
│  ─ Model inference: 6 runs, gate_prob 3.5-4.6%, correct NO_TRADE  │
│                                                                    │
│  BUGS FIXED:                                                       │
│  ─ service.py:85   b.open → b.open_ (ib_insync keyword conflict)  │
│  ─ decision.py:215 MKT → LMT entry (IBKR rejects MKT on SPXW)    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 8. Results & Artifacts Layout

```
               CANONICAL RESULTS STRUCTURE
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  results/                                                    │
│  ├── current_run.txt          ◀── pointer to active run      │
│  ├── sync.log                 ◀── rsync download log         │
│  │                                                           │
│  ├── promoted/                ◀── CROSS-RUN BEST MODELS      │
│  │   ├── current.txt              (survives run restarts)    │
│  │   └── history.jsonl            promotion events log       │
│  │                                                           │
│  ├── run-2026-03-20-045017/   ◀── ONE TRAINING RUN           │
│  │   ├── run_metadata.json        config snapshot            │
│  │   ├── status.json              live loop state            │
│  │   ├── experiments.v2.jsonl     structured experiment log  │
│  │   ├── data_quality_report.json data integrity check       │
│  │   │                                                       │
│  │   └── artifacts/                                          │
│  │       ├── exp-1/           ◀── PER-EXPERIMENT ARTIFACTS   │
│  │       │   ├── reasoning.txt        Claude's hypothesis    │
│  │       │   ├── prompt_system.txt    system prompt sent     │
│  │       │   ├── prompt_user.txt      user prompt sent       │
│  │       │   ├── train_before.py      code before mutation   │
│  │       │   ├── train_after_candidate.py  proposed code     │
│  │       │   ├── train_output.log     training stdout/stderr │
│  │       │   ├── decision.json        keep/revert + score    │
│  │       │   ├── sanitize_actions.json code safety checks    │
│  │       │   ├── experiment.v2.json   full structured record │
│  │       │   └── program.md           contract snapshot      │
│  │       │                                                   │
│  │       └── exp-2/                                          │
│  │           └── ...                                         │
│  │                                                           │
│  └── analysis/                ◀── CROSS-RUN ANALYSIS         │
│      ├── evidence.parquet         ingested experiment data    │
│      ├── incidents.jsonl          anomaly/incident log        │
│      └── decision-digest-*.md     daily decision summary     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. End-to-End Data & Signal Flow

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
│  run_loop.py         │──▶│ Claude Sonnet │──▶│  Mutated train.py   │
│  (orchestrator)      │   │ API           │   │  (experiment N+1)    │
│                      │   │               │   │                      │
└──────────┬───────────┘   └───────────────┘   └──────────────────────┘
           │                                              │
           │  After 100s of experiments:                  │
           │  best_model.pt + best_train.py promoted      │
           ▼                                              │
┌──────────────────────┐                                  │
│  replay.py           │◀─────────────────────────────────┘
│  ─ Historical days   │   Validates model on unseen days
│  ─ Bar-by-bar sim    │
│  ─ Trades CSV output │
└──────────┬───────────┘
           │  Model passes replay validation
           ▼
┌──────────────────────┐   ┌───────────────┐   ┌──────────────────────┐
│                      │   │               │   │                      │
│  live/context.py     │──▶│ live/         │──▶│  live/execution.py   │
│  30-day feature      │   │ decision.py   │   │  OCO orders → IBKR  │
│  context bundle      │   │ Model infer   │   │  Paper account       │
│                      │   │               │   │                      │
└──────────────────────┘   └───────────────┘   └──────────────────────┘
         ▲                        ▲
         │                        │
┌────────┴────────┐   ┌──────────┴──────────┐
│ IBKR + Polygon  │   │ live/features.py    │
│ Historical data │   │ 5s bars → 1m bars   │
│ (context window)│   │ Same 32 features    │
└─────────────────┘   │ as training         │
                      └─────────────────────┘
```

---

## 10. ART² Meta-Loop (`tools/art2.py`)

ART² is a two-loop autonomous system. The **outer loop** (Opus) makes strategic decisions grounded in domain knowledge. The **inner loop** (Sonnet) optimizes train.py within the constraints set by the outer loop. ART² has full authority over every file in the project.

```
                    ART² TWO-LOOP ARCHITECTURE
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   OUTER LOOP (Claude Opus — strategic brain)                         │
│   ════════════════════════════════════════                            │
│                                                                      │
│   ┌────────────┐  ┌──────────────────┐  ┌─────────────────────────┐ │
│   │ Domain     │  │ Research Phase   │  │ Strategic Decision      │ │
│   │ Knowledge  │  │ (trade-level     │  │ (1 change per cycle)    │ │
│   │ 0DTE mech. │──▶  analysis vs    │──▶                         │ │
│   │ Pickles    │  │  domain rules)   │  │ Actions A-G:            │ │
│   └────────────┘  └──────────────────┘  │ A) Let it cook          │ │
│                                          │ B) Steer inner loop     │ │
│   Owns: ALL files in project             │ C) Change constraints   │ │
│   Decides: what to change, when to       │ D) Change features      │ │
│   rebuild data, when to fresh-start      │ E) Rebuild data         │ │
│   Prohibited: score_config,              │ F) Fix infrastructure   │ │
│   run_loop.py safety checks              │ G) Modify train.py      │ │
│                                          └───────────┬─────────────┘ │
│                                                      │               │
│   7-PHASE LIFECYCLE                                  │               │
│   ┌──────┬───────┬──────────┬─────────┬──────────┬───┴────┬───────┐ │
│   │SETUP │TRAIN│TEARDOWN│ANALYZE│RESEARCH│IMPROVE│DOCUMENT│REPEAT│ │
│   └──────┴───────┴──────────┴─────────┴──────────┴────────┴───────┘ │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   INNER LOOP (Claude Sonnet — tactical optimizer)                    │
│   ═══════════════════════════════════════════════                     │
│                                                                      │
│   Runs on Akash H100 via run_loop.py                                 │
│   Owns: train.py hyperparameters + training dynamics                 │
│   May: tune _env_float values, loss weight scheduling,               │
│        sample weighting, LR schedules, bias init, feature noise      │
│   Prohibited: new nn.Module subclasses, new loss terms,              │
│              architecture changes, score formula                     │
│                                                                      │
│   Each experiment: Claude proposes mutation → validate → train       │
│   (~4 min) → score → keep/revert. ~6 min per experiment.            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

```
  art2.py cycle (or daemon --max-cycles N)
       │
       ├── SETUP: preflight, IBKR gate, duration sizing
       │
       ├── TRAIN ───► Akash H100 ───► run_loop.py (Sonnet)
       │                                  │
       ├── TEARDOWN: deploy.sh stop ◄─────┘
       │
       ├── ANALYZE: parse experiments.v2.jsonl → analysis.json
       │
       ├── REPLAY: backtest best_model.pt on val dates
       │
       ├── DIAGNOSE: compare training vs replay metrics
       │
       ├── RESEARCH: trade-level analysis vs domain knowledge
       │              → time-of-day, exits, strikes, direction bias
       │              → hypotheses grounded in 0DTE mechanics
       │
       ├── REPORT: briefing.md with research findings
       │
       └── IMPROVE: Opus reads briefing, makes 1 strategic change
                    via file_edits JSON (whitelisted paths)
```

### Full Autonomy (Daemon Mode)

The daemon runs continuously. Opus outputs file edits in JSON that are applied with safety checks against a whitelisted path set. Opus can return `needs_human: true` to pause if genuinely uncertain.

```
  ┌─────────────────────────────────────────────────────┐
  │ High-confidence A     → auto-decide (no Opus call)  │
  │ All other actions     → invoke Opus for decision     │
  │ Opus returns edits    → apply to whitelisted files   │
  │ rebuild_data: true    → runs prepare.py              │
  │ fresh_start: true     → moves best_model.pt to .bak  │
  │ needs_human: true     → pauses for human review      │
  └─────────────────────────────────────────────────────┘
```

### Domain Knowledge Injection

Opus receives both domain knowledge files (~25K tokens) injected directly into its prompt, since `claude -p` mode cannot read files. This enables domain-grounded reasoning about theta decay, gamma dynamics, time-of-day patterns, VIX regimes, and exit timing.

### IBKR Compatibility Gate (4 checks)

Runs before and after every training cycle:
1. **model_exists** — `best_model.pt` present + hash
2. **model_checkpoint** — Gate head outputs=2, Direction head outputs=6
3. **feature_parity** — data.pt has 32 features + `val_start_idx`
4. **ibkr_probe** — IBKR TWS connectivity and market data availability

### Decision Tree (Automated Recommendations)

| Priority | Condition | Action |
|----------|-----------|--------|
| 1 | Train/eval mismatch (PF diverges >20%) | Fix mismatch |
| 2 | Accept rate ≥33% | A) Let it cook |
| 3 | Gaming detected (score up, PF down) | F) Fix infrastructure |
| 4 | >80% safety-blocked | F) Fix infrastructure |
| 5 | Paper trading divergence >30% | E) Rebuild data |
| 6 | Stall (<5% accept, scores flat) | B) Steer inner loop |
| 7 | Accept rate 15%+ | A) Let it cook |
| 8 | Default | B) Steer inner loop |

### API Budget Tracking

- Tier 3: $1,000/month limit
- Inner loop: ~$0.20/experiment (Sonnet 4), ~6 min/experiment on H100
- Outer loop: ~$0.38/Opus call (with domain knowledge injection)
- Pre-flight budget check before each training session
- Spend tracked in `~/.cache/autoresearch-trading/api_spend_tracker.json`

---

## 11. Tooling Overview (see also [art2.md](art2.md))

```
                         TOOLS ECOSYSTEM
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  MONITORING & DASHBOARDS                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  monitor.py          Rich terminal UI, auto-refresh      │   │
│  │  ─ Remote: SSH into H100, GPU stats, loop progress       │   │
│  │  ─ Local: reads synced results/                           │   │
│  │  ─ Experiment table, score trends, phase indicators       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  LIVE TRADING                                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  paper_live.py       CLI entry: connects all live/ mods   │   │
│  │  ─ --context-only    Refresh market context bundle only   │   │
│  │  ─ --dry-run         Full decision loop, no orders        │   │
│  │  ─ --max-minutes     Safety cap for test sessions         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  VALIDATION                                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  replay_battery.py   Multi-day replay validation suite    │   │
│  │  ─ Runs replay.py across date ranges                      │   │
│  │  ─ Aggregates trade stats and P&L curves                  │   │
│  │                                                           │   │
│  │  live_feature_parity_report.py                            │   │
│  │  ─ Verifies live features match training features         │   │
│  │                                                           │   │
│  │  live_order_parity_report.py                              │   │
│  │  ─ Verifies live orders match model signals               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  DATA QUALITY                                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  data_quality_report.py   Validates data.pt integrity     │   │
│  │  ingest_evidence.py       Ingests experiment results      │   │
│  │                           → evidence.parquet + incidents  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  IBKR DIAGNOSTICS                                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  ib_account_snapshot.py   Account status check            │   │
│  │  ib_probe.py              Connection + market data probe  │   │
│  │  ib_entitlements.py       Data subscription verification  │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 12. Complete Project File Map

```
autoresearch-trading/
├── training/                          ═══ CORE ENGINE ═══
│   ├── prepare.py                     Data pipeline: bars + options → data.pt
│   ├── train.py                       Neural net model (AGENT MODIFIES THIS)
│   ├── run_loop.py                    Autoresearch orchestrator (Claude API)
│   ├── replay.py                      Historical replay simulation
│   ├── program.md                     Strict contract (injected in prompt)
│   ├── lab_notebook.md                Persistent experiment memory
│   ├── __init__.py
│   └── live/                          ═══ LIVE TRADING STACK ═══
│       ├── service.py                 Session orchestrator (IBKR connection)
│       ├── decision.py                Model inference → intents
│       ├── execution.py               OCO order engine (entry + stop)
│       ├── context.py                 30-day context bundle builder
│       ├── features.py                Real-time 5s→1m feature engine
│       ├── resolver.py                SPXW contract resolution
│       ├── contracts.py               Typed data contracts
│       ├── entitlements.py            IBKR data entitlement check
│       └── __init__.py
│
├── tools/                             ═══ TOOLING ═══
│   ├── art2.py                        ART² outer loop orchestrator (meta-loop)
│   ├── daily_pipeline.py              Pre-market data refresh + weekly retrain
│   ├── paper_live.py                  CLI entry for paper trading
│   ├── monitor.py                     Rich terminal dashboard
│   ├── replay_battery.py              Multi-day replay suite
│   ├── data_quality_report.py         data.pt validation
│   ├── ingest_evidence.py             Experiment data ingestion
│   ├── live_feature_parity_report.py  Feature parity check
│   ├── live_order_parity_report.py    Order parity check
│   ├── ib_account_snapshot.py         IBKR account probe
│   ├── ib_entitlements.py             IBKR entitlement probe
│   ├── ib_probe.py                    IBKR connection probe
│   └── run_ibkr_mock_training.py      Mock IBKR training tool
│
├── infra/                             ═══ INFRASTRUCTURE ═══
│   ├── deploy.sh                      Akash GPU lifecycle manager
│   ├── deploy-autoresearch.yaml       Akash SDL (H100/A100 spec)
│   ├── start_loop.sh                  Remote loop launcher
│   └── watchdog.sh                    GPU health monitor
│
├── tests/                             ═══ TEST SUITE ═══
│   ├── test_ibkr_mock_training.py
│   ├── test_live_contracts.py
│   ├── test_live_decision.py
│   ├── test_live_execution.py
│   ├── test_live_feature_parity_report.py
│   ├── test_live_order_parity_report.py
│   ├── test_live_resolver.py
│   ├── test_observability_upgrade.py
│   ├── test_replay_battery.py
│   ├── test_replay_ledger_qa.py
│   └── test_replay_loader.py
│
├── docs/                              ═══ DOCUMENTATION ═══
│   ├── CLAUDE.md                      Codebase guide for Claude
│   ├── ARCHITECTURE.md                System architecture diagrams (this file)
│   ├── art2.md                        ART² subcommand reference
│   ├── art2-notebook.md               ART² outer loop strategic memory
│   ├── art2-opus-system-prompt.md     Prompt injected into Opus calls
│   ├── daily-pipeline.md              Daily automation pipeline docs
│   ├── 0dte-domain-knowledge.md       SPX 0DTE trading primer
│   └── pickles-trading-knowledge.md   Trading domain knowledge
│
├── .claude/rules/                     ═══ AUTO-LOADED RULES ═══
│   └── art2-operating-manual.md       ART² lifecycle, roles, policies
│
├── results/                           ═══ RUNTIME OUTPUT ═══
│   ├── current_run.txt
│   ├── promoted/
│   ├── run-YYYY-MM-DD-HHMMSS/
│   └── analysis/
│
├── archive/                           ═══ LEGACY ═══
│   └── legacy-runtime/
│
├── pyproject.toml
├── uv.lock
├── .env
└── .gitignore
```
