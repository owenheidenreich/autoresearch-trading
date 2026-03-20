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
│  │              ~/.cache/autoresearch-trading/                  │   │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌──────────┐  │   │
│  │  │ spy_bars │  │ spx_bars │  │  vix_bars  │  │ spxw_*   │  │   │
│  │  │ .parquet │  │ .parquet │  │  .parquet  │  │ .parquet │  │   │
│  │  └────┬─────┘  └────┬─────┘  └─────┬──────┘  └────┬─────┘  │   │
│  └───────┼──────────────┼──────────────┼──────────────┼────────┘   │
│          └──────────────┴──────┬───────┴──────────────┘            │
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
│  │  targets:   call_pnl, put_pnl, exit labels                 │   │
│  │  prices:    atm/otm5/otm10 call/put price arrays           │   │
│  │  metadata:  dates, timestamps, day boundaries               │   │
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
│  │  │  ─ Translates DecisionIntent → IBKR MarketOrder     │    │      │
│  │  │  ─ Max 1 SPX contract position                       │    │      │
│  │  │  ─ 30% hard stop loss (emergency backstop)          │    │      │
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

## 7. Results & Artifacts Layout

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

## 9. Tooling Overview

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

## 10. Complete Project File Map

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
│   ├── 0dte-domain-knowledge.md       SPX 0DTE trading primer
│   └── pickles-trading-knowledge.md   Trading domain knowledge
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
