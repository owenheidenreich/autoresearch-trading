# autoresearch-trading

Autonomous AI research for day trading model optimization on a single GPU.
Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) —
same concept (AI agent iterates experiments overnight) applied to financial
markets instead of language modeling.

**The idea**: give an AI agent a trading model + backtesting setup and let it
experiment autonomously overnight. It modifies the code, trains for 5 minutes,
checks if Sharpe ratio improved, keeps or discards, and repeats. You wake up
to a log of experiments and (hopefully) a better trading model.

## Quick start

Requirements: Single NVIDIA GPU (optimized for H100), Python 3.10+,
[uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and compute features (one-time, ~1 min)
uv run prepare.py

# 4. Run a single training experiment (~5 min)
uv run train.py
```

If the above works, your setup is ready for autonomous research.

## Running the agent

Spin up Claude, Codex, or your preferred coding agent in this repo, then prompt:

```
Hi have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

The `program.md` file contains the agent's research instructions, domain
knowledge about day trading, and a prioritized list of experiment ideas.

## Project structure

```
prepare.py      — constants, data download, features, evaluation (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions + trading domain knowledge
pyproject.toml  — dependencies
```

## How it works

- **prepare.py** downloads SPY, VIX, and Treasury data via yfinance, computes
  39 market features (returns, volatility, momentum, VIX regime, etc.), normalizes
  with rolling z-scores, and provides a fixed evaluation harness.

- **train.py** contains a causal temporal transformer that takes 60 days of
  features and outputs a position signal (−1 to +1). It trains for exactly 5
  minutes, then evaluates on held-out 2025 data.

- **Metric**: `val_sharpe` — annualized Sharpe ratio on the validation period.
  Higher is better. Includes realistic transaction costs (5 bps round-trip).

- **The agent** modifies only `train.py` — architecture, optimizer, hyperparameters,
  loss function, feature selection, position sizing. Everything is fair game.

## Data

| Source | Ticker | Features derived |
|--------|--------|-----------------|
| SPY    | SPY    | Price returns, volatility, momentum, volume, range |
| VIX    | ^VIX   | Vol regime, fear gauge, term structure proxy |
| 10Y Treasury | ^TNX | Rate level and changes |

Training period: 2010–2024. Validation period: 2025 (held out).

## Design choices

- **Single file to modify**: The agent only touches `train.py`.
- **Fixed time budget**: Always 5 minutes, making experiments comparable.
- **Walk-forward evaluation**: Train on past, evaluate on future. No lookahead.
- **Transaction costs**: Built into evaluation (5 bps per position change).
- **Self-contained**: No external APIs needed at training time.

## Deployment (Akash H100)

This repo is designed to run on an H100 leased from the Akash network:

1. Lease an H100 instance on Akash
2. Clone this repo to the instance
3. Run `uv sync && uv run prepare.py`
4. Point your AI agent at `program.md` and let it run overnight
5. Wake up to ~100 experiments in `results.tsv`

## License

MIT
