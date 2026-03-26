# Documentation Index

## Directory Structure

| Directory | Purpose | When to Read | When to Write |
|-----------|---------|-------------|---------------|
| `domain/` | Trading domain knowledge — 0DTE options mechanics, Greeks, practical trading rules | Diagnosing model behavior, proposing features, evaluating hypotheses | Rarely — only when new domain knowledge is discovered |
| `architecture/` | System design — model architecture, data flow, component diagrams | Understanding how the system works, planning changes | When architecture, pipeline, or data flow changes |
| `operations/` | Procedures and reference — key files, constants, subcommands, IBKR operations | Running the system, debugging issues, looking up constants | When key files, constants, subcommands, or procedures change |
| `journal/` | Ongoing project memory — chronicle (human), notebook (machine) | Understanding project history, what's been tried, current status | Every ART² cycle (append new entries) |
| `ai/` | AI agent configuration — system prompts for programmatic Opus invocations | When modifying how Opus makes strategic decisions | When changing Opus decision framework or response format |

## File Map

### domain/ — Trading Domain Knowledge
- `0dte-domain-knowledge.md` — Greeks behavior, theta decay, gamma dynamics, dealer mechanics, volatility regimes, formulas
- `pickles-trading-knowledge.md` — Practical entry/exit rules, VWAP framework, time-of-day patterns, risk management

### architecture/ — System Design
- `ARCHITECTURE.md` — Full system architecture: data pipeline, training loop, live stack, model heads, infrastructure
- `INNER-LOOP-ARCHITECTURE.md` — Detailed inner loop mechanics: experiment flow, validation, scoring, PBT

### operations/ — Procedures & Reference
- `reference.md` — Single project reference: key files, constants, subcommands, IBKR live trading, daily pipeline, troubleshooting
- `ibkr-trade-analysis-guide.md` — Guide for analyzing IBKR paper trading sessions (event types, signal-to-trade pipeline, metrics)

### journal/ — Project Memory
- `project-chronicle.md` — Human-readable narrative project log (reverse-chronological, for the project owner)
- `art2-notebook.md` — Machine-readable outer loop memory (strategic changes, paper P&L, dead ends, per-cycle)

### ai/ — Agent Configuration
- `art2-opus-system-prompt.md` — System prompt prepended to every programmatic Opus invocation by art2.py
