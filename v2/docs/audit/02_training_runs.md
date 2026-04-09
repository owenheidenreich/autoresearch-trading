# Section 2: Training Runs

## Scope
Model architecture, loss function, hyperparameters, training loop execution, GPU orchestration, and the mutable research surface that the ART2 loop iterates on.

## Critical Files

| File | Role | Lines | Mutable? |
|------|------|------:|----------|
| `v2/train.py` | Model definition (Transformer + FiLM conditioning), loss function (P&L regression + risk regularization), training loop, hyperparameters via env vars. The primary mutable research surface. | 523 | Yes (research) |
| `v2/core/policy.py` | DecisionPolicy dataclass. Gate threshold, risk output ranges (stop/target/hold), cooldown bars, time blocks, daily loss cap, order style, exit policy. Second mutable research surface. | 83 | Yes (research) |
| `v2/core/schema.py` | TradeIntent and SimulatedTrade contracts. The atomic unit connecting training predictions to trade execution. | 206 | No (harness) |
| `v2/ops/run_experiment.py` | Single experiment runner. Calls train_model(), replay_validation(), computes baselines, returns score. Entry point for GPU execution. | 210 | No (infra) |
| `v2/ops/deploy.sh` | Akash GPU deployment orchestrator. Commands: boot, start, run_one, ssh, status, download, stop. Manages H100 lifecycle, code upload, result download. | 921 | No (infra) |
| `v2/ops/fast_sweep.py` | Fast inline hyperparameter sweep. Trains multiple configs without subprocess overhead. | 356 | No (infra) |
| `v2/ops/gpu_sweep.py` | GPU hyperparameter grid sweep. SweepConfig for epochs, batch_size, LR, d_model, depth, dropout. | 235 | No (infra) |
| `v2/ops/preflight.py` | Pre-flight GPU check. Verifies CUDA, data.pt, train.py presence. | 21 | No (infra) |

## Data Flow

```
data.pt (from Market Data section)
    |
    v
train.py
    |  loads: features, labels, option prices from data.pt
    |  constructs: TradeDataset with LOOKBACK window batching
    |  architecture: PositionalEncoding -> Transformer -> FiLM -> regression heads
    |  predicts: call_pnl, put_pnl (dual-direction P&L regression)
    |  loss: PNL_W * pnl_loss + RISK_W * risk_loss
    |  optimizer: AdamW (LR, weight_decay, time_budget)
    |  writes: model_candidate.pt
    |
    v
run_experiment.py (on GPU)
    |  calls: train.py -> model_candidate.pt
    |  calls: replay.py -> score
    |  compares: score vs 4 baselines
    |  prints: score to stdout
    |
    v
deploy.sh run_one exp_NNN
    |  uploads: v2/ code to GPU
    |  runs: run_experiment.py remotely
    |  downloads: model_candidate.pt to local
    |  prints: score to caller (Claude)
```

## Key Interfaces

**Inputs:**
- `data.pt` -- features, labels, masks, option prices
- Hyperparameters via environment variables: TRAIN_LOOKBACK, TRAIN_D_MODEL, TRAIN_DEPTH, TRAIN_DROPOUT, TRAIN_BATCH_SIZE, TRAIN_LR, TRAIN_WEIGHT_DECAY, TRAIN_EPOCHS, TIME_BUDGET, WEIGHT_PNL, WEIGHT_RISK
- `core/policy.py` -- DecisionPolicy parameters that shape trade decisions at inference

**Outputs:**
- `model_candidate.pt` -- trained model checkpoint (weights + architecture metadata)
- Score printed to stdout (consumed by ART2 loop for keep/revert decision)

**Model architecture details:**
- PositionalEncoding for temporal bar sequences
- FiLMLayer (Feature-wise Linear Modulation) for regime-conditional gating
- Transformer encoder (configurable depth, heads, d_model)
- Dual regression heads: predict call P&L and put P&L
- Gate decision: max(pred_call_pnl, pred_put_pnl) > threshold
- Direction: argmax of call vs put prediction

## Dependencies on Other Sections

| Section | Dependency |
|---------|------------|
| Market Data | Consumes `data.pt` (features, labels, prices) |
| Validation | `run_experiment.py` calls `replay.py` to score each trained model |
| ART2 Pipeline | `deploy.sh run_one` wraps the full train+score cycle; ART2 loop decides keep/revert |
| IBKR Paper Trading | Trained `model.pt` is loaded by live decision engine |

## Audit Surface Area

- Model architecture: does the Transformer + FiLM design match the problem structure?
- Loss function: P&L regression + risk regularization -- are the weights appropriate?
- Training from scratch: every experiment starts with random init (no warm-starting) -- is this enforced?
- Hyperparameter configuration: env vars vs hardcoded defaults -- which take precedence?
- Time budget: training stops at TIME_BUDGET seconds -- does this cause premature stopping?
- Gate threshold: how is the trade/no-trade decision derived from P&L predictions?
- DecisionPolicy: are policy parameters consistent between training labels and inference?
- GPU orchestration: does deploy.sh correctly upload code, run training, download results?

---

## Audit Questions -- Direct Improvements

**1. Risk head has no activation -- model wastes capacity learning to stay in bounds.**
`train.py:160-163` outputs raw unbounded values for stop/target/hold. `replay.py:112-115` clamps these to policy ranges (stop: 0.10-0.50, target: 0.15-1.65, hold: 0.0-1.0). The model gets no gradient signal when outputs land outside ranges. Adding sigmoid/tanh activations matched to target ranges would focus capacity on prediction quality, not range-finding.

**2. `NUM_FEATURES` is set from env var with no runtime validation against `data.pt`.**
`train.py:31`: `NUM_FEATURES = int(os.environ.get("NUM_FEATURES", 47))`. The model's `input_proj` is sized to this value but `train()` never checks `features.shape[-1] == NUM_FEATURES`. If the pipeline changes feature count, the model crashes with an inscrutable dimension mismatch. One-line assertion fix.

**3. `deploy.sh start` uploads model.pt as a "warm-start" (line 428-432) contradicting the "train from scratch" rule.**
`deploy.sh:428-432` says "Upload model.pt if it exists -- warm-start from previous training run." But `train.py:415` always creates a fresh model with `TradingModel().to(device)`. The uploaded model.pt is immediately overwritten. This comment and upload should be removed to enforce the "every experiment trains from scratch" rule in `program.md`.

**4. Direction uses `softmax` on raw P&L regression outputs -- semantically wrong.**
`train.py:192-193` emits `direction = [call_pnl, put_pnl]` as raw P&L values. `replay.py:93` applies `torch.softmax(outputs['direction'], dim=-1)`. Softmax on regression values is meaningless -- [0.3, 0.1] and [3.0, 1.0] produce different softmax distributions. The argmax still picks correctly, but if `dir_probs` is used for confidence/sizing, the values are wrong. Replace with simple comparison.

**5. `daily_loss_cap_pct` (5%) is defined in policy.py and enforced in simulator.py but absent from evaluator docs.**
`policy.py:46` defines a 5% daily loss cap. `simulator.py:285` enforces it. `evaluator.md` never mentions it. The score appears better than it should if daily losses are silently capped. Either document it or examine whether it distorts comparisons.

**6. Confidence output uses arbitrary 3.0x multiplier then sigmoid, creating a narrow effective range.**
`train.py:200`: `confidence = torch.abs(call_v - put_v) * 3.0`. Then sigmoid compresses most predictions to 0.55-0.82. The 3.0x constant was chosen arbitrarily. If confidence is used for filtering or sizing, this compression matters.

**7. Strike selection is hardcoded to ATM with no learning mechanism -- dead infrastructure.**
`train.py:195-197` outputs a fixed one-hot vector at center index (ATM). `replay.py:98-100` applies softmax+argmax on this fixed output, always selecting ATM. The 13-strike grid (ATM +/- 30pt) exists in the data but is completely unused. For 0DTE, strike selection materially affects P&L due to gamma/theta tradeoffs.

## Audit Questions -- Deeper Planning

**8. Is P&L magnitude the right gate signal for trade/no-trade?**
The gate (`train.py:188-190`, `replay.py:83-85`) trades whenever max predicted P&L > 0 (sigmoid(0)=0.5, threshold=0.5). There is no explicit loss term teaching which bars are good to trade. A bar with predicted P&L +0.01 (barely above zero) is treated the same as +0.50. For 0DTE where spread costs alone are 1-3%, many "positive P&L" predictions are losers after costs. Should the gate have its own training signal, or should the threshold account for expected spread cost?

**9. FiLM regime encoder uses only the last bar's 47 features -- is this sufficient for regime conditioning?**
`train.py:168`: `regime = self.regime_encoder(x[:, -1, :])`. The regime is derived from a single snapshot. But market regime (trending vs mean-reverting, high vs low vol) is a multi-bar concept. The transformer already processes the 30-bar sequence, so the regime encoder sees strictly less information than the transformer output it modifies. Consider feeding pooled sequence representation or explicit regime features instead.

**10. Labels are generated with TRAILING exit policy but the model cannot learn WHEN to trail vs hold.**
`relabel_tier3.py:68-75` uses trailing stop tiers when computing label P&L. The label is a single final P&L number. The model predicts this number and replay re-simulates with trailing stops. The model is trained to predict an outcome that depends on an exit strategy it has no control over. A trade hitting +80% then trailing to +50% has a label of +50%, but a naive stop-target exit might capture +80%.

**11. Asymmetric loss weights (4x calls, 6x puts for false optimism) may systematically suppress the gate.**
`train.py:279-286`: loss multiplies errors by 4x/6x when model predicts profit on actual loss, but 1x for the reverse. This creates a strong incentive to predict negative P&L (pessimistic = safe). Combined with the gate logic, this could cause systematic under-prediction, leading to fewer trades than optimal. The "gate stuck at 79%" observation may be a direct consequence.

**12. The causal mask + FiLM bypass: is the regime encoder a leak around the transformer's causal processing?**
`train.py:168` feeds raw last-bar features to the regime encoder. `train.py:176` processes the sequence with a causal mask. `train.py:179` applies FiLM to modulate transformer output with the regime encoding. The regime encoder operates on raw features while the transformer operates on projected/normalized features. The regime encoder has an unmasked bypass around causal processing. May or may not be intentional -- needs explicit design decision.

## Related Documentation

- `v2/docs/how_training_works.md` -- training loop mechanics
- `v2/program.md` -- protocol rules for mutable files (only train.py and policy.py)
