# Engineering Handoff: AI Paper Concepts Integration into ART²

**Date:** 2026-03-26
**Author:** Claude Opus 4.6 (research & analysis)
**Status:** Ready for implementation

---

## 1. Background

Three recent AI research papers were analyzed for concepts transferable to the ART² automated trading pipeline:

| Paper | Authors | Core Contribution |
|-------|---------|-------------------|
| **LLM in a Flash** | Apple (Alizadeh et al.) | Streaming model weights from flash storage using activation sparsity, sliding windows, and predictive loading |
| **QJL: 1-Bit Quantized JL Transform** | Zandieh et al. | Zero-overhead KV cache quantization via Johnson-Lindenstrauss random projection + sign-bit quantization |
| **TurboQuant** | Google Research (Zandieh et al., ICLR 2026) | Near-optimal vector quantization via random rotation + scalar quantizers, with two-stage residual correction |

**Key constraint:** ART² uses a tiny 127K-parameter transformer (d_model=64, 3 layers, 4 heads). The literal techniques (flash weight streaming, KV cache compression for billion-param LLMs) do not apply at this scale. The value lies in **conceptual transfers** — applying the underlying principles to ART²'s actual bottlenecks.

---

## 2. The Seven Proposals

### P1: Sliding Window Evaluation Caching

**Problem:** `evaluate_trades()` in `training/train.py` is the pipeline's #1 bottleneck at 30-60 seconds per run. It processes ~130K validation bars sequentially, one bar at a time, running a full model forward pass for each bar. Adjacent bars share 119 of 120 input positions, yet this overlap is never exploited.

**Paper concept:** LLM in a Flash's sliding window technique — cache recently-used neuron data, only load/compute the incremental difference for new tokens.

**What will be done:** Restructure `evaluate_trades()` to batch-infer bars during flat position states (not holding a trade). Between trades, the position state is constant (all zeros for in_trade, bars_held, etc.), so all flat-state bars are independent and can be inferred in a single batched forward pass. Sequential per-bar inference is only needed during active positions where position state changes every bar.

**Pseudocode:**
```python
def evaluate_trades_optimized(model, features, ...):
    # Phase 1: Batch-compute ALL direction logits (already done in current code)
    all_dir_logits = batch_inference(model, features)  # existing

    # Phase 2: Identify flat-state segments
    bars = list(range(num_val_bars))
    flat_segments = []    # [(start, end), ...] where position is flat
    active_segments = []  # [(start, end), ...] where position is active

    # Phase 2a: Batch-infer gate logits for ALL bars with flat state
    flat_bar_indices = concatenate(flat_segments)
    flat_windows = features[flat_bar_indices]  # (N_flat, 120, 37)
    flat_position_state = torch.zeros(len(flat_bar_indices), 7)  # all zeros
    flat_gate_logits = model.batch_forward(flat_windows, flat_position_state)

    # Phase 2b: Sequential inference ONLY for active-position bars
    for segment in active_segments:
        for bar in range(segment.start, segment.end):
            position_state = update_position_state(...)  # changes every bar
            gate_logits = model.forward(features[bar], position_state)
            # ... execute trade logic ...

    # ~75% of bars are flat -> batched (fast)
    # ~25% of bars are active -> sequential (unavoidable)
```

**Expected outcome:** Evaluation time drops from 30-60s to 10-20s. This reclaims 10-40 seconds per experiment, translating to 1-2 additional training steps within the 300-second budget. Model quality improves indirectly through more training iterations.

**Warm-start safe:** Yes. No model architecture changes.
**Complexity:** Medium-High. The evaluate_trades function is 600+ lines with tightly coupled state tracking. The flat/active segmentation must be done in two passes (first pass to identify trade boundaries, second pass for batched inference).

---

### P2: Asymmetric Precision Training

**Problem:** The entire training forward pass runs under `torch.amp.autocast('cuda', dtype=torch.bfloat16)`. The gate head outputs only 2 logits (TRADE vs NO_TRADE). Near the decision boundary (logit ~ 0), bf16 has a precision of ~0.0078, meaning values like 0.003 and -0.003 (which should produce different trade decisions) can round to the same number. With only 30-40 training steps per experiment, each gradient must be maximally informative — noisy gate gradients waste precious steps.

**Paper concept:** QJL's asymmetric estimator — quantize one side (keys) to 1 bit, keep the other (queries) at full precision. The asymmetry provides an unbiased estimator. Applied here: keep the transformer backbone in bf16 (fast bulk computation), but compute the loss in fp32 (accurate gradients where decisions are made).

**What will be done:** Add `.float()` casts on the head outputs before they enter the loss function. The transformer backbone (the expensive part) stays in bf16. Only the small loss computation moves to fp32.

**Pseudocode:**
```python
def sniper_loss(gate_logits, dir_logits, value_pred, risk_output, targets, ...):
    # NEW: Cast head outputs to float32 for precise loss computation
    gate_logits = gate_logits.float()    # (batch, 2) -- tiny tensor
    dir_logits = dir_logits.float()      # (batch, 6) -- tiny tensor
    value_pred = value_pred.float()      # (batch, 1) -- tiny tensor
    risk_output = risk_output.float()    # (batch, 3) -- tiny tensor

    # Also cast PnL target arrays to float32
    best_stopped_pnl = best_stopped_pnl.float()
    all_pnl = all_pnl.float()

    # ... rest of loss computation now runs in fp32 ...
    # Gate cross-entropy, direction KL-div, PnL alignment, etc.
    # Gradients flow back through the cast, so backbone still gets bf16 compute
```

**Expected outcome:** Sharper gate probability calibration. The trade/no-trade decision boundary becomes more precise, potentially improving win rate by 0.5-2%. The cost is essentially zero — the cast tensors are tiny (batch x 2-6 elements).

**Warm-start safe:** Yes. No architecture or data changes. Pure optimizer-level improvement.
**Complexity:** Low. 5-8 lines of code.

---

### P3: "Trust the Hardware" — Remove Software Overhead

**Problem:** Several lines in `training/train.py` add software overhead that duplicates or fights against what PyTorch/CUDA already handles optimally:

1. `torch.cuda.synchronize()` is called at the start AND end of every training step. This forces the CPU to wait for the GPU to finish all work before proceeding, stalling the async compute pipeline. On H100 with bf16, a training step is fast enough that these syncs represent measurable overhead.
2. `torch.nan_to_num(data['features'], nan=0.0)` at data load time materializes a full copy of the entire features tensor (~59MB) even when there are no NaNs.
3. `gc.freeze(); gc.disable()` after step 0 may interfere with PyTorch's tensor reference counting on GPU.

**Paper concept:** LLM in a Flash's most counterintuitive finding — deleting a carefully engineered 9.8 GB Metal LRU cache and letting macOS handle caching natively made everything 38% faster. The custom cache forced the hardware memory compressor to work overtime. Lesson: trust the hardware, get the software out of the way.

**What will be done:** Remove the two per-step `cuda.synchronize()` calls from the training loop (keep only the synchronization around evaluation timing where wall-clock accuracy matters). Move NaN handling to per-batch in the dataloader or eliminate it if data is clean. Profile the gc.freeze impact.

**Pseudocode:**
```python
# BEFORE (current code):
def train_step(model, batch, optimizer, scaler):
    torch.cuda.synchronize()  # REMOVE THIS — stalls pipeline
    optimizer.zero_grad()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        loss = compute_loss(model, batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()  # REMOVE THIS — stalls pipeline

# AFTER:
def train_step(model, batch, optimizer, scaler):
    optimizer.zero_grad()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        loss = compute_loss(model, batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    # No sync — let CUDA pipeline run asynchronously

# For NaN handling:
# BEFORE: data['features'] = torch.nan_to_num(data['features'], nan=0.0)  # copies 59MB
# AFTER:  handle per-batch in dataloader, or assert no NaNs at data prep time
```

**Expected outcome:** 0.5-2ms saved per training step. Over 3000+ steps in the inner loop, this accumulates to 1.5-6 seconds, potentially enabling 1-2 extra training steps within the 300-second budget.

**Warm-start safe:** Yes.
**Complexity:** Low. Remove 2-3 lines, move 1 operation.

---

### P4: Data.pt Float16 Quantization

**Problem:** `data.pt` stores all tensors as float32 (~225MB). Features are z-scored (values in [-3, 3]), PnL arrays have limited dynamic range ([-1, 5]). Float32 provides 7 decimal digits of precision for values that only need 3-4 digits. This wastes disk space, slows I/O, and uses unnecessary GPU memory during upload.

**Paper concept:** TurboQuant's provably near-optimal quantization — compress vectors to lower bit-width while maintaining bounded distortion. For our data ranges, float16 provides sufficient precision with zero measurable loss. The two-stage concept applies: store the bulk (features, PnL) at reduced precision, keep critical metadata (masks, dates, indices) at full precision.

**What will be done:** In `prepare.py`'s `_build_dataset()`, store feature and PnL tensors as float16 in data.pt. In the data loading path, cast back to float32 for GPU computation (model still trains at full precision).

**Pseudocode:**
```python
# In prepare.py, when saving data.pt:
def _build_dataset():
    # ... compute features, PnL arrays ...

    # Two-stage: reduced precision for bulk data, full precision for metadata
    data = {
        # FLOAT16 — values with bounded range, safe for half precision
        'features': features.to(torch.float16),          # (N, 37), was float32
        'call_pnl': call_pnl.to(torch.float16),          # (N,)
        'put_pnl': put_pnl.to(torch.float16),            # (N,)
        'call_stopped_pnl': stopped.to(torch.float16),   # (N,)
        'exit_call_label': exit_labels.to(torch.float16), # (N,)
        # ... all PnL and label arrays ...

        # FULL PRECISION — masks, indices, critical metadata
        'valid_mask': valid_mask,                          # bool, keep as-is
        'actionable_mask': actionable_mask,                # bool
        'dates': dates,                                    # string/int
    }
    torch.save(data, 'data.pt')

# In train.py, when loading:
def load_data():
    data = torch.load('data.pt')
    # Cast back to float32 for training — model never sees float16
    features = data['features'].float()  # float16 -> float32
    call_pnl = data['call_pnl'].float()
    # ... etc ...
```

**Expected outcome:** data.pt shrinks from ~225MB to ~120MB (~47% reduction). Faster disk I/O when staging to Akash GPU. Faster torch.load(). Lower peak host memory. Model trains on identical float32 values (the float16 roundtrip introduces at most 0.1% error on z-scored features, well within noise).

**Warm-start safe:** Yes. Model sees float32 tensors at training time — architecture unchanged.
**Complexity:** Low. Change dtype in ~20 torch.tensor() calls, add .float() in load path.

---

### P5: JL Random Projection Feature Compression

**Problem:** The model's input projection is `nn.Linear(37, 64)` — 2,368 trainable parameters. With only 30-40 training steps per experiment, the input projection may be under-trained. Additionally, many of the 37 features are correlated (e.g., ret_6/ret_12, volume_ratio/volume_zscore, realized_vol/bar_range), meaning the input space has redundant dimensions.

**Paper concept:** QJL's Johnson-Lindenstrauss transform — a random Gaussian projection preserves pairwise distances within (1 +/- epsilon) bounds. TurboQuant's two-stage approach — apply the main compression (JL) to the bulk of features, then preserve critical features at full precision as the "residual" channel.

**What will be done:** Apply a fixed (non-trainable) random Gaussian projection matrix to compress the ~27 correlated continuous features into ~12-16 dimensions. Keep ~10 critical features (time_sin, time_cos, minutes_to_close, vix_regime, atm_iv, atm_gamma, bollinger_position, rsi_14, ib_break, theta_pressure) at full precision. Final input dimension: ~22-26 features (down from 37).

**Pseudocode:**
```python
# In prepare.py, after computing 37 features:
import numpy as np

# Fixed random seed for reproducibility (NOT learned)
rng = np.random.RandomState(42)

# Split features into critical (kept) and compressible (projected)
CRITICAL_INDICES = [29, 30, 31, 32, 33, ...]  # time, VIX, IV, gamma
COMPRESS_INDICES = [0, 1, 2, 3, ...]           # returns, volume, volatility, etc.

# JL projection matrix: (27 input dims) -> (14 output dims)
# Gaussian entries, scaled by 1/sqrt(output_dim) per JL lemma
d_in = len(COMPRESS_INDICES)   # ~27
d_out = 14                      # compressed dim
JL_MATRIX = rng.randn(d_in, d_out) / np.sqrt(d_out)  # (27, 14)

def compress_features(features_37):
    """Apply two-stage feature compression.
    Stage 1: Keep critical features at full precision (the "exact" channel)
    Stage 2: JL-project correlated features (the "compressed" channel)
    """
    critical = features_37[:, CRITICAL_INDICES]        # (N, 10)
    compressible = features_37[:, COMPRESS_INDICES]     # (N, 27)
    compressed = compressible @ JL_MATRIX               # (N, 14)
    return np.concatenate([critical, compressed], axis=1)  # (N, 24)

# Save in data.pt:
data['features'] = torch.tensor(compress_features(raw_features))  # (N, 24)
data['jl_matrix'] = torch.tensor(JL_MATRIX)  # for reproducibility

# In train.py:
NUM_FEATURES = 24  # was 37
# Input projection: nn.Linear(24, 64) — 1,536 params (was 2,368)
```

**Expected outcome:** Faster convergence — fewer input parameters means less to learn in 30-40 steps. The JL projection acts as implicit regularization by mixing correlated features into orthogonal dimensions (decorrelation). data.pt features tensor shrinks by ~35%. The two-stage approach ensures time-of-day and regime features (which the model relies on heavily for entry timing) are not distorted.

**Risk:** If the 27 compressed features contain non-redundant information that the JL projection dilutes, model quality could decrease. The JL lemma guarantees distance preservation but not signal preservation for supervised learning.

**Warm-start safe:** NO — FRESH START required. Feature count changes (37 -> 24), so `input_proj` weights have incompatible shape.
**Complexity:** Medium. JL projection is ~10 lines of numpy. Main work is updating FEATURE_GROUPS, NUM_FEATURES, and any feature-index references throughout train.py.

---

### P6: Predictive Day-Sequential Prefetching

**Problem:** The day-sequential dataloader (used for 92% of training steps) randomly samples days, then iterates bar-by-bar within each day. Each bar accesses a 120-bar window from the full features tensor on GPU (~400K bars). These are scattered random reads across GPU memory — poor cache locality.

**Paper concept:** LLM in a Flash's predictive loading — anticipate which data will be needed and pre-load it into a contiguous buffer before computation begins.

**What will be done:** When the day-sequential loader samples its batch of days, pre-slice the features for those days (plus lookback) into a contiguous GPU buffer. Index into this compact buffer instead of the full 400K-bar tensor.

**Pseudocode:**
```python
def make_day_sequential_loader(features, day_boundaries, lookback=120):
    for epoch in itertools.count():
        # Sample random days for this epoch
        sampled_days = random.sample(day_boundaries, k=batch_size)

        # PREDICTIVE PREFETCH: pre-slice features into contiguous buffer
        day_buffers = []
        for day_start, day_end in sampled_days:
            # Include lookback window before day start
            slice_start = max(0, day_start - lookback)
            slice_end = day_end
            day_buffers.append(features[slice_start:slice_end].clone())
            # .clone() forces contiguous memory layout

        # Now iterate bars within each day using the compact buffer
        for bar_idx in range(max_bars_per_day):
            for day_idx, (day_start, day_end) in enumerate(sampled_days):
                local_bar = bar_idx
                # Index into compact buffer (sequential reads)
                window = day_buffers[day_idx][local_bar:local_bar + lookback]
                yield window  # contiguous memory, good cache behavior
```

**Expected outcome:** ~5-10% faster per training step due to improved GPU memory access patterns. Over 30-40 steps, this could gain 1-2 extra steps.

**Warm-start safe:** Yes.
**Complexity:** Low-Medium.

---

### P7: Sparse FFN Regularization via TopK Activation

**Problem:** The model uses GELU activation in its FFN layers (d_model=64 -> 192 -> 64). GELU produces near-zero but non-zero outputs for negative inputs, meaning all 192 FFN neurons contribute to every prediction. This gives the model maximum capacity but also maximum overfitting potential — a known issue where train PF and val PF diverge.

**Paper concept:** LLM in a Flash demonstrated that FFN layers exhibit 90-97% activation sparsity with ReLU. Only a small fraction of neurons carry meaningful signal for any given input. Applied to ART²: forcing sparsity acts as structured regularization.

**What will be done:** Replace GELU with ReLU in the transformer FFN layers. Optionally add a TopK mask that zeros out all but the top K=96 (of 192) activations. This forces the model to make robust decisions using only half its FFN capacity, reducing overfitting.

**Pseudocode:**
```python
# In TradingModel.__init__:
# BEFORE:
encoder_layer = nn.TransformerEncoderLayer(
    d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FF_DIM,
    dropout=DROPOUT, activation='gelu', batch_first=True, norm_first=True
)

# AFTER (Option A — simple ReLU swap):
encoder_layer = nn.TransformerEncoderLayer(
    d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FF_DIM,
    dropout=DROPOUT, activation='relu', batch_first=True, norm_first=True
)

# AFTER (Option B — TopK sparse activation, custom FFN):
class SparseFFN(nn.Module):
    def __init__(self, d_model, d_ff, k, dropout=0.3):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)      # 64 -> 192
        self.down = nn.Linear(d_ff, d_model)      # 192 -> 64
        self.dropout = nn.Dropout(dropout)
        self.k = k  # top-K active neurons (e.g., 96)

    def forward(self, x):
        h = F.relu(self.up(x))                    # (batch, seq, 192)
        # TopK mask: keep only the K largest activations
        topk_vals, topk_idx = h.topk(self.k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, topk_idx, 1.0)
        h = h * mask                               # zero out bottom 50%
        return self.dropout(self.down(h))

# Usage: Replace the FFN in each TransformerEncoderLayer
# (requires subclassing or monkey-patching the layer)
```

**Expected outcome:** Uncertain — this is an experimental hypothesis. If the model is overfitting (train PF >> val PF), TopK sparsity should narrow the gap. If the model is underfitting, this will hurt. The ReLU-only swap (Option A) is lower risk; TopK (Option B) is more aggressive.

**Warm-start safe:** NO — FRESH START required. Activation function change means weight distributions are incompatible.
**Complexity:** Low for Option A (one string change), Medium for Option B (custom FFN module).

---

## 3. Implementation Order

```
Phase A (Safe, immediate):  P2 + P3  ->  single experiment, warm-start
Phase B (High impact):      P1       ->  eval speedup, warm-start
Phase C (Data pipeline):    P4       ->  data.pt compression, warm-start
Phase D (Architecture):     P5       ->  JL features, FRESH START
Phase E (Experimental):     P6 + P7  ->  if earlier phases show promise
```

**Rationale:** Phases A-C are all warm-start safe and low-to-medium complexity. They can be validated immediately without risking the current best model. Phase D requires a fresh start (expensive) and should only be attempted after the safe optimizations are in place. Phase E is experimental and depends on earlier results.

---

## 4. Expected End State

After full implementation:

| Metric | Current | After Phase A-C | After Phase D-E |
|--------|---------|-----------------|-----------------|
| Eval time | 30-60s | 10-20s | 10-20s |
| Training steps/experiment | 30-40 | 33-44 | 35-46 |
| data.pt size | ~225MB | ~120MB | ~100MB |
| Input params | 2,368 | 2,368 | ~1,536 |
| Gate precision | bf16 (0.0078) | fp32 (1e-7) | fp32 (1e-7) |
| Train/val PF gap | Varies | Slightly tighter | Potentially much tighter |

**Key success metric:** Score improvement (profit_factor - drawdown + reward_ratio) as measured by the existing inner_loop experiment framework. Each proposal is testable via standard KEEP/REVERT experiments — no new evaluation infrastructure needed.

---

## 5. Files Modified

| File | Proposals | Type of Change |
|------|-----------|----------------|
| `training/train.py` | P1, P2, P3, P6, P7 | Loss precision, sync removal, eval batching, FFN activation |
| `training/prepare.py` | P4, P5 | Data quantization, JL feature projection |
| `training/program.md` | P5, P7 | Update agent contract if architecture changes |
| `training/lab_notebook.md` | All | Document experiment results |

---

## 6. Risks and Mitigations

| Risk | Proposal | Mitigation |
|------|----------|------------|
| Eval batching produces different trade results | P1 | Diff trade logs against sequential baseline before accepting |
| fp32 loss changes convergence dynamics | P2 | Compare score distributions across 5+ experiments |
| Removing cuda.synchronize masks timing bugs | P3 | Keep sync around evaluation only; profile step times |
| float16 roundtrip introduces feature drift | P4 | Assert max absolute error < 0.01 per feature |
| JL projection dilutes critical signal | P5 | Two-stage approach preserves critical features; A/B test |
| TopK sparsity hurts small model capacity | P7 | Start with K=128 (conservative), tune down |
