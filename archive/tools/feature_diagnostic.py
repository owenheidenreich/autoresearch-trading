#!/usr/bin/env python3
"""Feature diagnostic: variance, correlation, and permutation importance.

Usage:
  python3 tools/feature_diagnostic.py                  # variance + correlation
  python3 tools/feature_diagnostic.py --permutation    # + permutation importance (needs model)
"""
import os, sys, argparse
import numpy as np
import torch

# Allow imports from training/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-trading")
DATA_PT = os.path.join(CACHE_DIR, "features", "data.pt")





def run_variance_analysis(features, valid, names):
    """Analyze per-feature variance, zero rates, and NaN rates."""
    N, F = features.shape
    valid_feat = features[valid]
    n_valid = valid_feat.shape[0]

    print(f"\n{'='*80}")
    print(f"  FEATURE VARIANCE ANALYSIS ({n_valid:,} valid bars)")
    print(f"{'='*80}\n")
    print(f"{'Feature':<25} {'Variance':>10} {'Std':>10} {'Zero%':>8} {'NaN%':>8} {'Clip%':>8} {'Flag'}")
    print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    results = []
    for i in range(F):
        col = valid_feat[:, i]
        nan_rate = np.isnan(col).mean()
        finite = col[np.isfinite(col)]
        if len(finite) == 0:
            results.append((names[i], 0.0, 0.0, 0.0, nan_rate, 0.0, 'ALL_NAN'))
            continue

        var = np.var(finite)
        std = np.std(finite)
        zero_rate = (finite == 0.0).mean()
        # Check clipping at boundaries
        clip_rate = ((np.abs(finite) > 4.9) | (finite == finite.max()) & (np.sum(finite == finite.max()) > n_valid * 0.1)).mean()

        flags = []
        if var < 0.001:
            flags.append('LOW_VAR')
        if zero_rate > 0.8:
            flags.append('MOSTLY_ZERO')
        if nan_rate > 0.5:
            flags.append('HIGH_NAN')

        flag_str = ','.join(flags) if flags else ''
        results.append((names[i], var, std, zero_rate, nan_rate, clip_rate, flag_str))
        print(f"{names[i]:<25} {var:>10.6f} {std:>10.6f} {zero_rate:>7.1%} {nan_rate:>7.1%} {clip_rate:>7.1%} {flag_str}")

    # Summary
    flagged = [r for r in results if r[6]]
    print(f"\n--- Flagged features: {len(flagged)} ---")
    for r in flagged:
        print(f"  {r[0]}: {r[6]}")

    return results


def run_correlation_analysis(features, valid, names, threshold=0.85):
    """Find highly correlated feature pairs."""
    valid_feat = features[valid]
    # Replace NaN with 0 for correlation
    clean = np.nan_to_num(valid_feat, nan=0.0)

    print(f"\n{'='*80}")
    print(f"  FEATURE CORRELATION ANALYSIS (|corr| > {threshold})")
    print(f"{'='*80}\n")

    F = clean.shape[1]
    corr = np.corrcoef(clean.T)

    pairs = []
    for i in range(F):
        for j in range(i + 1, F):
            if abs(corr[i, j]) > threshold:
                pairs.append((names[i], names[j], corr[i, j]))

    pairs.sort(key=lambda x: -abs(x[2]))

    if not pairs:
        print(f"  No feature pairs with |corr| > {threshold}")
    else:
        print(f"{'Feature A':<25} {'Feature B':<25} {'Correlation':>12}")
        print(f"{'-'*25} {'-'*25} {'-'*12}")
        for a, b, c in pairs:
            print(f"{a:<25} {b:<25} {c:>12.4f}")

    # Also show top correlations at lower threshold for reference
    print(f"\n--- All pairs with |corr| > 0.70 ---")
    lower_pairs = []
    for i in range(F):
        for j in range(i + 1, F):
            if abs(corr[i, j]) > 0.70:
                lower_pairs.append((names[i], names[j], corr[i, j]))
    lower_pairs.sort(key=lambda x: -abs(x[2]))
    for a, b, c in lower_pairs:
        marker = ' *** REDUNDANT' if abs(c) > threshold else ''
        print(f"  {a:<25} {b:<25} {c:>8.4f}{marker}")

    return pairs, corr


def run_permutation_importance(model_path, data, names, n_shuffles=3):
    """Measure per-feature importance by shuffling and re-evaluating."""
    from prepare import evaluate_trades

    device = torch.device('cpu')

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})

    # Try loading model class from best_train.py
    train_py = os.path.join(os.path.dirname(model_path), 'best_train.py')
    if not os.path.exists(train_py):
        train_py = os.path.join(os.path.dirname(model_path), 'train.py')

    from replay import _load_model_class_from_train_py
    ModelClass = _load_model_class_from_train_py(train_py, config)
    model = ModelClass()
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    lookback = config.get('lookback', 120)

    # Score config
    score_config = {
        'profit_factor_weight': 2.0,
        'win_rate_weight': 0.0,
        'trade_sharpe_weight': 1.0,
        'freq_mult_weight': 1.0,
        'drawdown_penalty': 0.5,
        'hold_bonus': 0.0,
        'freq_center': 2.5,
        'freq_width': 2.5,
        'consec_loss_threshold': 3,
        'short_hold_threshold': 0.30,
        'stop_rate_threshold': 0.30,
        'ruin_penalty': 1.0,
        'ruin_threshold': 0.25,
        'risk_fraction_penalty': 0.5,
    }

    print(f"\n{'='*80}")
    print(f"  PERMUTATION IMPORTANCE ({n_shuffles} shuffles per feature)")
    print(f"{'='*80}\n")

    # Baseline evaluation
    print("Computing baseline score...")
    baseline = evaluate_trades(model, data, lookback, device, score_config=score_config)
    base_score = baseline.get('score', 0.0)
    base_pf = baseline.get('profit_factor', 0.0)
    base_trades = baseline.get('total_trades', 0)
    print(f"  Baseline: score={base_score:.2f}, PF={base_pf:.2f}, trades={base_trades}")

    F = data['features'].shape[1]
    importances = []

    for i in range(F):
        scores = []
        pfs = []
        for trial in range(n_shuffles):
            # Shuffle feature i across all bars
            data_copy = dict(data)
            feat_copy = data['features'].clone()
            perm = torch.randperm(feat_copy.shape[0])
            feat_copy[:, i] = feat_copy[perm, i]
            data_copy['features'] = feat_copy

            result = evaluate_trades(model, data_copy, lookback, device, score_config=score_config)
            scores.append(result.get('score', 0.0))
            pfs.append(result.get('profit_factor', 0.0))

        avg_score = np.mean(scores)
        avg_pf = np.mean(pfs)
        score_drop = base_score - avg_score
        pf_drop = base_pf - avg_pf
        importances.append((names[i], score_drop, pf_drop, avg_score, avg_pf))

    # Sort by score drop (most important first)
    importances.sort(key=lambda x: -x[1])

    print(f"\n{'Feature':<25} {'Score Drop':>12} {'PF Drop':>10} {'Shuffled Score':>15} {'Shuffled PF':>12}")
    print(f"{'-'*25} {'-'*12} {'-'*10} {'-'*15} {'-'*12}")
    for name, sd, pd, ss, sp in importances:
        marker = ''
        if sd < -0.5:
            marker = ' << NOISE (helps when removed)'
        elif abs(sd) < 0.3:
            marker = ' ~ no impact'
        elif sd > 2.0:
            marker = ' ** IMPORTANT'
        print(f"{name:<25} {sd:>12.2f} {pd:>10.3f} {ss:>15.2f} {sp:>12.3f}{marker}")

    return importances


def main():
    parser = argparse.ArgumentParser(description='Feature diagnostic')
    parser.add_argument('--permutation', action='store_true', help='Run permutation importance (needs model)')
    parser.add_argument('--model', default=None, help='Model path for permutation test')
    parser.add_argument('--shuffles', type=int, default=3, help='Number of shuffles per feature')
    args = parser.parse_args()

    print("Loading data.pt...")
    data = torch.load(DATA_PT, weights_only=False)
    features = data['features'].numpy()
    valid = data['valid_mask'].numpy().astype(bool)
    N, F = features.shape

    # Use stored names if available, else import from prepare
    if 'feature_names' in data:
        names = list(data['feature_names'])
    else:
        try:
            from prepare import FEATURE_NAMES
            names = FEATURE_NAMES if len(FEATURE_NAMES) == F else [f'feat_{i}' for i in range(F)]
        except ImportError:
            names = [f'feat_{i}' for i in range(F)]

    print(f"Data: {N:,} bars, {F} features, {valid.sum():,} valid")

    # 1. Variance analysis
    var_results = run_variance_analysis(features, valid, names)

    # 2. Correlation analysis
    corr_pairs, corr_matrix = run_correlation_analysis(features, valid, names)

    # 3. Permutation importance (optional)
    if args.permutation:
        model_path = args.model
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'training', 'best_model.pt')
        if not os.path.exists(model_path):
            print(f"\nERROR: Model not found at {model_path}")
            return
        importances = run_permutation_importance(model_path, data, names, args.shuffles)

    print(f"\n{'='*80}")
    print(f"  DONE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
