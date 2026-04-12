from __future__ import annotations

import numpy as np

from v2.core.chain_data import QUALITY_VALID, build_contract_row
from v3.core.data import EpisodeMarket
from v3.core.env import ExactChainEnv, RewardConfig
from v3.core.execution import ExecutionConfig, range_to_squashed, risk_budget_to_qty, size_aware_slippage_frac
from v3.core.market_state import FIVE_MINUTE_FEATURE_NAMES, _compute_session_state
from v3.core.schema import AgentAction, SESSION_STATE_FEATURE_NAMES, TRADE_STATE_FEATURE_NAMES


def make_episode_market(n_bars: int = 12) -> EpisodeMarket:
    spot = np.linspace(100.0, 100.0 + 0.5 * (n_bars - 1), n_bars, dtype=np.float32)
    mids = np.vstack(
        [
            np.linspace(1.0, 2.5, n_bars, dtype=np.float32),
            np.linspace(1.0, 0.4, n_bars, dtype=np.float32),
            np.linspace(0.6, 1.6, n_bars, dtype=np.float32),
        ]
    )
    bid = np.maximum(0.01, mids - 0.05)
    ask = mids + 0.05
    contract_strike = np.array([100.0, 100.0, 105.0], dtype=np.float32)
    contract_right = np.array([0, 1, 0], dtype=np.int8)
    row_features = []
    row_contract_idx = []
    row_labels = []
    bar_ptrs = [0]
    for bar in range(n_bars):
        for idx in range(3):
            right = "P" if contract_right[idx] == 1 else "C"
            delta = -0.50 if right == "P" else 0.50 if idx == 0 else 0.30
            row_features.append(
                build_contract_row(
                    strike=float(contract_strike[idx]),
                    right=right,
                    mid=float(mids[idx, bar]),
                    spread_frac=float((ask[idx, bar] - bid[idx, bar]) / mids[idx, bar]),
                    volume=100,
                    transactions=25,
                    iv=0.20,
                    delta=delta,
                    gamma=0.01,
                    theta=-0.02,
                    spot=float(spot[bar]),
                    minutes_to_close=max(1, 390 - bar),
                    quality=QUALITY_VALID,
                    is_executable=True,
                )
            )
            row_contract_idx.append(idx)
            row_labels.append(0.0)
        bar_ptrs.append(len(row_features))
    sidecar = {
        "date": "20260101",
        "expiry": "20260101",
        "bar_timestamps": np.arange(n_bars, dtype=np.int64),
        "contract_strike": contract_strike,
        "contract_right": contract_right,
        "contract_mid": mids,
        "contract_bid": bid,
        "contract_ask": ask,
        "contract_quality": np.full_like(mids, QUALITY_VALID, dtype=np.int8),
        "row_features": np.asarray(row_features, dtype=np.float32),
        "row_labels": np.asarray(row_labels, dtype=np.float32),
        "row_contract_idx": np.asarray(row_contract_idx, dtype=np.int32),
        "bar_ptrs": np.asarray(bar_ptrs, dtype=np.int32),
    }

    context = np.zeros((n_bars, 47), dtype=np.float32)
    context[:, 1] = np.linspace(0.2, 0.8, n_bars)
    context[:, 6] = np.linspace(-0.1, 0.1, n_bars)
    context[:, 9] = np.linspace(0.3, 0.6, n_bars)
    context[:, 14] = 0.33
    context[:, 17] = np.linspace(0.1, 0.9, n_bars)
    context[:, 41] = np.linspace(0.4, 0.8, n_bars)

    n_buckets = int(np.ceil(n_bars / 5.0))
    context_5m = np.zeros((n_buckets, len(FIVE_MINUTE_FEATURE_NAMES)), dtype=np.float32)
    for bucket in range(n_buckets):
        context_5m[bucket, 0] = 0.1 * (bucket + 1)
        context_5m[bucket, 6] = 0.05 * bucket
        context_5m[bucket, 16] = 0.5
        context_5m[bucket, 17] = 0.6

    session_state = np.zeros((n_bars, len(SESSION_STATE_FEATURE_NAMES)), dtype=np.float32)
    session_state[:, 0] = np.arange(n_bars, dtype=np.float32) / 390.0
    session_state[:, 1] = np.maximum(0, n_bars - 1 - np.arange(n_bars, dtype=np.float32)) / 390.0
    session_state[:, 2] = np.linspace(0.0, 0.02, n_bars)
    session_state[:, 3] = np.linspace(-0.01, 0.01, n_bars)
    session_state[:, 10] = np.linspace(0.0, 0.5, n_bars)
    session_state[:, 15] = np.linspace(-0.2, 0.2, n_bars)

    return EpisodeMarket(
        date="2026-01-01",
        global_indices=np.arange(n_bars, dtype=np.int32),
        context_source=context,
        spot_prices=spot,
        bar_of_day=np.arange(n_bars, dtype=np.int32),
        sidecar=sidecar,
        max_contracts=3,
        context_5m_source=context_5m,
        session_state_source=session_state,
    )


def make_open_action(
    config: ExecutionConfig,
    contract_row: int = 0,
    risk: float = 0.03,
    *,
    exit_style: str = "STATIC",
    stop: float = 0.30,
    target: float = 0.50,
    time_stop: float = 1.0,
) -> AgentAction:
    return AgentAction(
        action_type="OPEN",
        contract_row=contract_row,
        exit_style=exit_style,
        risk_budget_frac=range_to_squashed(risk, 0.0, config.max_risk_budget_frac),
        stop_frac=range_to_squashed(stop, config.min_stop_frac, config.max_stop_frac),
        target_frac=range_to_squashed(target, config.min_target_frac, config.max_target_frac),
        time_stop_frac=range_to_squashed(time_stop, config.min_time_stop_frac, config.max_time_stop_frac),
        confidence=0.9,
    )


def test_context_windows_pad_and_align_causally() -> None:
    episode = make_episode_market(n_bars=10)
    ctx_1m_start = episode.context_1m_window(0)
    assert ctx_1m_start.shape == (90, 47)
    assert np.allclose(ctx_1m_start, 0.0)

    ctx_1m_mid = episode.context_1m_window(5)
    assert np.isclose(ctx_1m_mid[-1, 1], episode.context_source[4, 1])

    ctx_5m_pre_close = episode.context_5m_window(3)
    assert ctx_5m_pre_close.shape == (78, len(FIVE_MINUTE_FEATURE_NAMES))
    assert np.allclose(ctx_5m_pre_close, 0.0)

    ctx_5m_first_close = episode.context_5m_window(4)
    assert np.isclose(ctx_5m_first_close[-1, 0], episode.context_5m_source[0, 0])

    ctx_5m_second_close = episode.context_5m_window(9)
    assert np.isclose(ctx_5m_second_close[-2, 0], episode.context_5m_source[0, 0])
    assert np.isclose(ctx_5m_second_close[-1, 0], episode.context_5m_source[1, 0])


def test_session_state_partial_opening_range_and_volume_baseline_are_causal() -> None:
    n = 65
    day_open = np.linspace(100.0, 106.4, n)
    day_close = np.linspace(100.2, 106.6, n)
    day_high = np.maximum(day_open, day_close) + 0.3
    day_low = np.minimum(day_open, day_close) - 0.3
    day_spy_volume = np.full(n, 1000.0)
    option_volume = np.linspace(100.0, 300.0, n)
    baseline = np.full(n, 150.0).cumsum()

    session = _compute_session_state(
        day_open=day_open,
        day_high=day_high,
        day_low=day_low,
        day_close=day_close,
        day_spy_volume=day_spy_volume,
        option_volume=option_volume,
        baseline_cum_volume=baseline,
    )
    assert session.shape == (n, len(SESSION_STATE_FEATURE_NAMES))
    assert np.isclose(session[0, 10], 0.0)
    assert np.isclose(session[29, 10], 0.0)
    assert session[40, 10] >= 0.0
    assert session[10, 12] <= session[11, 12] + (1.0 / 390.0)
    assert np.isfinite(session[:, 15]).all()


def test_size_aware_slippage_is_monotonic() -> None:
    cfg = ExecutionConfig()
    s1 = size_aware_slippage_frac(mid=2.0, bid=1.95, ask=2.05, qty=1, config=cfg)
    s2 = size_aware_slippage_frac(mid=2.0, bid=1.95, ask=2.05, qty=2, config=cfg)
    s4 = size_aware_slippage_frac(mid=2.0, bid=1.95, ask=2.05, qty=4, config=cfg)
    assert s1 < s2 < s4


def test_risk_budget_to_qty_is_positive_and_affordable() -> None:
    cfg = ExecutionConfig()
    qty = risk_budget_to_qty(
        risk_budget_frac=0.02,
        equity=10_000.0,
        cash=10_000.0,
        entry_mid=2.0,
        bid=1.95,
        ask=2.05,
        stop_frac=0.30,
        config=cfg,
    )
    assert qty > 0
    assert isinstance(qty, int)


def test_invalid_contract_row_is_rejected() -> None:
    env = ExactChainEnv(make_episode_market())
    env.reset()
    _, _, _, info = env.step(make_open_action(ExecutionConfig(), contract_row=99))
    assert info["execution"].executed is False
    assert info["execution"].event == "reject"


def test_open_then_eod_flatten_creates_trade_record() -> None:
    env = ExactChainEnv(make_episode_market(), execution_config=ExecutionConfig(), reward_config=RewardConfig())
    env.reset()
    env.step(
        make_open_action(
            ExecutionConfig(),
            contract_row=0,
            risk=0.02,
            exit_style="MODEL_EXIT",
            stop=0.95,
            target=3.0,
        )
    )
    done = False
    while not done:
        _, _, done, info = env.step(AgentAction(action_type="HOLD", manage_mode="HOLD", confidence=0.5))
    assert info["summary"]["total_trades"] == 1
    assert env.trade_records[0].exit_reason in {"EOD", "TIME_STOP"}


def test_scale_out_reduces_qty_on_same_contract() -> None:
    cfg = ExecutionConfig()
    env = ExactChainEnv(make_episode_market(), execution_config=cfg, reward_config=RewardConfig())
    env.reset()
    env.step(make_open_action(cfg, contract_row=0, risk=0.05))
    starting_qty = env.position.qty
    adjust = AgentAction(
        action_type="ADJUST",
        manage_mode="ADJUST",
        updated_stop_frac=range_to_squashed(0.25, cfg.min_stop_frac, cfg.max_stop_frac),
        updated_target_frac=range_to_squashed(0.60, cfg.min_target_frac, cfg.max_target_frac),
        updated_time_stop_frac=range_to_squashed(0.50, cfg.min_time_stop_frac, cfg.max_time_stop_frac),
        size_delta_frac=-0.5,
        confidence=0.8,
    )
    env.step(adjust)
    assert env.position.qty < starting_qty
    assert env.position.contract_index == 0


def test_trade_memory_tracks_long_hold_mfe_and_entry_anchors() -> None:
    cfg = ExecutionConfig()
    env = ExactChainEnv(make_episode_market(n_bars=120), execution_config=cfg, reward_config=RewardConfig())
    obs = env.reset()
    obs, _, _, _ = env.step(
        make_open_action(
            cfg,
            contract_row=0,
            risk=0.03,
            exit_style="MODEL_EXIT",
            stop=0.95,
            target=3.0,
        )
    )
    initial_entry_vwap = env.position.entry_vwap_dist
    for _ in range(95):
        obs, _, done, _ = env.step(AgentAction(action_type="HOLD", manage_mode="HOLD", confidence=0.5))
        assert done is False
    trade = {name: float(obs.trade_state[i]) for i, name in enumerate(TRADE_STATE_FEATURE_NAMES)}
    assert trade["in_position"] == 1.0
    assert trade["bars_held_norm"] > (90.0 / 390.0)
    assert trade["mfe_frac_equity"] >= 0.0
    assert trade["peak_unrealized_frac_equity"] >= 0.0
    assert env.position.entry_vwap_dist == initial_entry_vwap
