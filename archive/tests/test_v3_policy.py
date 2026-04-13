from __future__ import annotations

import tempfile

import torch

from tests.test_v3_env import make_episode_market
from v3.core.artifact import load_checkpoint, save_checkpoint
from v3.core.env import ExactChainEnv
from v3.core.schema import MODEL_SCHEMA_VERSION
from v3.model import RLPolicy, observation_to_tensors
from v3.replay import load_policy_from_checkpoint, run_policy_episode


def test_policy_act_and_evaluate_are_finite() -> None:
    torch.manual_seed(0)
    env = ExactChainEnv(make_episode_market())
    obs = env.reset()
    policy = RLPolicy()
    obs_t = observation_to_tensors(obs, torch.device("cpu"))
    action, log_prob, value, aux = policy.act(obs_t, deterministic=False)
    new_log_prob, entropy, new_value = policy.evaluate_actions(obs_t, action)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(value).all()
    assert torch.isfinite(new_log_prob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(new_value).all()
    env_action = policy.to_agent_action(obs, action, aux)
    assert env_action.action_type in {"NOOP", "OPEN", "HOLD", "CLOSE", "ADJUST"}


def test_deterministic_replay_is_repeatable() -> None:
    torch.manual_seed(0)
    policy = RLPolicy()
    env1 = ExactChainEnv(make_episode_market())
    env2 = ExactChainEnv(make_episode_market())
    summary1, trades1, traces1, _, _ = run_policy_episode(policy, env1, device=torch.device("cpu"), deterministic=True)
    summary2, trades2, traces2, _, _ = run_policy_episode(policy, env2, device=torch.device("cpu"), deterministic=True)
    assert summary1.ending_equity == summary2.ending_equity
    assert summary1.total_reward == summary2.total_reward
    assert len(trades1) == len(trades2)
    assert len(traces1) == len(traces2)


def test_checkpoint_reload_replays_identically() -> None:
    torch.manual_seed(0)
    policy = RLPolicy()
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        save_checkpoint(
            tmp.name,
            {
                "model_schema_version": MODEL_SCHEMA_VERSION,
                "model_state_dict": {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()},
                "model_config": {},
                "execution_config": {},
                "reward_config": {},
                "eval_config": {},
            },
        )
        reloaded, checkpoint = load_policy_from_checkpoint(tmp.name, device="cpu")
        raw = load_checkpoint(tmp.name, device="cpu")
        assert checkpoint["model_schema_version"] == MODEL_SCHEMA_VERSION
        assert checkpoint["model_config"] == raw["model_config"]
        env1 = ExactChainEnv(make_episode_market())
        env2 = ExactChainEnv(make_episode_market())
        summary1, _, _, _, _ = run_policy_episode(policy, env1, device=torch.device("cpu"), deterministic=True)
        summary2, _, _, _, _ = run_policy_episode(reloaded, env2, device=torch.device("cpu"), deterministic=True)
        assert summary1.ending_equity == summary2.ending_equity
