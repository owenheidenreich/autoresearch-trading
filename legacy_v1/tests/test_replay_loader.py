from __future__ import annotations

import os
import textwrap

import importlib


replay = importlib.import_module("training.replay")


def test_replay_loader_skips_top_level_side_effect_assignments(tmp_path, monkeypatch) -> None:
    train_py = tmp_path / "best_train.py"
    train_py.write_text(
        textwrap.dedent(
            """
            import os
            import torch
            import torch.nn as nn

            os.environ["REPLAY_SIDE_EFFECT"] = "set_by_exec"
            SAFE_CONST = 42
            FEATURE_GROUPS = {"g": (0, 1)}

            class TradingModel(nn.Module):
                def __init__(
                    self,
                    num_features=60,
                    lookback=120,
                    d_model=16,
                    n_heads=1,
                    n_layers=1,
                    ff_mult=2,
                    dropout=0.0,
                ):
                    super().__init__()
                    self.feature_gate = nn.Module()
                    self.feature_gate.gate_net = nn.Sequential(nn.Linear(num_features, 2))

                def forward(self, x):
                    b = x.shape[0]
                    return torch.zeros((b, 2)), torch.zeros((b, 6))
            """
        )
    )

    monkeypatch.delenv("REPLAY_SIDE_EFFECT", raising=False)
    model_cls = replay._load_model_class_from_train_py(str(train_py))

    assert model_cls is not None
    assert os.environ.get("REPLAY_SIDE_EFFECT") is None


def test_format_time_handles_iso_datetime_strings() -> None:
    # Explicit ET offset should preserve local wall clock.
    assert replay._format_time("2026-03-18T09:41:00-04:00") == "09:41"
    # UTC timestamp should convert to ET.
    assert replay._format_time("2026-03-18T14:41:00+00:00") == "10:41"
