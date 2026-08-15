from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from v4.model import protocol101_fresh_model_artifact as artifact


def _model() -> HistGradientBoostingRegressor:
    features = np.arange(80, dtype=float).reshape(40, 2)
    target = features[:, 0] * 0.25 - features[:, 1] * 0.1
    return HistGradientBoostingRegressor(
        max_iter=5,
        max_depth=2,
        random_state=42,
    ).fit(features, target)


def test_campaign_binding_changes_only_pickle_identity(tmp_path: Path) -> None:
    model = _model()
    probe = np.asarray([[4.0, 5.0], [20.0, 21.0]])
    expected = model.predict(probe)
    raw_hash = hashlib.sha256(
        pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()
    root = tmp_path / "fresh"
    path = root / "H0/policy0/seed42/fold1/model.pkl"

    binding = artifact.write_bound_model(
        path,
        model,
        allowed_root=root,
        campaign_namespace="fresh-campaign",
        binding_sources={"goal_sha256": "a" * 64},
    )

    restored = pickle.loads(path.read_bytes())
    assert np.array_equal(restored.predict(probe), expected)
    assert hashlib.sha256(path.read_bytes()).hexdigest() != raw_hash
    assert binding["prediction_semantics_changed"] is False
    assert binding["scientific_model_state_changed"] is False
    assert getattr(
        restored,
        artifact.MODEL_ARTIFACT_BINDING_ATTRIBUTE,
    ) == binding


def test_unit_identity_makes_equal_fits_distinct(tmp_path: Path) -> None:
    root = tmp_path / "fresh"
    paths = [
        root / "H0/policy0/seed42/fold1/model.pkl",
        root / "H0/policy0/seed42/fold2/model.pkl",
    ]
    for path in paths:
        artifact.write_bound_model(
            path,
            _model(),
            allowed_root=root,
            campaign_namespace="fresh-campaign",
            binding_sources={"goal_sha256": "a" * 64},
        )

    assert hashlib.sha256(paths[0].read_bytes()).hexdigest() != hashlib.sha256(
        paths[1].read_bytes()
    ).hexdigest()


def test_binding_rejects_path_outside_fitted_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside fitted root"):
        artifact.write_bound_model(
            tmp_path / "outside/model.pkl",
            _model(),
            allowed_root=tmp_path / "fresh",
            campaign_namespace="fresh-campaign",
            binding_sources={"goal_sha256": "a" * 64},
        )
