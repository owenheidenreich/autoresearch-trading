from v5.ops.audit_causal_day_fit_features import feature_ledger
from v5.research.causal_day_tensorizer import CANDLE_FEATURES, LADDER_FEATURES


def test_every_fitted_feature_has_a_single_causal_availability_row() -> None:
    ledger = feature_ledger()
    assert len(ledger) == len(CANDLE_FEATURES) + len(LADDER_FEATURES) + 20
    assert not ledger.duplicated(["group", "feature"]).any()
    assert ledger["maximum_timestamp"].eq("t").all()
    assert not ledger["future_value_allowed"].any()
