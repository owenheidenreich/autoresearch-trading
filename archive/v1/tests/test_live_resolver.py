from __future__ import annotations

from training.prepare import (
    ACTION_BUY_CALL_ATM,
    ACTION_BUY_CALL_OTM5,
    ACTION_BUY_CALL_OTM10,
    ACTION_BUY_PUT_ATM,
    ACTION_BUY_PUT_OTM5,
    ACTION_BUY_PUT_OTM10,
)
from training.live.resolver import SPXWContractResolver


def test_action_to_contract_mapping() -> None:
    resolver = SPXWContractResolver(ib=None, auto_qualify=False)
    contracts = resolver.resolve_all(5003.2)

    assert contracts[ACTION_BUY_CALL_ATM].strike == 5005
    assert contracts[ACTION_BUY_CALL_OTM5].strike == 5010
    assert contracts[ACTION_BUY_CALL_OTM10].strike == 5015
    assert contracts[ACTION_BUY_PUT_ATM].strike == 5005
    assert contracts[ACTION_BUY_PUT_OTM5].strike == 5000
    assert contracts[ACTION_BUY_PUT_OTM10].strike == 4995

    assert contracts[ACTION_BUY_CALL_ATM].right == "C"
    assert contracts[ACTION_BUY_PUT_ATM].right == "P"
    assert contracts[ACTION_BUY_CALL_ATM].tradingClass == "SPXW"

