"""IBKR order placement and management.

Receives a TradeIntent, resolves to an IBKR contract, places bracket
orders (entry + stop + take-profit), and manages the order lifecycle.
See docs/v2/execution.md for the full state machine.

Execution is the ONLY module that talks to the IBKR order API.

v1 origin: training/live/execution.py (OCOExecutionEngine, ExecutionState) +
training/live/resolver.py (SPXWContractResolver)
"""
# TODO: ExecutionEngine class (state machine per execution.md)
# TODO: resolve_to_ibkr(intent: TradeIntent) -> ib_insync.Option
# TODO: execute_intent(intent: TradeIntent) -> ExecutionState
# TODO: Order tracking, fill monitoring, bracket management
# TODO: Reconnect recovery, orphan detection
# TODO: Audit trail (JSONL logging of all state transitions)
