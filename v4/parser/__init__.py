"""v4.parser — canonical contract ID parser.

See contract_id.py for the canonical form spec.
"""
from .contract_id import ContractId, from_columns, parse

__all__ = ["ContractId", "from_columns", "parse"]
