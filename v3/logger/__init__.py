"""Stage 1 opportunity-surface logger.

The logger produces an exhaustive, append-only dataset of every eligible bar
in every session, capturing teacher outputs, full-chain guardrail outcomes,
selected contracts under guardrails, and (via a later pass) oracle labels.

`schema.py` defines the record types. `writer.py` / `reader.py` will come
later once the schema is locked. No logger code is wired to real data until
the schema has been reviewed.
"""
