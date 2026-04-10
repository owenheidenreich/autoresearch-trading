# v2 Execution Roadmap

This file is a roadmap note, not a description of a current running system.

## Current Truth

- `v2/live/*` is still mostly stubbed
- there is no active live or paper-trading service in this repo
- replay is the only fully implemented execution path

## What This Means

Humans and agents should not read this file as the current runtime contract.

For current behavior use:

- [current_state.md](current_state.md)
- [evaluator.md](evaluator.md)
- [contracts.md](contracts.md)

## Planned Future Scope

When live execution is implemented, this area will need to define:

- broker contract resolution
- order lifecycle
- bracket management
- reconnect handling
- kill switch behavior
- audit logging

Until that exists in code, this is intentionally just a marker that live execution remains future work.
