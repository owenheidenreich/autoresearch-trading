# Protocol 143: IBKR LaunchAgent Activation Check

This verifies morning automation only. It does not submit orders.

- Decision: `pass_ibkr_paper_launchagents_installed`

| label | installed | loaded |
| --- | ---: | ---: |
| com.autoresearch.ibgateway.paper | `True` | `True` |
| com.autoresearch.protocol101.paper-preflight | `True` | `True` |

## Next Gate

At the next market session, confirm the preflight log shows a live IBKR paper API connection and then run live Protocol101 shadow parity before paper orders.
