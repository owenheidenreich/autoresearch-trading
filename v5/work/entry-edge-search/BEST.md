# Best configuration so far

**None. The search has not started.**

The baseline every iteration must beat, measured 2026-08-13 on 35,586 trades over 745 sessions
([receipt](../../../v4/audit/autoresearch/full_system_2026_08_13/receipt.json)):

| Configuration | Gross/trade | Net/trade |
|---|---:|---:|
| Optimal-stopping exit, random entry | **−$0.2** | −$23.2 |
| Optimal-stopping exit, best trained entry (greeks + path, top 25%) | **+$0.2** | −$22.8 |

**The bar is a gross figure clearly above $0.00**, with a session-block bootstrap lower bound above zero
after multiplicity correction, beating a trade-count-matched random selection and a shuffled-label null.

Update this file only when an iteration is KEPT.
