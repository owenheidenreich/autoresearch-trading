# Owner scope ruling — long single-leg SPXW 0DTE only

**Date:** 2026-08-14  
**Authority:** Owen Heidenreich, project owner  
**Owner instruction:** “its a long call or long put SPXW model only.”

This supersedes every active proposal for futures, longer tenor, short premium, debit spreads, credit
spreads, iron flies, iron condors or any other multileg structure.

The only permitted trading game is:

- observe SPXW and, optionally, SPX context available at the decision time;
- while flat, choose `WAIT` or buy to open one SPXW **0DTE** call or put;
- while holding, choose `HOLD` or sell to close that same long contract;
- never write an option, sell short, add a second leg or trade another instrument; and
- retain the $10,000 account, measured ask-in/bid-out execution, fees, causality and chronological
  validation laws.

The number of model heads remains an empirical architecture question. The morning/afternoon distinction
must be represented and audited, but this ruling does not force four independent networks.

Old spread, next-expiry and MES packets remain historical evidence only. They are not active routes and
cannot satisfy the goal.
