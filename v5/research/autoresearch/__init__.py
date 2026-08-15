"""Alpha-budgeted autonomous research.

A loop that runs experiments until one succeeds is a machine for manufacturing
false positives: with enough attempts something always passes. This project has
the receipts -- 77 rejected fair-contract attempts, a validation PF of 4.454
whose diagnostic was 1.008, an offline 2.142 that scored 0.72 on unseen days.

The fix is not to run fewer experiments. Multiplicity is cheap: on a ten-year
option corpus, ten thousand experiments cost only 2.7 accuracy points over one.
The fix is to *count* them, so the bar the winner must clear rises with every
attempt, and to stop when that bar passes what is plausibly reachable.
"""
from __future__ import annotations

__all__ = ["budget", "experiment", "generator", "outer"]


def __getattr__(name: str):
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(name)
