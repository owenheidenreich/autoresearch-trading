from __future__ import annotations

import pytest

from v5.research.prior_art import LoopError, blocked_by_prior_art, prior_art_check


def test_do_not_retest_hit_blocks() -> None:
    assert blocked_by_prior_art(prior_art_check("score coverage threshold"))


def test_rejected_protocol_lineage_hit_blocks() -> None:
    hits = prior_art_check("loss-only damage-control exit")
    assert hits
    assert blocked_by_prior_art(hits)


def test_novel_term_is_quiet() -> None:
    assert prior_art_check("a mechanism nobody has ever proposed xyzzy") == []


def test_blank_mechanism_fails_closed() -> None:
    with pytest.raises(LoopError):
        prior_art_check("   ")
