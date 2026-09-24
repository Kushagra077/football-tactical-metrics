"""Known-answer tests for ``ftm.metrics.physical`` using hand-placed data.

Trivial to write, and they catch sign errors that a plot happily hides
(spec step 3).
"""

from __future__ import annotations


def test_distance_covered_refused_for_skillcorner():
    """``physical.distance_covered(..., provider="skillcorner")`` raises
    or returns an explicitly-flagged empty frame — never a silent
    number."""
    raise NotImplementedError


def test_sprint_detection_respects_duration_and_recovery():
    """A 0.5 s burst above threshold is NOT a sprint; a 1.5 s burst IS;
    two 1.2 s bursts 0.4 s apart merge into ONE sprint
    (recovery < ``min_recovery_s``)."""
    raise NotImplementedError


def test_sprint_count_grid_spread_is_reported():
    """``count_sprints_grid`` over a small window x threshold grid returns
    one row per cell and the counts are monotonic in threshold (higher
    threshold -> fewer sprints)."""
    raise NotImplementedError
