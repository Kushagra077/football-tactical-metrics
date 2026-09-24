"""Known-answer tests for ``ftm.metrics.space`` using hand-placed data.

Trivial to write, and they catch sign errors that a plot happily hides
(spec step 3).
"""

from __future__ import annotations


def test_team_areas_sum_to_pitch_area():
    """Any frame with >= 2 players per team -> home area + away area ==
    105 * 68 == 7140 m^2 within floating-point tolerance. THE gate for
    step 6."""
    raise NotImplementedError


def test_two_symmetric_players_split_pitch_in_half():
    """One player at (-20, 0), one at (+20, 0), no others -> each controls
    3570 m^2."""
    raise NotImplementedError


def test_voronoi_handles_coincident_points():
    """Two players at the exact same coordinate -> no crash, areas still
    sum to 7140."""
    raise NotImplementedError
