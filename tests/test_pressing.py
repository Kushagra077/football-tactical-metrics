"""Known-answer tests for ``ftm.metrics.pressing`` using hand-placed data.

Trivial to write, and they catch sign errors that a plot happily hides
(spec step 3).
"""

from __future__ import annotations


def test_ball_carrier_is_nearest_player_within_radius():
    """Ball at origin, closest player 2 m away (radius 3) -> that player
    is the carrier; move them to 5 m -> no carrier that frame."""
    raise NotImplementedError


def test_pressure_count_counts_only_opponents_in_radius():
    """Carrier on home; 2 away players within 5 m, 1 away player at 8 m,
    1 home team-mate at 1 m -> ``n_pressers`` == 2."""
    raise NotImplementedError
