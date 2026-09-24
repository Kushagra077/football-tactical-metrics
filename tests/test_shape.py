"""Known-answer tests for ``ftm.metrics.shape`` using hand-placed data.

Trivial to write, and they catch sign errors that a plot happily hides
(spec step 3).
"""

from __future__ import annotations


def test_line_height_flat_back_four():
    """Flat back four at x = -30 (plus midfield/attack further up, GK
    behind) -> ``line_height`` for that team == -30 within tolerance."""
    raise NotImplementedError


def test_line_height_sign_follows_attacking_direction():
    """The same physical shape, with the team's attacking direction
    flipped, returns the sign-consistent value (a deeper line stays the
    'deeper' number). Guards the second-half orientation bug."""
    raise NotImplementedError


def test_line_height_excludes_goalkeeper():
    """Moving only the GK far back does not change ``line_height``
    (proves ``~is_gk`` filtering)."""
    raise NotImplementedError


def test_width_and_length_on_a_known_rectangle():
    """Outfield players placed on a 40 m (y) by 25 m (x) box -> width ==
    40, length == 25."""
    raise NotImplementedError


def test_compactness_equals_known_hull_area():
    """Players on the vertices of a 20x20 square -> hull area == 400 m^2;
    add an interior player -> area unchanged."""
    raise NotImplementedError


def test_compactness_degenerate_frames_return_nan():
    """< 3 players, or all collinear -> NaN, not a crash."""
    raise NotImplementedError


def test_shape_metrics_ignore_dead_ball_frames():
    """A frame with ``ball_state == "dead"`` is excluded from
    ``compute_all`` output."""
    raise NotImplementedError


def test_low_outfield_count_frames_are_flagged_not_dropped():
    """A frame with 6 outfield players (SkillCorner-like) still produces
    shape rows but with ``low_outfield_count == True``."""
    raise NotImplementedError
