"""Known-answer tests for ``ftm.metrics`` using hand-placed formations.

Trivial to write, and they catch sign errors that a plot happily hides
(spec step 3).
"""

from __future__ import annotations

# --- shape ---------------------------------------------------------------------


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


# --- pressing ----------------------------------------------------------------


def test_ball_carrier_is_nearest_player_within_radius():
    """Ball at origin, closest player 2 m away (radius 3) -> that player
    is the carrier; move them to 5 m -> no carrier that frame."""
    raise NotImplementedError


def test_pressure_count_counts_only_opponents_in_radius():
    """Carrier on home; 2 away players within 5 m, 1 away player at 8 m,
    1 home team-mate at 1 m -> ``n_pressers`` == 2."""
    raise NotImplementedError


# --- space (Voronoi) -------------------------------------------------------


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


# --- physical --------------------------------------------------------------


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
