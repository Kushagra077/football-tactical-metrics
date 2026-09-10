"""Synthetic-trajectory tests for ``ftm.kinematics``.

Spec step 2: these are the only cheap way to know the physical metrics
are right. CI runs them on every push. The noisy-stationary test is the
important one — it is the difference between measuring player movement
and measuring tracking noise, and it is exactly the failure that makes
naive distance-covered numbers 30% too high.
"""

from __future__ import annotations


def test_straight_line_distance():
    """Player at exactly 5 m/s for 10 s -> total ``step_dist`` == 50 m
    within tolerance (say +/- 1 m after smoothing lag)."""
    raise NotImplementedError


def test_noisy_stationary_distance_is_near_zero():
    """Stationary player + Gaussian position noise (sd ~5 cm) -> total
    distance << the summed frame-to-frame noise, and close to 0 (e.g.
    < 2 m over 10 s). A naive ``diff().abs().sum()`` on this input would
    return tens of meters."""
    raise NotImplementedError


def test_circular_path_circumference():
    """Player on a circle of known radius for one revolution -> total
    distance == 2*pi*r within tolerance."""
    raise NotImplementedError


def test_dt_comes_from_frame_rate_not_hardcoded():
    """Same synthetic path sampled at 10 Hz and at 25 Hz yields the same
    total distance (proves ``dt = 1 / frame_rate`` is honoured, not a
    hardcoded 0.04)."""
    raise NotImplementedError


def test_impossible_speeds_are_clipped_and_counted():
    """Inject a single-frame 40 m/s jump -> ``speed`` is capped at
    ``max_speed_mps`` and the returned ``ClipReport.n_clipped`` >= 1."""
    raise NotImplementedError


def test_gap_is_not_interpolated_into_a_teleport():
    """A player missing for 3 s then reappearing 40 m away does not add a
    40 m step (the gap is treated as a fresh sub-track)."""
    raise NotImplementedError


def test_speed_zones_partition_total_distance():
    """Distance summed across all speed zones == total distance for the
    same player (zones tile the speed axis with no gaps/overlaps)."""
    raise NotImplementedError
