"""Physical metrics: distance covered, speed zones, high-speed running,
sprints — and the inputs to B1's headline honest result.

Requires kinematics columns (``speed``, ``step_dist``) already added by
``ftm.kinematics.add_kinematics``.

Thresholds are CONVENTIONAL, not universal — providers disagree on them.
That disagreement is the point of the sensitivity analysis. All values
from ``metrics.yaml``:
    physical:
      hsr_threshold_mps: 5.5
      sprint_threshold_mps: 7.0
      sprint_min_duration_s: 1.0
      sprint_min_recovery_s: 1.0     # gap required between two sprints
      speed_zones_mps: [[0,2],[2,4],[4,5.5],[5.5,7.0],[7.0,99]]

IMPORTANT (spec step 5): do NOT report distance-covered metrics for
SkillCorner players — broadcast tracking makes them non-comparable to
Metrica, and an analyst reading the repo will spot it instantly. The
provider gate lives in ``pipeline.py`` / the report writer, but keep
these functions honest by accepting a ``provider`` arg and refusing.
"""

from __future__ import annotations

import pandas as pd


def distance_covered(
    df_with_kin: pd.DataFrame,
    *,
    provider: str,
) -> pd.DataFrame:
    """Total distance per player (and per period).

    Raise or return an empty frame with a warning column when
    ``provider == "skillcorner"`` — see module note.

    Returns: (track_id, team, period, distance_m) and a rolled-up
    (track_id, team, distance_m).

    Gate (spec step 4): Metrica full-match per-player totals in 9–12 km.
    """
    raise NotImplementedError


def speed_zones(
    df_with_kin: pd.DataFrame,
    *,
    zones_mps: list[tuple[float, float]],
    provider: str,
) -> pd.DataFrame:
    """Distance per player in each speed band.

    Thin wrapper over ``ftm.kinematics.distance_by_speed_zone`` that adds
    human-readable zone labels from config and applies the same
    SkillCorner refusal as ``distance_covered``.

    Returns: (track_id, team, zone_label, distance_m).
    """
    raise NotImplementedError


def high_speed_running(
    df_with_kin: pd.DataFrame,
    *,
    threshold_mps: float,
    provider: str,
) -> pd.DataFrame:
    """Distance covered above ``threshold_mps`` (default 5.5) per player.

    Returns: (track_id, team, hsr_distance_m). SkillCorner refused.
    """
    raise NotImplementedError


def detect_sprints(
    df_with_kin: pd.DataFrame,
    *,
    threshold_mps: float,
    min_duration_s: float,
    min_recovery_s: float,
    frame_rate: float,
) -> pd.DataFrame:
    """Find sprint efforts per player.

    A sprint = a run of consecutive frames with ``speed >= threshold_mps``
    lasting at least ``min_duration_s``. Two candidate runs separated by
    less than ``min_recovery_s`` below threshold are merged into one
    effort (prevents double-counting a single sprint that dips for one
    noisy frame).

    Algorithm per (track_id, period), sorted by timestamp:
    1. Boolean mask ``speed >= threshold``.
    2. Close gaps shorter than ``min_recovery_s * frame_rate`` frames.
    3. Label contiguous True runs; keep those with duration
       >= ``min_duration_s``.
    4. One row per surviving run.

    Returns: (track_id, team, period, start_ts, end_ts, duration_s,
    peak_speed_mps, mean_speed_mps, distance_m).

    A ``sprint_count`` per player is just a group-by size on this. That
    count is what the sensitivity sweep perturbs — see
    ``count_sprints_grid``.
    """
    raise NotImplementedError


def sprint_count(sprints: pd.DataFrame) -> pd.DataFrame:
    """Collapse ``detect_sprints`` output to counts per player.

    Returns: (track_id, team, n_sprints, total_sprint_distance_m).
    """
    raise NotImplementedError


def count_sprints_grid(
    df_with_kin_by_window: dict[int, pd.DataFrame],
    *,
    threshold_grid_mps: list[float],
    min_duration_s: float,
    min_recovery_s: float,
    frame_rate: float,
) -> pd.DataFrame:
    """Sprint counts across a (smoothing window x sprint threshold) grid.

    This produces the numbers behind B1's headline sentence:
    *"sprint count varies by N% across defensible threshold choices, so I
    report the parameters alongside the number."*

    Parameters
    ----------
    df_with_kin_by_window:
        Map ``savgol_window_frames -> canonical frame re-run through
        ``add_kinematics`` with that window``. Built by
        ``scripts/validate.py``.
    threshold_grid_mps:
        Sprint-speed thresholds to sweep, e.g. [6.5, 7.0, 7.5, 8.0].

    Returns
    -------
    tidy frame: (savgol_window_frames, sprint_threshold_mps,
    total_sprints, sprints_per_player_mean). ``scripts/validate.py``
    turns this into ``reports/sensitivity.json`` and the README table,
    including the spread as a percentage of the mid-grid value (the "N%").
    """
    raise NotImplementedError
