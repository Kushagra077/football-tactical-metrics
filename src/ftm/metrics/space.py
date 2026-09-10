"""Voronoi space control — the visual payoff and the hard scope stop.

Per frame: partition the pitch by nearest player; the area of each
player's cell (clipped to the pitch rectangle) is the space they control.
Aggregate per team.

STOP HERE. The moment cells get weighted by player velocity or reaction
time, this is B2 pitch control, not B1. Voronoi only.

Config (``metrics.yaml``):
    space:
      pitch_length_m: 105.0
      pitch_width_m: 68.0
      include_gk: true          # keepers occupy space too; document either way

The trap
--------
``scipy.spatial.Voronoi`` returns unbounded cells for the outer points —
their ``regions`` entries contain ``-1`` (point at infinity). Do NOT try
to reconstruct those vertices by hand. Instead build each cell as a
Shapely polygon and INTERSECT it with the pitch rectangle. Two robust
routes:
  * clip the finite-region polygons, then for unbounded ones fall back to
    a large bounding box before intersecting; or
  * add 4 far-away mirror points so every real player's cell becomes
    finite, then clip.
Document which you chose in ``design_decision.md``.
"""

from __future__ import annotations

import pandas as pd


def voronoi_areas_frame(
    frame: pd.DataFrame,
    *,
    pitch_length_m: float,
    pitch_width_m: float,
    include_gk: bool,
) -> pd.DataFrame:
    """Per-player controlled area for a SINGLE frame.

    Parameters
    ----------
    frame:
        Rows for one (period, frame_id): all players (ball excluded).
        Needs >= 2 distinct player positions; return empty / NaN
        otherwise.
    pitch_length_m, pitch_width_m:
        Rectangle is ``[-L/2, L/2] x [-W/2, W/2]``.
    include_gk:
        If False, drop ``is_gk`` rows before the tessellation.

    Steps:
    1. Collect player ``(x, y)`` and ids. Deduplicate exact coincident
       points (jitter them by 1e-6 or merge) — Qhull rejects duplicates.
    2. ``scipy.spatial.Voronoi`` on the points.
    3. For each input point, build its region polygon; clip unbounded
       regions via the bounding-box / mirror-point trick above.
    4. ``shapely`` intersect every cell with the pitch rectangle;
       ``.area`` is the controlled area.

    Returns: (track_id, team, area_m2) for this frame.

    Correctness check used by the gate: the returned areas sum to
    ``pitch_length_m * pitch_width_m`` (7140) within fp tolerance.
    """
    raise NotImplementedError


def team_space_control(
    df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Per-frame area controlled by each team, plus the per-player detail.

    Loops ``voronoi_areas_frame`` over every ball-alive frame (this is the
    expensive metric — ``build_cache.py`` should downsample first).

    Returns two things (as a tuple, or one wide + one long frame):
      * per-frame team totals: (period, frame_id, timestamp, team,
        area_m2, area_share)  where the two teams' ``area_m2`` sum to 7140
      * per-player: (period, frame_id, timestamp, track_id, team, area_m2)
        for the pitch-snapshot overlay in the dashboard.

    Gate (spec step 6): summed team areas == 105*68 within fp tolerance.
    Treat that as a real assertion in tests, not a vibe check.
    """
    raise NotImplementedError
