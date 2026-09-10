"""Team-shape metrics: line height, width, length, compactness, centroid.

Computed PER FRAME, PER TEAM, over outfield players only (``~is_gk``),
ball-alive frames only (``ball_state == "alive"``). The ball row is
excluded throughout.

Orientation convention (must be established by the loader):
coordinates are static for the whole match, origin at center, +x toward
one fixed goal. Each team has an "attacking direction" (+x or -x). "Signed
toward own goal" for line height means: express the value so that a
deeper defensive line is a smaller number regardless of which way the
team attacks. Encode that sign flip HERE and document it; a plot will
happily hide a sign error.

Every threshold ("4 deepest") comes from ``metrics.yaml``:
    shape:
      line_height_n_deepest: 4
      min_outfield_players: 8      # below this, flag the frame (SkillCorner)
"""

from __future__ import annotations

import pandas as pd


def line_height(
    df: pd.DataFrame,
    *,
    n_deepest: int,
    attacking_direction: dict[str, int],
) -> pd.DataFrame:
    """Per-frame defensive line height for each team.

    Definition: mean ``x_pitch`` of the ``n_deepest`` outfield players
    (those nearest the team's own goal), signed so that "deeper" is more
    negative in a common frame of reference.

    Parameters
    ----------
    df:
        Canonical frame (already filtered to outfield + ball-alive, or
        filter inside and document it).
    n_deepest:
        From ``shape.line_height_n_deepest``.
    attacking_direction:
        ``{"home": +1 or -1, "away": ...}`` — which way each team attacks
        in the static coordinate system. Used to pick "deepest" (min or
        max x) and to apply the sign convention.

    Returns
    -------
    tidy frame: (period, frame_id, timestamp, team, line_height_m)

    Test (``tests/test_metrics.py``): a hand-placed flat back four at
    x = -30 must return -30 (within tol) for that team.

    Gate (spec step 3): plotted over 90 minutes it sits deep when
    defending and pushes up when the team dominates. A flat line means
    the orientation transform did not take — go back to the loader.
    """
    raise NotImplementedError


def width(df: pd.DataFrame) -> pd.DataFrame:
    """Per-frame team width = ``y_pitch.max() - y_pitch.min()`` over
    outfield players.

    Returns: (period, frame_id, timestamp, team, width_m).
    """
    raise NotImplementedError


def length(df: pd.DataFrame) -> pd.DataFrame:
    """Per-frame team length = ``x_pitch.max() - x_pitch.min()`` over
    outfield players.

    Returns: (period, frame_id, timestamp, team, length_m).
    """
    raise NotImplementedError


def compactness(df: pd.DataFrame) -> pd.DataFrame:
    """Per-frame team compactness = convex-hull area (m^2) of outfield
    players.

    Use ``scipy.spatial.ConvexHull``; its ``.volume`` is the polygon area
    in 2-D. Guard the degenerate cases: < 3 players, or all collinear
    (``ConvexHull`` raises ``QhullError``) -> return NaN for that frame.

    Returns: (period, frame_id, timestamp, team, hull_area_m2).
    """
    raise NotImplementedError


def centroid(df: pd.DataFrame) -> pd.DataFrame:
    """Per-frame team centroid = mean (x, y) of outfield players.

    Returns: (period, frame_id, timestamp, team, cx_m, cy_m). The
    dashboard plots this as a trace over the time window.
    """
    raise NotImplementedError


def compute_all(
    df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Run every shape metric and return one wide per-(frame, team) frame.

    Applies the standard filters once (``~is_gk``, ``~is_ball``,
    ``ball_state == "alive"``), then joins the individual metric outputs
    on (period, frame_id, timestamp, team). Adds a boolean
    ``low_outfield_count`` column when fewer than
    ``cfg["shape"]["min_outfield_players"]`` outfield players are present
    (relevant for SkillCorner; shape metrics stay usable but flagged).

    This is what ``pipeline.py`` calls.
    """
    raise NotImplementedError
