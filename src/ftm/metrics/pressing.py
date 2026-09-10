"""Pressing metric: opponents near the ball carrier.

Kept deliberately small — B1's pressing story is "how many opponents are
close to whoever has the ball", nothing more. Weighting by velocity or
angle is B2 territory (scope guard).

Config (``metrics.yaml``):
    pressing:
      press_radius_m: 5.0        # counts as "pressing" within this radius
      carrier_search_radius_m: 3.0   # ball-to-player distance to call it possession
"""

from __future__ import annotations

import pandas as pd


def identify_ball_carrier(
    df: pd.DataFrame,
    *,
    search_radius_m: float,
) -> pd.DataFrame:
    """Per frame, name the player closest to the ball (if close enough).

    For each ball-alive frame: take the ball ``(x, y)``, find the nearest
    outfield player across BOTH teams; if that distance <=
    ``search_radius_m`` call them the carrier, else no carrier that frame.

    Returns: (period, frame_id, timestamp, carrier_track_id, carrier_team,
    ball_to_carrier_m). Frames with no carrier still appear, with nulls.

    Note the limitation in a docstring/README line: without event data
    this is a proximity proxy, not true possession.
    """
    raise NotImplementedError


def pressure_count(
    df: pd.DataFrame,
    carriers: pd.DataFrame,
    *,
    press_radius_m: float,
) -> pd.DataFrame:
    """Per frame, count opponents within ``press_radius_m`` of the carrier.

    Opponents = players whose ``team`` differs from ``carrier_team`` and
    are not the GK-exclusion? (keep GKs here — a keeper stepping out to
    press is real). Exclude the ball.

    Returns: (period, frame_id, timestamp, carrier_team, n_pressers,
    nearest_opponent_m).

    Aggregations the dashboard/report want (add small helpers or do it in
    the pipeline): mean pressers per frame by team, and share of frames
    with >= 2 pressers ("double-team rate").
    """
    raise NotImplementedError


def compute_all(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run carrier ID + pressure count, return one per-frame frame.

    Called by ``pipeline.py``. Applies ball-alive filtering; leaves GKs
    in. Output joins onto the shape-metric frame on
    (period, frame_id, timestamp).
    """
    raise NotImplementedError
