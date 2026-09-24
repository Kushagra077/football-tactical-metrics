"""Team-shape metrics: line height, width, length, compactness, centroid.

Computed PER FRAME, PER TEAM, over outfield players only (``~is_gk``),
ball-alive frames only (``ball_state == "alive"``). The ball row is
excluded throughout.

Filtering contract: every individual metric function drops ball and GK
rows itself (cheap and idempotent), but does NOT filter on
``ball_state`` -- it computes over whatever frames it is given.
``compute_all`` applies the ball-alive filter once before calling them.

Orientation convention (must be established by the loader):
coordinates are static for the whole match, origin at center, +x toward
one fixed goal. Each team has an "attacking direction" ``d``: ``+1`` if it
attacks toward +x, ``-1`` if toward -x.

Line-height sign convention: values are expressed in each team's OWN
frame of reference, ``line_height = d * mean(x of the n deepest players)``
where "deepest" = smallest ``d * x``. Own goal line is therefore always
at -52.5 and a deeper line is always a more negative number, whichever
way the team attacks. Example: a home back four at x = -30 attacking +x
gives -30; an away back four at x = +30 attacking -x also gives -30.
Width, length, hull area and centroid are reported in the raw static
pitch frame (width/length/area are sign-free anyway).

Every threshold ("4 deepest") comes from ``metrics.yaml``:
    shape:
      line_height_n_deepest: 4
      min_outfield_players: 8      # below this, flag the frame (SkillCorner)
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

KEYS = ["period", "frame_id", "timestamp", "team"]
OUTPUT_COLUMNS = [
    *KEYS,
    "line_height_m",
    "width_m",
    "length_m",
    "hull_area_m2",
    "cx_m",
    "cy_m",
    "n_outfield",
    "low_outfield_count",
]

Direction = Mapping[str, int] | Mapping[tuple[str, int], int]


def _outfield(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ball and GK rows; keep only the key and coordinate columns."""
    mask = ~df["is_ball"] & ~df["is_gk"] & (df["team"] != "ball")
    out = df.loc[mask, [*KEYS, "x_pitch", "y_pitch"]].copy()
    out["team"] = out["team"].astype(str)
    return out


def _group(of: pd.DataFrame):
    return of.groupby(KEYS, sort=True, observed=True)


def _direction_per_row(of: pd.DataFrame, attacking_direction: Direction) -> np.ndarray:
    """Map each row to its team's attacking direction. Keys may be a team
    name (applies to all periods) or a ``(team, period)`` tuple; the tuple
    form wins when both are present."""
    for key, d in attacking_direction.items():
        if d not in (1, -1):
            raise ValueError(f"attacking_direction[{key!r}] must be +1 or -1, got {d!r}")
    per_pair = {}
    pairs = of[["team", "period"]].drop_duplicates()
    for team, period in pairs.itertuples(index=False):
        key = (team, int(period))
        if key in attacking_direction:
            per_pair[key] = int(attacking_direction[key])
        elif team in attacking_direction:
            per_pair[key] = int(attacking_direction[team])
        else:
            raise KeyError(f"no attacking direction for team={team!r} period={period}")
    idx = pd.MultiIndex.from_arrays([of["team"], of["period"].astype(int)])
    return pd.Series(per_pair, dtype=float).reindex(idx).to_numpy()


def _line_height(of: pd.DataFrame, n_deepest: int, attacking_direction: Direction) -> pd.DataFrame:
    if n_deepest < 1:
        raise ValueError(f"n_deepest must be >= 1, got {n_deepest}")
    if of.empty:
        return pd.DataFrame(columns=[*KEYS, "line_height_m"])
    of = of.assign(_depth=_direction_per_row(of, attacking_direction) * of["x_pitch"].to_numpy())
    of = of.sort_values([*KEYS, "_depth"], kind="stable")
    deepest = of[of.groupby(KEYS, sort=False, observed=True).cumcount() < n_deepest]
    return _group(deepest)["_depth"].mean().rename("line_height_m").reset_index()


def line_height(
    df: pd.DataFrame,
    *,
    n_deepest: int,
    attacking_direction: Direction,
) -> pd.DataFrame:
    """Per-frame defensive line height for each team.

    Definition: ``d * mean(x_pitch)`` of the ``n_deepest`` outfield players
    with the smallest ``d * x_pitch`` (those nearest the team's own goal),
    where ``d`` is the team's attacking direction. "Deeper" is therefore
    more negative in every team's own frame (see module docstring). If a
    team has fewer than ``n_deepest`` outfield players in a frame, all of
    them are used.

    Parameters
    ----------
    df:
        Canonical frame. Ball and GK rows are dropped here; ``ball_state``
        is NOT filtered (``compute_all`` does that).
    n_deepest:
        From ``shape.line_height_n_deepest``.
    attacking_direction:
        ``{"home": +1 or -1, "away": ...}`` -- which way each team attacks
        in the static coordinate system. Keys may alternatively be
        ``(team, period)`` tuples for per-period directions.

    Returns
    -------
    tidy frame: (period, frame_id, timestamp, team, line_height_m)
    """
    return _line_height(_outfield(df), n_deepest, attacking_direction)


def _extent(of: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    g = _group(of)[col]
    return (g.max() - g.min()).rename(name).reset_index()


def width(df: pd.DataFrame) -> pd.DataFrame:
    """Per-frame team width = ``y_pitch.max() - y_pitch.min()`` over
    outfield players (ball/GK dropped here; ``ball_state`` not filtered).

    Returns: (period, frame_id, timestamp, team, width_m).
    """
    return _extent(_outfield(df), "y_pitch", "width_m")


def length(df: pd.DataFrame) -> pd.DataFrame:
    """Per-frame team length = ``x_pitch.max() - x_pitch.min()`` over
    outfield players (ball/GK dropped here; ``ball_state`` not filtered).

    Returns: (period, frame_id, timestamp, team, length_m).
    """
    return _extent(_outfield(df), "x_pitch", "length_m")


def _hull_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return np.nan
    try:
        return float(ConvexHull(points).volume)
    except QhullError:
        return np.nan


def _compactness(of: pd.DataFrame) -> pd.DataFrame:
    if of.empty:
        return pd.DataFrame(columns=[*KEYS, "hull_area_m2"])
    of = of.sort_values(KEYS, kind="stable")
    keys = of[KEYS].reset_index(drop=True)
    starts = np.flatnonzero(_group(keys).cumcount().to_numpy() == 0)
    xy = of[["x_pitch", "y_pitch"]].to_numpy()
    out = keys.iloc[starts].reset_index(drop=True)
    out["hull_area_m2"] = [_hull_area(chunk) for chunk in np.split(xy, starts[1:])]
    return out


def compactness(df: pd.DataFrame) -> pd.DataFrame:
    """Per-frame team compactness = convex-hull area (m^2) of outfield
    players (ball/GK dropped here; ``ball_state`` not filtered).

    ``scipy.spatial.ConvexHull(...).volume`` is the polygon area in 2-D.
    Degenerate cases -- < 3 players, or all collinear/coincident
    (``QhullError``) -- give NaN for that frame.

    Returns: (period, frame_id, timestamp, team, hull_area_m2).
    """
    return _compactness(_outfield(df))


def _centroid(of: pd.DataFrame) -> pd.DataFrame:
    return _group(of).agg(cx_m=("x_pitch", "mean"), cy_m=("y_pitch", "mean")).reset_index()


def centroid(df: pd.DataFrame) -> pd.DataFrame:
    """Per-frame team centroid = mean (x, y) of outfield players, in the
    static pitch frame (ball/GK dropped here; ``ball_state`` not filtered).

    Returns: (period, frame_id, timestamp, team, cx_m, cy_m).
    """
    return _centroid(_outfield(df))


def infer_attacking_direction(df: pd.DataFrame) -> dict[tuple[str, int], int]:
    """Infer each team's attacking direction per ``(team, period)`` from
    the canonical frame alone.

    Reference x per (team, period) = mean ``x_pitch`` of that team's GK
    rows (all frames, alive or dead), falling back to the outfield mean x
    when the team has no GK rows. When both teams are present in a
    period, the team whose reference is further toward -x attacks +x and
    the other attacks -x. With only one team present, it attacks +x iff
    its reference is on the negative half (x <= 0).

    Inference is per period (not per match) so it stays correct even if a
    loader's orientation flips at half time; under STATIC_HOME_AWAY the
    result is identical for both periods.
    """
    mask = ~df["is_ball"] & (df["team"] != "ball")
    players = df.loc[mask, ["team", "period", "is_gk", "x_pitch"]]
    players = players.assign(team=players["team"].astype(str), period=players["period"].astype(int))
    gk_ref = players[players["is_gk"]].groupby(["period", "team"])["x_pitch"].mean()
    all_ref = players[~players["is_gk"]].groupby(["period", "team"])["x_pitch"].mean()
    ref = gk_ref.combine_first(all_ref)

    result: dict[tuple[str, int], int] = {}
    for period, sub in ref.groupby(level="period"):
        sub = sub.droplevel("period")
        if len(sub) == 2:
            deep, high = sub.sort_values().index
            result[(deep, int(period))] = 1
            result[(high, int(period))] = -1
        else:
            for team, x in sub.items():
                result[(team, int(period))] = 1 if x <= 0 else -1
    return result


def compute_all(
    df: pd.DataFrame,
    cfg: dict,
    *,
    attacking_direction: Direction | None = None,
) -> pd.DataFrame:
    """Run every shape metric and return one wide per-(frame, team) frame.

    Applies the standard filters once (``~is_gk``, ``~is_ball``,
    ``ball_state == "alive"``), then joins the individual metric outputs
    on (period, frame_id, timestamp, team). Adds ``n_outfield`` and a
    boolean ``low_outfield_count`` column (``n_outfield <
    cfg["shape"]["min_outfield_players"]``); such frames are flagged,
    not dropped.

    ``attacking_direction`` as in :func:`line_height`; when None it is
    inferred per (team, period) from the full (unfiltered) frame by
    :func:`infer_attacking_direction`.

    Columns: period, frame_id, timestamp, team, line_height_m, width_m,
    length_m, hull_area_m2, cx_m, cy_m, n_outfield, low_outfield_count.
    ``team`` is a plain string column ("home"/"away").

    This is what ``pipeline.py`` calls.
    """
    shape_cfg = cfg["shape"]
    n_deepest = int(shape_cfg["line_height_n_deepest"])
    min_outfield = int(shape_cfg["min_outfield_players"])

    if attacking_direction is None:
        attacking_direction = infer_attacking_direction(df)

    of = _outfield(df[df["ball_state"] == "alive"])
    if of.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    g = _group(of)
    ymin, ymax = g["y_pitch"].min(), g["y_pitch"].max()
    xmin, xmax = g["x_pitch"].min(), g["x_pitch"].max()
    base = pd.DataFrame(
        {
            "width_m": ymax - ymin,
            "length_m": xmax - xmin,
            "cx_m": g["x_pitch"].mean(),
            "cy_m": g["y_pitch"].mean(),
            "n_outfield": g.size(),
        }
    ).reset_index()
    lh = _line_height(of, n_deepest, attacking_direction)
    hull = _compactness(of)
    out = base.merge(lh, on=KEYS, how="left").merge(hull, on=KEYS, how="left")
    out["n_outfield"] = out["n_outfield"].astype("int64")
    out["low_outfield_count"] = out["n_outfield"] < min_outfield
    return out[OUTPUT_COLUMNS]
