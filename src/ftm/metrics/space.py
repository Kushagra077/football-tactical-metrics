"""Voronoi space control — the visual payoff and the hard scope stop.

Per frame: partition the pitch by nearest player; the area of each
player's cell (clipped to the pitch rectangle) is the space they control.
Aggregate per team.

STOP HERE. The moment cells get weighted by player velocity or reaction
time, this is B2 pitch control, not B1. Voronoi only.

Config (``metrics.yaml``):
    pitch:
      length_m: 105.0
      width_m: 68.0
    space:
      include_gk: true          # keepers occupy space too

Unbounded cells: the mirror-point trick
---------------------------------------
``scipy.spatial.Voronoi`` returns unbounded cells (``-1`` vertices) for
the outer points. We never patch those. Instead every real point is
reflected across the 4 pitch edges and Qhull runs on the 5N points. For a
rectangle, each edge is then the perpendicular bisector between a point
and its mirror, so every real point's cell is finite and already exactly
bounded by the pitch. The cells are still intersected with the pitch
polygon in ``shapely`` to absorb floating-point drift of vertices that
should lie on the lines.

Data hygiene, applied per frame before tessellating:
  * Positions are clamped into the rectangle, inset by ``_EDGE_INSET_M``
    (players can stand a few meters off the pitch; a point exactly on an
    edge would coincide with its own mirror). The player's cell is then
    the nearest-player region of the pitch as seen from the touchline.
  * Coincident players (equal after rounding to ``_DEDUP_DECIMALS``) are
    merged into one site; the merged cell's area is split equally between
    them and each gets the same ``cell_wkt``.
  * One distinct position -> that site owns the whole pitch. No players ->
    empty result. Either way areas still sum to ``L * W`` whenever anyone
    is on the pitch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from scipy.spatial import Voronoi

_EDGE_INSET_M = 1e-3
_DEDUP_DECIMALS = 6

PLAYER_COLUMNS = ["track_id", "team", "area_m2", "cell_wkt"]
TEAM_FRAME_COLUMNS = ["period", "frame_id", "timestamp", "team", "area_m2", "area_share"]
PLAYER_FRAME_COLUMNS = [
    "period", "frame_id", "timestamp", "track_id", "team", "area_m2", "cell_wkt"
]


def _cells(
    xy: np.ndarray, length: float, width: float, with_cells: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """Areas (and WKT cells) for ``xy`` of shape (n, 2), n >= 1."""
    hx, hy = length / 2.0, width / 2.0
    pts = np.column_stack(
        [
            np.clip(xy[:, 0], -hx + _EDGE_INSET_M, hx - _EDGE_INSET_M),
            np.clip(xy[:, 1], -hy + _EDGE_INSET_M, hy - _EDGE_INSET_M),
        ]
    )
    sites, inverse, counts = np.unique(
        np.round(pts, _DEDUP_DECIMALS), axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.reshape(-1)
    pitch = shapely.box(-hx, -hy, hx, hy)
    m = len(sites)

    if m == 1:
        polys = np.array([pitch], dtype=object)
    else:
        x, y = sites[:, 0], sites[:, 1]
        mirrored = np.vstack(
            [
                sites,
                np.column_stack([-2 * hx - x, y]),
                np.column_stack([2 * hx - x, y]),
                np.column_stack([x, -2 * hy - y]),
                np.column_stack([x, 2 * hy - y]),
            ]
        )
        vor = Voronoi(mirrored)
        rings = []
        for i in range(m):
            region = vor.regions[vor.point_region[i]]
            verts = vor.vertices[region]
            c = verts.mean(axis=0)
            order = np.argsort(np.arctan2(verts[:, 1] - c[1], verts[:, 0] - c[0]))
            rings.append(verts[order])
        ring_idx = np.repeat(np.arange(m), [len(r) for r in rings])
        cells = shapely.polygons(shapely.linearrings(np.vstack(rings), indices=ring_idx))
        polys = shapely.intersection(cells, pitch)

    site_area = shapely.area(polys)
    areas = site_area[inverse] / counts[inverse]
    wkt = shapely.to_wkt(polys, rounding_precision=3)[inverse] if with_cells else None
    return areas, wkt


def voronoi_areas_frame(
    frame: pd.DataFrame,
    *,
    pitch_length_m: float,
    pitch_width_m: float,
    include_gk: bool,
    with_cells: bool = True,
) -> pd.DataFrame:
    """Per-player controlled area for a SINGLE frame.

    Parameters
    ----------
    frame:
        Rows for one (period, frame_id). Ball rows (``is_ball``) are
        dropped here if present.
    pitch_length_m, pitch_width_m:
        Rectangle is ``[-L/2, L/2] x [-W/2, W/2]``.
    include_gk:
        If False, drop ``is_gk`` rows before the tessellation.
    with_cells:
        If False, ``cell_wkt`` is None (skips WKT serialisation).

    Returns
    -------
    DataFrame with columns (track_id, team, area_m2, cell_wkt), one row
    per player, in input order. ``cell_wkt`` is the pitch-clipped cell as
    a WKT polygon string (3 decimals). No players -> empty frame with
    those columns; one distinct position -> it owns the whole pitch.

    Gate: whenever there is at least one player the areas sum to
    ``pitch_length_m * pitch_width_m`` (7140) within fp tolerance.
    """
    mask = ~frame["is_ball"].to_numpy(dtype=bool)
    if not include_gk:
        mask &= ~frame["is_gk"].to_numpy(dtype=bool)
    players = frame.loc[mask]
    if players.empty:
        return pd.DataFrame(
            {
                "track_id": pd.Series(dtype="string"),
                "team": pd.Series(dtype=frame["team"].dtype),
                "area_m2": pd.Series(dtype="float64"),
                "cell_wkt": pd.Series(dtype="object"),
            }
        )
    xy = players[["x_pitch", "y_pitch"]].to_numpy(dtype=float)
    areas, wkt = _cells(xy, pitch_length_m, pitch_width_m, with_cells)
    return pd.DataFrame(
        {
            "track_id": players["track_id"].to_numpy(),
            "team": players["team"].to_numpy(),
            "area_m2": areas,
            "cell_wkt": wkt if wkt is not None else [None] * len(areas),
        }
    ).astype({"track_id": "string", "team": frame["team"].dtype})


def team_space_control(
    df: pd.DataFrame,
    cfg: dict,
    *,
    with_cells: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-frame area controlled by each team, plus the per-player detail.

    Tessellates every ball-alive frame (ball rows excluded; keepers kept
    or dropped per ``cfg["space"]["include_gk"]``; pitch from
    ``cfg["pitch"]``). This is the expensive metric — ``build_cache.py``
    should downsample first.

    Returns
    -------
    ``(team_frame, player_frame)``:
      * ``team_frame``: (period, frame_id, timestamp, team, area_m2,
        area_share) — for every ball-alive frame with at least one player,
        one row each for "home" and "away" (0 if a team has nobody on). Per frame the two
        ``area_m2`` sum to ``L * W`` and ``area_share`` sums to 1.
      * ``player_frame``: (period, frame_id, timestamp, track_id, team,
        area_m2, cell_wkt) for the pitch-snapshot overlay. ``cell_wkt`` is
        None when ``with_cells=False``.

    Gate (spec step 6): summed team areas == 105*68 within fp tolerance.
    """
    length = float(cfg["pitch"]["length_m"])
    width = float(cfg["pitch"]["width_m"])
    include_gk = bool(cfg["space"]["include_gk"])
    total = length * width

    mask = (df["ball_state"] == "alive").to_numpy() & ~df["is_ball"].to_numpy(dtype=bool)
    if not include_gk:
        mask &= ~df["is_gk"].to_numpy(dtype=bool)
    players = df.loc[mask].sort_values(["period", "frame_id"], kind="stable")

    period = players["period"].to_numpy()
    frame_id = players["frame_id"].to_numpy()
    xy = players[["x_pitch", "y_pitch"]].to_numpy(dtype=float)
    n = len(players)
    area = np.empty(n)
    wkt = np.empty(n, dtype=object)
    if n:
        new = np.r_[True, (period[1:] != period[:-1]) | (frame_id[1:] != frame_id[:-1])]
        bounds = np.r_[np.flatnonzero(new), n]
        for s, e in zip(bounds[:-1], bounds[1:], strict=True):
            a, w = _cells(xy[s:e], length, width, with_cells)
            area[s:e] = a
            if w is not None:
                wkt[s:e] = w

    player_frame = pd.DataFrame(
        {
            "period": players["period"].to_numpy(),
            "frame_id": frame_id,
            "timestamp": players["timestamp"].to_numpy(),
            "track_id": players["track_id"].to_numpy(),
            "team": players["team"].astype(str).to_numpy(),
            "area_m2": area,
            "cell_wkt": wkt,
        }
    ).astype({"track_id": "string"})

    frames = player_frame.groupby(["period", "frame_id"], sort=True)["timestamp"].first()
    team_area = (
        player_frame.groupby(["period", "frame_id", "team"])["area_m2"]
        .sum()
        .unstack("team")
        .reindex(index=frames.index, columns=["home", "away"])
        .fillna(0.0)
    )
    team_frame = (
        team_area.stack()
        .rename("area_m2")
        .reset_index()
        .merge(frames.reset_index(), on=["period", "frame_id"])
    )
    team_frame["area_share"] = team_frame["area_m2"] / total
    team_frame["team"] = pd.Categorical(team_frame["team"], categories=["home", "away"])
    player_frame["team"] = pd.Categorical(player_frame["team"], categories=["home", "away"])
    return team_frame[TEAM_FRAME_COLUMNS], player_frame[PLAYER_FRAME_COLUMNS]
