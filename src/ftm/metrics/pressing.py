"""Pressing metric: opponents near the ball carrier.

Kept deliberately small — B1's pressing story is "how many opponents are
close to whoever has the ball", nothing more. Weighting by velocity or
angle is B2 territory (scope guard).

Possession here is a PROXIMITY PROXY: without event data we call the
nearest outfield player to the ball the "carrier" when they are within
``carrier_search_radius_m``. Loose balls, passes in flight and aerial
duels therefore show up as no-carrier frames, and a player shielding
near a loose ball can be mislabelled as the carrier.

Null policy: frames with no carrier are kept with ``n_pressers`` = <NA>
(not 0), so "mean pressers" is averaged over carrier frames only rather
than dragged down by loose-ball frames.

Config (``metrics.yaml``):
    pressing:
      press_radius_m: 5.0        # counts as "pressing" within this radius
      carrier_search_radius_m: 3.0   # ball-to-player distance to call it possession
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FRAME_KEYS = ["period", "frame_id"]
OUTPUT_KEYS = ["period", "frame_id", "timestamp"]


def _alive(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["ball_state"] == "alive"]


def _frames(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[OUTPUT_KEYS]
        .drop_duplicates(FRAME_KEYS)
        .sort_values(FRAME_KEYS)
        .reset_index(drop=True)
    )


def _dist(dx: pd.Series, dy: pd.Series) -> np.ndarray:
    return np.hypot(dx.to_numpy(dtype=float), dy.to_numpy(dtype=float))


def identify_ball_carrier(
    df: pd.DataFrame,
    *,
    search_radius_m: float,
) -> pd.DataFrame:
    """Per frame, name the player closest to the ball (if close enough).

    For each ball-alive frame: take the ball ``(x, y)``, find the nearest
    outfield player (``~is_gk``) across BOTH teams; if that distance <=
    ``search_radius_m`` call them the carrier, else no carrier that frame.
    Frames without a ball row also get no carrier.

    This is a proximity proxy for possession (no event data), not true
    possession.

    Returns: (period, frame_id, timestamp, carrier_track_id, carrier_team,
    ball_to_carrier_m). Frames with no carrier still appear, with nulls.
    """
    df = _alive(df)
    frames = _frames(df)

    ball = (
        df.loc[df["is_ball"], [*FRAME_KEYS, "x_pitch", "y_pitch"]]
        .drop_duplicates(FRAME_KEYS)
        .rename(columns={"x_pitch": "ball_x", "y_pitch": "ball_y"})
    )
    players = df.loc[~df["is_ball"] & ~df["is_gk"], [*FRAME_KEYS, "track_id", "team", "x_pitch",
                                                     "y_pitch"]]
    pb = players.merge(ball, on=FRAME_KEYS, how="inner")
    pb["ball_to_carrier_m"] = _dist(pb["x_pitch"] - pb["ball_x"], pb["y_pitch"] - pb["ball_y"])

    nearest = pb.sort_values("ball_to_carrier_m", kind="stable").drop_duplicates(FRAME_KEYS)
    nearest = nearest[nearest["ball_to_carrier_m"] <= search_radius_m]
    nearest = nearest.rename(columns={"track_id": "carrier_track_id", "team": "carrier_team"})

    out = frames.merge(
        nearest[[*FRAME_KEYS, "carrier_track_id", "carrier_team", "ball_to_carrier_m"]],
        on=FRAME_KEYS,
        how="left",
    )
    return out[[*OUTPUT_KEYS, "carrier_track_id", "carrier_team", "ball_to_carrier_m"]]


def pressure_count(
    df: pd.DataFrame,
    carriers: pd.DataFrame,
    *,
    press_radius_m: float,
) -> pd.DataFrame:
    """Per frame, count opponents within ``press_radius_m`` of the carrier.

    Opponents = non-ball players whose ``team`` differs from
    ``carrier_team``. GKs are INCLUDED (a keeper stepping out to press is
    real); the ball is excluded.

    ``carriers`` is the output of :func:`identify_ball_carrier`; every one
    of its rows appears in the result. Frames with no carrier get
    ``n_pressers`` = <NA> and ``nearest_opponent_m`` = NaN. Carrier frames
    with no opponents tracked get ``n_pressers`` = 0 and NaN distance.

    Returns: (period, frame_id, timestamp, carrier_team, n_pressers,
    nearest_opponent_m). ``n_pressers`` is nullable ``Int64``.
    """
    base = carriers[[*OUTPUT_KEYS, "carrier_track_id", "carrier_team"]].reset_index(drop=True)
    with_carrier = base.dropna(subset=["carrier_track_id"])

    players = df.loc[~df["is_ball"], [*FRAME_KEYS, "track_id", "team", "x_pitch", "y_pitch"]]
    carrier_pos = with_carrier[[*FRAME_KEYS, "carrier_track_id", "carrier_team"]].merge(
        players.rename(
            columns={"track_id": "carrier_track_id", "x_pitch": "cx", "y_pitch": "cy"}
        ).drop(columns="team"),
        on=[*FRAME_KEYS, "carrier_track_id"],
        how="inner",
    )
    pairs = players.merge(carrier_pos, on=FRAME_KEYS, how="inner")
    is_opp = pairs["team"].astype(str).to_numpy() != pairs["carrier_team"].astype(str).to_numpy()
    opp = pairs.loc[is_opp, FRAME_KEYS].copy()
    d = _dist(
        pairs.loc[is_opp, "x_pitch"] - pairs.loc[is_opp, "cx"],
        pairs.loc[is_opp, "y_pitch"] - pairs.loc[is_opp, "cy"],
    )
    opp["d"] = d
    opp["in_radius"] = d <= press_radius_m
    agg = (
        opp.groupby(FRAME_KEYS, sort=False)
        .agg(n_pressers=("in_radius", "sum"), nearest_opponent_m=("d", "min"))
        .reset_index()
    )

    out = base.merge(agg, on=FRAME_KEYS, how="left")
    has_carrier = out["carrier_track_id"].notna().to_numpy()
    n = out["n_pressers"].astype("Float64").fillna(0).astype("Int64")
    out["n_pressers"] = n.where(has_carrier, pd.NA)
    out["nearest_opponent_m"] = out["nearest_opponent_m"].astype(float)
    return out[[*OUTPUT_KEYS, "carrier_team", "n_pressers", "nearest_opponent_m"]]


def compute_all(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run carrier ID + pressure count, return one per-frame frame.

    Called by ``pipeline.py``. Applies ball-alive filtering; leaves GKs
    in as pressers (carriers are outfield only). Output is keyed on
    (period, frame_id, timestamp) so it joins onto the shape-metric frame,
    with columns carrier_track_id, carrier_team, ball_to_carrier_m,
    n_pressers, nearest_opponent_m.
    """
    pcfg = cfg["pressing"]
    df = _alive(df)
    carriers = identify_ball_carrier(df, search_radius_m=pcfg["carrier_search_radius_m"])
    pressure = pressure_count(df, carriers, press_radius_m=pcfg["press_radius_m"])
    return carriers.merge(
        pressure.drop(columns="carrier_team"), on=OUTPUT_KEYS, how="left", validate="one_to_one"
    )


def summarize_pressing(per_frame: pd.DataFrame, *, double_team_min: int = 2) -> pd.DataFrame:
    """Aggregate :func:`compute_all` output per team being pressed.

    One row per ``carrier_team`` (the team in possession, i.e. the team
    being pressed): ``carrier_frames`` (frames with a carrier on that
    team), ``mean_pressers`` (mean ``n_pressers`` over those frames) and
    ``double_team_rate`` (share of those frames with
    ``n_pressers >= double_team_min``). No-carrier frames are excluded.
    """
    cols = ["carrier_team", "carrier_frames", "mean_pressers", "double_team_rate"]
    pf = per_frame.dropna(subset=["carrier_team", "n_pressers"])
    if pf.empty:
        return pd.DataFrame(columns=cols)
    n = pf["n_pressers"].astype(float)
    tmp = pd.DataFrame(
        {
            "carrier_team": pf["carrier_team"].astype(str).to_numpy(),
            "n": n.to_numpy(),
            "double": (n >= double_team_min).to_numpy(dtype=float),
        }
    )
    out = (
        tmp.groupby("carrier_team", sort=True)
        .agg(carrier_frames=("n", "size"), mean_pressers=("n", "mean"),
             double_team_rate=("double", "mean"))
        .reset_index()
    )
    return out[cols]
