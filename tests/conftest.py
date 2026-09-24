"""Shared fixtures and synthetic-data builders for the test suite.

Nothing here touches a real provider. These builders are also imported by
``scripts/validate.py`` so the committed validation report exercises the
same known-answer inputs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ftm.schema import coerce

METRICS_YAML = Path(__file__).resolve().parents[1] / "configs" / "metrics.yaml"

XY = tuple[float, float]
Position = XY | Callable[[int], XY]


def _position_at(pos: Position, frame_idx: int) -> XY:
    return pos(frame_idx) if callable(pos) else pos


def make_canonical_frame(
    *,
    n_frames: int = 10,
    frame_rate: float = 25.0,
    home_xy: dict[str, Position] | None = None,
    away_xy: dict[str, Position] | None = None,
    ball_xy: Position | None = (0.0, 0.0),
    period: int = 1,
    ball_state: str = "alive",
    gk_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Build a canonical frame from hand-placed positions.

    ``home_xy`` / ``away_xy`` map ``track_id -> (x, y)`` held constant
    across all frames, or ``track_id -> callable(frame_idx) -> (x, y)``
    for moving players. ``ball_xy=None`` omits the ball. ``gk_ids``
    names the keepers explicitly; when None, the first key of each team
    dict is the GK. ``frame_id`` runs 0..n_frames-1, ``timestamp`` is
    ``frame_id / frame_rate``.

    Keep this the single place tests construct canonical data so a schema
    change only needs fixing here.
    """
    home_xy = home_xy or {}
    away_xy = away_xy or {}
    if gk_ids is None:
        gk_ids = {next(iter(d)) for d in (home_xy, away_xy) if d}

    tracks: list[tuple[str, str, Position]] = [
        (tid, "home", pos) for tid, pos in home_xy.items()
    ] + [(tid, "away", pos) for tid, pos in away_xy.items()]
    if ball_xy is not None:
        tracks.append(("ball", "ball", ball_xy))

    rows = []
    for frame_idx in range(n_frames):
        for track_id, team, pos in tracks:
            x, y = _position_at(pos, frame_idx)
            rows.append(
                {
                    "frame_id": frame_idx,
                    "period": period,
                    "timestamp": frame_idx / frame_rate,
                    "track_id": track_id,
                    "team": team,
                    "jersey_number": pd.NA,
                    "x_pitch": float(x),
                    "y_pitch": float(y),
                    "is_ball": team == "ball",
                    "is_gk": track_id in gk_ids and team != "ball",
                    "ball_state": ball_state,
                }
            )
    return coerce(pd.DataFrame(rows))


def _single_player_track(
    xs: np.ndarray, ys: np.ndarray, frame_rate: float, track_id: str = "p1"
) -> pd.DataFrame:
    n = len(xs)
    df = pd.DataFrame(
        {
            "frame_id": np.arange(n),
            "period": 1,
            "timestamp": np.arange(n) / frame_rate,
            "track_id": track_id,
            "team": "home",
            "jersey_number": pd.NA,
            "x_pitch": xs,
            "y_pitch": ys,
            "is_ball": False,
            "is_gk": False,
            "ball_state": "alive",
        }
    )
    return coerce(df)


def straight_line_track(
    *, speed_mps: float = 5.0, duration_s: float = 10.0, frame_rate: float = 25.0
) -> pd.DataFrame:
    """One player moving in +x at constant speed from x=-25. Known
    distance = ``speed_mps * duration_s``. Includes both endpoints, so
    there are ``duration_s * frame_rate + 1`` frames."""
    n = int(round(duration_s * frame_rate)) + 1
    t = np.arange(n) / frame_rate
    return _single_player_track(-25.0 + speed_mps * t, np.zeros(n), frame_rate)


def noisy_stationary_track(
    *, noise_sd_m: float = 0.05, duration_s: float = 10.0, frame_rate: float = 25.0,
    seed: int = 0,
) -> pd.DataFrame:
    """One player at the origin plus i.i.d. Gaussian position noise. True
    distance travelled ~= 0; naive differencing would report the summed
    noise. This is the test that separates 'measuring movement' from
    'measuring tracking noise'."""
    n = int(round(duration_s * frame_rate)) + 1
    rng = np.random.default_rng(seed)
    return _single_player_track(
        rng.normal(0.0, noise_sd_m, n), rng.normal(0.0, noise_sd_m, n), frame_rate
    )


def circular_track(
    *, radius_m: float = 9.15, revolutions: float = 1.0, speed_mps: float = 4.0,
    frame_rate: float = 25.0,
) -> pd.DataFrame:
    """One player on a circle centred on the origin. Known distance =
    ``2*pi*radius_m * revolutions``."""
    duration_s = 2 * np.pi * radius_m * revolutions / speed_mps
    n = int(round(duration_s * frame_rate)) + 1
    theta = np.linspace(0.0, 2 * np.pi * revolutions, n)
    return _single_player_track(radius_m * np.cos(theta), radius_m * np.sin(theta), frame_rate)


def flat_back_four(*, x_line: float = -30.0, team: str = "home") -> pd.DataFrame:
    """A single frame: GK behind, 4 defenders at ``x = x_line`` (spread in
    y), 4 midfielders and 2 forwards further up. Assumes ``team`` attacks
    toward +x, so "further up" means larger x. The other team is the
    point-mirror ``(-x, -y)`` of it, so both teams have 11 players. ``line_height`` for
    ``team`` must come back == ``x_line``."""
    own = {
        "gk": (x_line - 15.0, 0.0),
        "d1": (x_line, -20.0),
        "d2": (x_line, -7.0),
        "d3": (x_line, 7.0),
        "d4": (x_line, 20.0),
        "m1": (x_line + 15.0, -18.0),
        "m2": (x_line + 15.0, -6.0),
        "m3": (x_line + 15.0, 6.0),
        "m4": (x_line + 15.0, 18.0),
        "f1": (x_line + 28.0, -8.0),
        "f2": (x_line + 28.0, 8.0),
    }
    opp = {f"o_{tid}": (-x, -y) for tid, (x, y) in own.items()}
    own = {f"{team[0]}_{tid}": xy for tid, xy in own.items()}
    other = "away" if team == "home" else "home"
    kwargs = {f"{team}_xy": own, f"{other}_xy": opp}
    gk_ids = {f"{team[0]}_gk", "o_gk"}
    return make_canonical_frame(n_frames=1, gk_ids=gk_ids, **kwargs)


@pytest.fixture
def metrics_cfg() -> dict:
    """The parsed ``configs/metrics.yaml`` as a dict, for tests that need
    thresholds. Load the real file so tests fail if a key is renamed."""
    with METRICS_YAML.open() as fh:
        return yaml.safe_load(fh)
