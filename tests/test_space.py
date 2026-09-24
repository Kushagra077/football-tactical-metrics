"""Known-answer tests for ``ftm.metrics.space`` using hand-placed data.

Trivial to write, and they catch sign errors that a plot happily hides
(spec step 3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import shapely

from ftm.metrics.space import team_space_control, voronoi_areas_frame
from tests.conftest import flat_back_four, make_canonical_frame


def _pitch(cfg: dict) -> tuple[float, float]:
    return float(cfg["pitch"]["length_m"]), float(cfg["pitch"]["width_m"])


def _random_frames(n_frames: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # up to 3 m outside the lines, to exercise clamping
    pos = rng.uniform([-55.5, -37.0], [55.5, 37.0], size=(n_frames, 22, 2))

    def at(k: int):
        return lambda i: tuple(pos[i, k])

    return make_canonical_frame(
        n_frames=n_frames,
        home_xy={f"h{k}": at(k) for k in range(11)},
        away_xy={f"a{k}": at(11 + k) for k in range(11)},
    )


def test_team_areas_sum_to_pitch_area(metrics_cfg):
    """Any frame with >= 2 players per team -> home area + away area ==
    105 * 68 == 7140 m^2 within floating-point tolerance. THE gate for
    step 6."""
    length, width = _pitch(metrics_cfg)
    total = length * width
    assert total == pytest.approx(7140.0)

    for df in (flat_back_four(), _random_frames(25)):
        team, players = team_space_control(df, metrics_cfg)
        n_frames = df["frame_id"].nunique()
        assert len(team) == 2 * n_frames
        sums = team.groupby("frame_id", observed=True)["area_m2"].sum()
        np.testing.assert_allclose(sums.to_numpy(), total, rtol=0, atol=1e-6)
        shares = team.groupby("frame_id", observed=True)["area_share"].sum()
        np.testing.assert_allclose(shares.to_numpy(), 1.0, rtol=0, atol=1e-9)
        assert (players["area_m2"] > 0).all()
        assert not players["track_id"].eq("ball").any()
        cell_area = shapely.area(shapely.from_wkt(players["cell_wkt"].to_numpy()))
        np.testing.assert_allclose(cell_area, players["area_m2"], atol=0.05)

    # the symmetric back-four frame splits the pitch exactly in half
    team, _ = team_space_control(flat_back_four(), metrics_cfg)
    np.testing.assert_allclose(team["area_m2"], total / 2, atol=1e-6)


def test_two_symmetric_players_split_pitch_in_half(metrics_cfg):
    """One player at (-20, 0), one at (+20, 0), no others -> each controls
    3570 m^2."""
    length, width = _pitch(metrics_cfg)
    df = make_canonical_frame(n_frames=1, home_xy={"h": (-20.0, 0.0)}, away_xy={"a": (20.0, 0.0)})
    out = voronoi_areas_frame(
        df, pitch_length_m=length, pitch_width_m=width, include_gk=True
    )
    assert list(out.columns) == ["track_id", "team", "area_m2", "cell_wkt"]
    assert list(out["track_id"]) == ["h", "a"]
    np.testing.assert_allclose(out["area_m2"], [3570.0, 3570.0], atol=1e-6)
    home_cell = shapely.from_wkt(out["cell_wkt"].iloc[0])
    assert home_cell.bounds == pytest.approx((-length / 2, -width / 2, 0.0, width / 2))


def test_voronoi_handles_coincident_points(metrics_cfg):
    """Two players at the exact same coordinate -> no crash, areas still
    sum to 7140."""
    length, width = _pitch(metrics_cfg)
    kw = dict(pitch_length_m=length, pitch_width_m=width, include_gk=True)
    df = make_canonical_frame(
        n_frames=1,
        home_xy={"h1": (-20.0, 0.0), "h2": (10.0, 5.0)},
        away_xy={"a1": (20.0, 0.0), "a2": (10.0, 5.0)},
    )
    out = voronoi_areas_frame(df, **kw)
    assert out["area_m2"].sum() == pytest.approx(length * width, abs=1e-6)
    shared = out.set_index("track_id")["area_m2"]
    assert shared["h2"] == pytest.approx(shared["a2"])

    team, _ = team_space_control(df, {"pitch": {"length_m": length, "width_m": width},
                                      "space": {"include_gk": True}})
    assert team["area_m2"].sum() == pytest.approx(length * width, abs=1e-6)

    # everyone on one spot (and off the pitch): that spot owns everything
    df = make_canonical_frame(
        n_frames=1, home_xy={"h": (60.0, 40.0)}, away_xy={"a": (60.0, 40.0)}
    )
    out = voronoi_areas_frame(df, **kw)
    np.testing.assert_allclose(out["area_m2"], [3570.0, 3570.0], atol=1e-6)


def test_include_gk_false_drops_keepers(metrics_cfg):
    length, width = _pitch(metrics_cfg)
    df = flat_back_four()
    out = voronoi_areas_frame(df, pitch_length_m=length, pitch_width_m=width, include_gk=False)
    assert len(out) == 20
    assert out["area_m2"].sum() == pytest.approx(length * width, abs=1e-6)
    empty = voronoi_areas_frame(
        df.iloc[0:0], pitch_length_m=length, pitch_width_m=width, include_gk=True
    )
    assert empty.empty and list(empty.columns) == ["track_id", "team", "area_m2", "cell_wkt"]


def test_dead_ball_frames_excluded(metrics_cfg):
    df = make_canonical_frame(
        n_frames=3, home_xy={"h": (-20.0, 0.0)}, away_xy={"a": (20.0, 0.0)}, ball_state="dead"
    )
    team, players = team_space_control(df, metrics_cfg)
    assert team.empty and players.empty
