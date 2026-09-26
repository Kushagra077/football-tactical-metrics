"""Known-answer tests for ``ftm.metrics.shape`` using hand-placed data.

Trivial to write, and they catch sign errors that a plot happily hides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ftm.metrics import shape
from tests.conftest import flat_back_four, make_canonical_frame

TOL = 1e-9


def _value(result: pd.DataFrame, team: str, col: str) -> float:
    row = result[result["team"] == team]
    assert len(row) == 1
    return float(row[col].iloc[0])


def _mirror_x(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["x_pitch"] = -out["x_pitch"]
    return out


def test_line_height_flat_back_four(metrics_cfg):
    """Flat back four at x = -30 (plus midfield/attack further up, GK
    behind) -> ``line_height`` for that team == -30 within tolerance."""
    n = metrics_cfg["shape"]["line_height_n_deepest"]
    df = flat_back_four(x_line=-30.0, team="home")
    lh = shape.line_height(df, n_deepest=n, attacking_direction={"home": 1, "away": -1})
    assert _value(lh, "home", "line_height_m") == pytest.approx(-30.0, abs=TOL)
    # The point-mirrored away team (back four at x=+30, attacking -x) is equally deep.
    assert _value(lh, "away", "line_height_m") == pytest.approx(-30.0, abs=TOL)

    # compute_all infers the same directions from GK positions.
    out = shape.compute_all(df, metrics_cfg)
    assert _value(out, "home", "line_height_m") == pytest.approx(-30.0, abs=TOL)
    assert _value(out, "away", "line_height_m") == pytest.approx(-30.0, abs=TOL)


def test_line_height_sign_follows_attacking_direction(metrics_cfg):
    """The same physical shape, with the team's attacking direction
    flipped, returns the sign-consistent value (a deeper line stays the
    'deeper' number). Guards the second-half orientation bug."""
    n = metrics_cfg["shape"]["line_height_n_deepest"]
    deep = flat_back_four(x_line=-35.0, team="home")
    high = flat_back_four(x_line=-10.0, team="home")
    flipped = {"home": -1, "away": 1}
    lh_deep = shape.line_height(_mirror_x(deep), n_deepest=n, attacking_direction=flipped)
    lh_high = shape.line_height(_mirror_x(high), n_deepest=n, attacking_direction=flipped)
    assert _value(lh_deep, "home", "line_height_m") == pytest.approx(-35.0, abs=TOL)
    assert _value(lh_high, "home", "line_height_m") == pytest.approx(-10.0, abs=TOL)

    # Second-half flip: inference is per period, so period 2 mirrored still reads -35.
    p2 = _mirror_x(deep).assign(period=np.int8(2))
    both = pd.concat([deep, p2], ignore_index=True)
    assert shape.infer_attacking_direction(both) == {
        ("home", 1): 1, ("away", 1): -1, ("home", 2): -1, ("away", 2): 1,
    }
    out = shape.compute_all(both, metrics_cfg)
    home = out[out["team"] == "home"].set_index("period")["line_height_m"]
    assert home.loc[1] == pytest.approx(-35.0, abs=TOL)
    assert home.loc[2] == pytest.approx(-35.0, abs=TOL)


def test_line_height_excludes_goalkeeper(metrics_cfg):
    """Moving only the GK far back does not change ``line_height``
    (proves ``~is_gk`` filtering)."""
    n = metrics_cfg["shape"]["line_height_n_deepest"]
    dirs = {"home": 1, "away": -1}
    df = flat_back_four(x_line=-30.0, team="home")
    moved = df.copy()
    moved.loc[moved["track_id"] == "h_gk", "x_pitch"] = -52.0
    for frame in (df, moved):
        lh = shape.line_height(frame, n_deepest=n, attacking_direction=dirs)
        assert _value(lh, "home", "line_height_m") == pytest.approx(-30.0, abs=TOL)


def test_width_and_length_on_a_known_rectangle(metrics_cfg):
    """Outfield players placed on a 40 m (y) by 25 m (x) box -> width ==
    40, length == 25."""
    home = {
        "gk": (-50.0, 30.0),  # outside the box: must be ignored
        "a": (-20.0, -20.0),
        "b": (5.0, -20.0),
        "c": (-20.0, 20.0),
        "d": (5.0, 20.0),
        "e": (-10.0, 0.0),
    }
    df = make_canonical_frame(n_frames=1, home_xy=home, ball_xy=(40.0, 30.0))
    assert _value(shape.width(df), "home", "width_m") == pytest.approx(40.0)
    assert _value(shape.length(df), "home", "length_m") == pytest.approx(25.0)
    cen = shape.centroid(df)
    assert _value(cen, "home", "cx_m") == pytest.approx(-8.0)
    assert _value(cen, "home", "cy_m") == pytest.approx(0.0)
    out = shape.compute_all(df, metrics_cfg)
    assert _value(out, "home", "width_m") == pytest.approx(40.0)
    assert _value(out, "home", "length_m") == pytest.approx(25.0)


def test_compactness_equals_known_hull_area():
    """Players on the vertices of a 20x20 square -> hull area == 400 m^2;
    add an interior player -> area unchanged."""
    square = {"gk": (-50.0, 0.0), "a": (0.0, 0.0), "b": (20.0, 0.0),
              "c": (20.0, 20.0), "d": (0.0, 20.0)}
    df = make_canonical_frame(n_frames=1, home_xy=square)
    assert _value(shape.compactness(df), "home", "hull_area_m2") == pytest.approx(400.0)

    df2 = make_canonical_frame(n_frames=1, home_xy={**square, "e": (7.0, 12.0)})
    assert _value(shape.compactness(df2), "home", "hull_area_m2") == pytest.approx(400.0)


def test_compactness_degenerate_frames_return_nan():
    """< 3 players, or all collinear -> NaN, not a crash."""
    two = make_canonical_frame(n_frames=1, home_xy={"gk": (-50.0, 0.0), "a": (0.0, 0.0),
                                                    "b": (5.0, 5.0)})
    assert np.isnan(_value(shape.compactness(two), "home", "hull_area_m2"))

    line = {"gk": (-50.0, 0.0), **{f"p{i}": (float(i), 2.0 * i) for i in range(5)}}
    collinear = make_canonical_frame(n_frames=1, home_xy=line)
    assert np.isnan(_value(shape.compactness(collinear), "home", "hull_area_m2"))


def test_shape_metrics_ignore_dead_ball_frames(metrics_cfg):
    """A frame with ``ball_state == "dead"`` is excluded from
    ``compute_all`` output."""
    alive = flat_back_four(x_line=-30.0, team="home")
    dead = alive.copy()
    dead["frame_id"] = 1
    dead["timestamp"] = 1 / 25.0
    dead["ball_state"] = pd.Categorical(["dead"] * len(dead), categories=["alive", "dead"])
    df = pd.concat([alive, dead], ignore_index=True)
    df["ball_state"] = pd.Categorical(df["ball_state"], categories=["alive", "dead"])

    out = shape.compute_all(df, metrics_cfg)
    assert list(out.columns) == shape.OUTPUT_COLUMNS
    assert set(out["frame_id"]) == {0}
    assert len(out) == 2

    all_dead = shape.compute_all(dead, metrics_cfg)
    assert all_dead.empty
    assert list(all_dead.columns) == shape.OUTPUT_COLUMNS


def test_low_outfield_count_frames_are_flagged_not_dropped(metrics_cfg):
    """A frame with 6 outfield players (SkillCorner-like) still produces
    shape rows but with ``low_outfield_count == True``."""
    df = flat_back_four(x_line=-30.0, team="home")
    drop = ["h_m1", "h_m2", "h_m3", "h_m4"]
    df = df[~df["track_id"].isin(drop)].reset_index(drop=True)

    out = shape.compute_all(df, metrics_cfg)
    assert len(out) == 2
    home = out[out["team"] == "home"].iloc[0]
    away = out[out["team"] == "away"].iloc[0]
    assert home["n_outfield"] == 6
    assert bool(home["low_outfield_count"]) is True
    assert away["n_outfield"] == 10
    assert bool(away["low_outfield_count"]) is False
    assert home["line_height_m"] == pytest.approx(-30.0)
    assert not np.isnan(home["hull_area_m2"])
