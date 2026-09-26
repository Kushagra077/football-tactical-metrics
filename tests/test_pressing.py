"""Known-answer tests for ``ftm.metrics.pressing`` using hand-placed data.

Trivial to write, and they catch sign errors that a plot happily hides.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ftm.metrics.pressing import (
    compute_all,
    identify_ball_carrier,
    pressure_count,
    summarize_pressing,
)
from ftm.schema import validate
from tests.conftest import make_canonical_frame

GKS = {"h_gk", "a_gk"}
FAR_GKS = {"h_gk": (-50.0, 0.0), "a_gk": (50.0, 0.0)}


def _frame(home: dict, away: dict, **kwargs) -> pd.DataFrame:
    return make_canonical_frame(
        home_xy={"h_gk": FAR_GKS["h_gk"], **home},
        away_xy={"a_gk": FAR_GKS["a_gk"], **away},
        gk_ids=GKS,
        **kwargs,
    )


def test_ball_carrier_is_nearest_player_within_radius():
    """Ball at origin, closest player 2 m away (radius 3) -> that player
    is the carrier; move them to 5 m -> no carrier that frame."""
    df = _frame(
        {"h1": lambda i: (2.0, 0.0) if i == 0 else (5.0, 0.0), "h2": (10.0, 10.0)},
        {"a1": (-10.0, 10.0)},
        n_frames=2,
    )
    out = identify_ball_carrier(df, search_radius_m=3.0)
    assert list(out.columns) == [
        "period", "frame_id", "timestamp", "carrier_track_id", "carrier_team",
        "ball_to_carrier_m",
    ]
    assert len(out) == 2
    f0, f1 = out.iloc[0], out.iloc[1]
    assert f0["carrier_track_id"] == "h1"
    assert f0["carrier_team"] == "home"
    assert f0["ball_to_carrier_m"] == pytest.approx(2.0)
    assert pd.isna(f1["carrier_track_id"])
    assert pd.isna(f1["carrier_team"])
    assert pd.isna(f1["ball_to_carrier_m"])


def test_carrier_ignores_goalkeeper():
    """A GK nearer the ball than any outfield player is not the carrier."""
    df = make_canonical_frame(
        home_xy={"h_gk": (0.5, 0.0), "h1": (2.5, 0.0)},
        away_xy={"a_gk": (50.0, 0.0), "a1": (-2.0, 0.0)},
        gk_ids=GKS,
        n_frames=1,
    )
    out = identify_ball_carrier(df, search_radius_m=3.0)
    assert out.iloc[0]["carrier_track_id"] == "a1"
    assert out.iloc[0]["carrier_team"] == "away"


def test_frames_without_ball_have_no_carrier():
    df = _frame({"h1": (0.0, 0.0)}, {"a1": (1.0, 0.0)}, ball_xy=None, n_frames=3)
    out = identify_ball_carrier(df, search_radius_m=3.0)
    assert len(out) == 3
    assert out["carrier_track_id"].isna().all()


def test_pressure_count_counts_only_opponents_in_radius():
    """Carrier on home; 2 away players within 5 m, 1 away player at 8 m,
    1 home team-mate at 1 m -> ``n_pressers`` == 2."""
    df = _frame(
        {"h1": (1.0, 0.0), "h2": (1.0, 1.0)},
        {"a1": (4.0, 0.0), "a2": (1.0, -4.0), "a3": (9.0, 0.0)},
        n_frames=1,
    )
    carriers = identify_ball_carrier(df, search_radius_m=3.0)
    out = pressure_count(df, carriers, press_radius_m=5.0)
    assert list(out.columns) == [
        "period", "frame_id", "timestamp", "carrier_team", "n_pressers", "nearest_opponent_m",
    ]
    row = out.iloc[0]
    assert row["carrier_team"] == "home"
    assert row["n_pressers"] == 2
    assert row["nearest_opponent_m"] == pytest.approx(3.0)


def test_goalkeeper_counts_as_presser():
    """A keeper stepping out within the radius is a presser."""
    df = make_canonical_frame(
        home_xy={"h_gk": (-50.0, 0.0), "h1": (40.0, 0.0)},
        away_xy={"a_gk": (42.0, 0.0), "a1": (0.0, 20.0)},
        ball_xy=(40.5, 0.0),
        gk_ids=GKS,
        n_frames=1,
    )
    carriers = identify_ball_carrier(df, search_radius_m=3.0)
    out = pressure_count(df, carriers, press_radius_m=5.0)
    assert carriers.iloc[0]["carrier_track_id"] == "h1"
    assert out.iloc[0]["n_pressers"] == 1
    assert out.iloc[0]["nearest_opponent_m"] == pytest.approx(2.0)


def test_no_carrier_frame_has_null_pressers():
    """Loose-ball frames report <NA>, not 0, so means are over carrier frames."""
    df = _frame({"h1": (10.0, 0.0)}, {"a1": (11.0, 0.0)}, n_frames=2)
    carriers = identify_ball_carrier(df, search_radius_m=3.0)
    out = pressure_count(df, carriers, press_radius_m=5.0)
    assert len(out) == 2
    assert out["n_pressers"].isna().all()
    assert out["nearest_opponent_m"].isna().all()
    assert str(out["n_pressers"].dtype) == "Int64"


def test_compute_all_excludes_dead_ball_and_summarizes(metrics_cfg):
    alive = _frame(
        {"h1": (1.0, 0.0)},
        {"a1": (3.0, 0.0), "a2": (1.0, 4.0)},
        n_frames=4,
    )
    dead = _frame(
        {"h1": (1.0, 0.0)},
        {"a1": (3.0, 0.0), "a2": (1.0, 4.0)},
        n_frames=3,
        ball_state="dead",
    )
    dead["frame_id"] += 4
    dead["timestamp"] += 4 / 25.0
    loose = _frame({"h1": (20.0, 0.0)}, {"a1": (21.0, 0.0)}, n_frames=2)
    loose["frame_id"] += 7
    loose["timestamp"] += 7 / 25.0
    df = validate(pd.concat([alive, dead, loose], ignore_index=True), strict_ranges=False)

    out = compute_all(df, metrics_cfg)
    assert list(out.columns) == [
        "period", "frame_id", "timestamp", "carrier_track_id", "carrier_team",
        "ball_to_carrier_m", "n_pressers", "nearest_opponent_m",
    ]
    assert out["frame_id"].tolist() == [0, 1, 2, 3, 7, 8]
    assert (out.loc[out["frame_id"] < 4, "n_pressers"] == 2).all()
    assert out.loc[out["frame_id"] >= 7, "n_pressers"].isna().all()

    summary = summarize_pressing(out)
    assert summary["carrier_team"].tolist() == ["home"]
    row = summary.iloc[0]
    assert row["carrier_frames"] == 4
    assert row["mean_pressers"] == pytest.approx(2.0)
    assert row["double_team_rate"] == pytest.approx(1.0)


def test_summarize_pressing_mixed_counts():
    per_frame = pd.DataFrame(
        {
            "period": [1] * 5,
            "frame_id": range(5),
            "timestamp": [0.0, 0.04, 0.08, 0.12, 0.16],
            "carrier_track_id": ["h1", "h1", "a1", "a1", pd.NA],
            "carrier_team": ["home", "home", "away", "away", pd.NA],
            "ball_to_carrier_m": [1.0, 1.0, 1.0, 1.0, float("nan")],
            "n_pressers": pd.array([0, 3, 1, 2, pd.NA], dtype="Int64"),
            "nearest_opponent_m": [6.0, 1.0, 2.0, 1.5, float("nan")],
        }
    )
    s = summarize_pressing(per_frame).set_index("carrier_team")
    assert s.loc["home", "carrier_frames"] == 2
    assert s.loc["home", "mean_pressers"] == pytest.approx(1.5)
    assert s.loc["home", "double_team_rate"] == pytest.approx(0.5)
    assert s.loc["away", "mean_pressers"] == pytest.approx(1.5)
    assert s.loc["away", "double_team_rate"] == pytest.approx(0.5)
