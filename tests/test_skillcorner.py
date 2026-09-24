"""SkillCorner loader tests.

Offline tests cover the pure reshaping helpers (wide -> long, detection
filter) on hand-made inputs; the integration test downloads a real match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ftm.loaders.skillcorner import (
    PlayerInfo,
    SkillCornerLoader,
    detected_pairs,
    keep_detected,
    wide_to_long,
)
from ftm.schema import coerce, validate

PLAYERS = {
    "11": PlayerInfo(team="home", jersey_number=1, is_gk=True),
    "22": PlayerInfo(team="away", jersey_number=None, is_gk=False),
    "33": PlayerInfo(team="away", jersey_number=9, is_gk=False),  # not in wide
}


def _wide() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_id": [10, 11, 12, 13],
            "period_id": [1, 1, 2, 2],
            "timestamp": pd.to_timedelta([0.0, 0.1, 0.0, 0.1], unit="s"),
            "ball_state": ["alive", None, "dead", "alive"],
            "ball_x": [0.0, np.nan, 1.0, 2.0],
            "ball_y": [0.0, np.nan, 1.0, 2.0],
            "11_x": [-50.0, -50.1, np.nan, 50.0],
            "11_y": [0.0, 0.1, np.nan, 0.0],
            "22_x": [10.0, np.nan, 12.0, 13.0],
            "22_y": [5.0, np.nan, 6.0, 7.0],
        }
    )


def test_wide_to_long_emits_only_tracked_rows_and_validates():
    long_df = wide_to_long(_wide(), PLAYERS)
    df = validate(coerce(long_df))

    assert len(df) == 3 + 3 + 3  # ball, GK, player 22 each miss one frame
    assert set(df["track_id"]) == {"ball", "11", "22"}
    assert df.loc[df["track_id"] == "11", "frame_id"].tolist() == [10, 11, 13]
    assert df.loc[df["track_id"] == "22", "frame_id"].tolist() == [10, 12, 13]

    gk = df[df["track_id"] == "11"]
    assert gk["is_gk"].all() and (gk["team"] == "home").all()
    assert (gk["jersey_number"] == 1).all()
    assert df.loc[df["track_id"] == "22", "jersey_number"].isna().all()

    ball = df[df["is_ball"]]
    assert (ball["team"] == "ball").all() and not ball["is_gk"].any()

    frame11 = df[df["frame_id"] == 11]
    assert (frame11["ball_state"] == "dead").all()  # missing state -> dead
    assert df.loc[df["frame_id"] == 12, "timestamp"].eq(0.0).all()


def test_wide_to_long_drops_non_canonical_periods():
    wide = _wide()
    wide.loc[3, "period_id"] = 3
    df = wide_to_long(wide, PLAYERS)
    assert 13 not in set(df["frame_id"])


def test_detected_pairs_and_keep_detected_drop_extrapolated_positions():
    records = [
        {
            "frame": 10,
            "period": 1,
            "ball_data": {"x": 0.0, "y": 0.0, "is_detected": True},
            "player_data": [
                {"player_id": 11, "x": -50.0, "y": 0.0, "is_detected": True},
                {"player_id": 22, "x": 10.0, "y": 5.0, "is_detected": False},
            ],
        },
        {"frame": 11, "period": None, "ball_data": {"x": None}, "player_data": []},
        {
            "frame": 13,
            "period": 2,
            "ball_data": {"x": 2.0, "y": 2.0, "is_detected": False},
            "player_data": [
                {"player_id": 11, "x": 50.0, "y": 0.0, "is_detected": True},
                {"player_id": 22, "x": 13.0, "y": 7.0, "is_detected": True},
            ],
        },
    ]
    detected = detected_pairs(records)
    assert set(zip(detected["frame_id"], detected["track_id"], strict=True)) == {
        (10, "11"),
        (10, "ball"),
        (13, "11"),
        (13, "22"),
    }

    kept = validate(coerce(keep_detected(wide_to_long(_wide(), PLAYERS), detected)))
    assert set(zip(kept["frame_id"], kept["track_id"], strict=True)) == {
        (10, "11"),
        (10, "ball"),
        (13, "11"),
        (13, "22"),
    }


@pytest.mark.integration
def test_skillcorner_match_loads_validates_and_has_coverage_gaps():
    """(integration, network) A SkillCorner match passes ``validate`` and
    at least one player has coverage < 100% in a period (proves broadcast
    drop-out is preserved, not forward-filled)."""
    loader = SkillCornerLoader()
    match_ids = loader.list_matches()
    assert len(match_ids) >= 1
    match_id = match_ids[0]

    meta = loader.load_meta(match_id)
    assert meta.provider == "skillcorner"
    assert meta.frame_rate > 0

    df = validate(loader.load(match_id))
    players = df[~df["is_ball"]]
    assert set(players["team"]) == {"home", "away"}

    frames_per_period = df.groupby("period")["frame_id"].nunique()
    present = players.groupby(["period", "track_id"])["frame_id"].nunique()
    coverage = present / frames_per_period.reindex(present.index.get_level_values("period")).values
    assert (coverage < 1.0).any()
