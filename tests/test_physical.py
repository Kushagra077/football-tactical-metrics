"""Known-answer tests for ``ftm.metrics.physical`` using hand-placed data.

Trivial to write, and they catch sign errors that a plot happily hides
(spec step 3).

Kinematics columns are attached by hand (``step_dist = speed / frame_rate``)
so these tests do not depend on ``ftm.kinematics``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ftm.metrics import physical
from tests.conftest import make_canonical_frame

FRAME_RATE = 10.0
BASE_SPEED = 3.0
BALL_SPEED = 20.0


def with_kinematics(df: pd.DataFrame, speeds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Attach ``speed``/``step_dist``/``vx``/``vy`` per track, indexed by
    ``frame_id``. The ball gets an absurd speed to prove it is excluded."""
    out = df.copy()
    speed = np.full(len(out), BALL_SPEED)
    for track_id, profile in speeds.items():
        rows = (out["track_id"] == track_id).to_numpy()
        speed[rows] = profile[out.loc[rows, "frame_id"].to_numpy()]
    out["speed"] = speed
    out["vx"] = speed
    out["vy"] = 0.0
    out["step_dist"] = speed / FRAME_RATE
    return out


def segments(*parts: tuple[float, float]) -> np.ndarray:
    """Concatenate ``(duration_s, speed_mps)`` constant-speed segments."""
    return np.concatenate([np.full(int(d * FRAME_RATE + 0.5), v) for d, v in parts])


def frame_for(profiles: dict[str, np.ndarray], *, period: int = 1) -> pd.DataFrame:
    n = len(next(iter(profiles.values())))
    base = make_canonical_frame(
        n_frames=n,
        frame_rate=FRAME_RATE,
        home_xy={tid: (0.0, 0.0) for tid in profiles},
        period=period,
    )
    return with_kinematics(base, profiles)


def test_distance_covered_refused_for_skillcorner():
    """``physical.distance_covered(..., provider="skillcorner")`` raises
    or returns an explicitly-flagged empty frame — never a silent
    number."""
    df = frame_for({"p1": segments((2.0, BASE_SPEED))})
    for provider in ("skillcorner", "SkillCorner", " SKILLCORNER "):
        with pytest.raises(ValueError, match="broadcast"):
            physical.distance_covered(df, provider=provider)
        with pytest.raises(ValueError, match="broadcast"):
            physical.speed_zones(df, zones_mps=[(0.0, 99.0)], provider=provider)
        with pytest.raises(ValueError, match="broadcast"):
            physical.high_speed_running(df, threshold_mps=5.5, provider=provider)


def test_distance_covered_per_period_and_total():
    p1 = frame_for({"p1": segments((4.0, 5.0)), "p2": segments((4.0, 2.0))}, period=1)
    p2 = frame_for({"p1": segments((4.0, 3.0)), "p2": segments((4.0, 2.0))}, period=2)
    df = pd.concat([p1, p2], ignore_index=True)
    df.loc[df.index[0], ["speed", "step_dist"]] = np.nan

    per_period, per_match = physical.distance_covered(df, provider="metrica")

    assert list(per_period.columns) == ["track_id", "team", "period", "distance_m"]
    assert list(per_match.columns) == ["track_id", "team", "distance_m"]
    assert "ball" not in set(per_match["track_id"])
    got = per_period.set_index(["track_id", "period"])["distance_m"]
    assert got[("p1", 1)] == pytest.approx(20.0 - 5.0 / FRAME_RATE)
    assert got[("p1", 2)] == pytest.approx(12.0)
    totals = per_match.set_index("track_id")["distance_m"]
    assert totals["p1"] == pytest.approx(32.0 - 5.0 / FRAME_RATE)
    assert totals["p2"] == pytest.approx(16.0)


def test_speed_zones_partition_distance(metrics_cfg):
    zones = metrics_cfg["physical"]["speed_zones_mps"]
    profile = segments((1.0, 1.0), (1.0, 2.0), (1.0, 5.5), (1.0, 6.0), (1.0, 7.0), (1.0, 150.0))
    df = frame_for({"p1": profile})

    out = physical.speed_zones(df, zones_mps=zones, provider="metrica")

    assert list(out["zone_label"]) == ["0-2", "2-4", "4-5.5", "5.5-7", "7+"]
    got = out.set_index("zone_label")["distance_m"]
    assert got["0-2"] == pytest.approx(1.0)
    assert got["2-4"] == pytest.approx(2.0)
    assert got["4-5.5"] == pytest.approx(0.0)
    assert got["5.5-7"] == pytest.approx(11.5)
    assert got["7+"] == pytest.approx(157.0)
    _, total = physical.distance_covered(df, provider="metrica")
    assert out["distance_m"].sum() == pytest.approx(total["distance_m"].sum())


def test_speed_zones_reject_non_tiling_bands():
    df = frame_for({"p1": segments((1.0, 1.0))})
    with pytest.raises(ValueError, match="tile"):
        physical.speed_zones(df, zones_mps=[(0.0, 2.0), (3.0, 99.0)], provider="metrica")


def test_high_speed_running_counts_only_at_or_above_threshold(metrics_cfg):
    threshold = metrics_cfg["physical"]["hsr_threshold_mps"]
    fast = segments((2.0, 4.0), (1.0, threshold), (1.0, threshold + 1.0))
    slow = segments((4.0, threshold - 0.1))
    df = frame_for({"fast": fast, "slow": slow})

    out = physical.high_speed_running(df, threshold_mps=threshold, provider="metrica")

    got = out.set_index("track_id")["hsr_distance_m"]
    assert set(got.index) == {"fast", "slow"}
    assert got["fast"] == pytest.approx(2 * threshold + 1.0)
    assert got["slow"] == 0.0


def test_sprint_detection_respects_duration_and_recovery(metrics_cfg):
    """A 0.5 s burst above threshold is NOT a sprint; a 1.5 s burst IS;
    two 1.2 s bursts 0.4 s apart merge into ONE sprint
    (recovery < ``min_recovery_s``)."""
    cfg = metrics_cfg["physical"]
    burst = cfg["sprint_threshold_mps"] + 1.0
    profile = segments(
        (2.0, BASE_SPEED),
        (0.5, burst),
        (2.0, BASE_SPEED),
        (1.5, burst),
        (2.0, BASE_SPEED),
        (1.2, burst),
        (0.4, BASE_SPEED),
        (1.2, burst),
        (2.0, BASE_SPEED),
    )
    df = frame_for({"p1": profile})

    sprints = physical.detect_sprints(
        df,
        threshold_mps=cfg["sprint_threshold_mps"],
        min_duration_s=cfg["sprint_min_duration_s"],
        min_recovery_s=cfg["sprint_min_recovery_s"],
        frame_rate=FRAME_RATE,
    )

    assert list(sprints.columns) == physical.SPRINT_COLUMNS
    assert len(sprints) == 2
    first, merged = sprints.sort_values("start_ts").itertuples(index=False)
    assert first.start_ts == pytest.approx(4.5)
    assert first.duration_s == pytest.approx(1.5)
    assert first.distance_m == pytest.approx(1.5 * burst)
    assert merged.start_ts == pytest.approx(8.0)
    assert merged.end_ts == pytest.approx(8.0 + 2.8 - 1 / FRAME_RATE)
    assert merged.duration_s == pytest.approx(2.8)
    assert merged.peak_speed_mps == pytest.approx(burst)
    assert merged.mean_speed_mps == pytest.approx((2.4 * burst + 0.4 * BASE_SPEED) / 2.8)
    assert merged.distance_m == pytest.approx(2.4 * burst + 0.4 * BASE_SPEED)

    counts = physical.sprint_count(sprints)
    assert list(counts.columns) == ["track_id", "team", "n_sprints", "total_sprint_distance_m"]
    assert counts.loc[0, "n_sprints"] == 2
    assert counts.loc[0, "total_sprint_distance_m"] == pytest.approx(sprints["distance_m"].sum())


def test_detect_sprints_empty_when_nobody_sprints(metrics_cfg):
    cfg = metrics_cfg["physical"]
    df = frame_for({"p1": segments((3.0, BASE_SPEED))})
    sprints = physical.detect_sprints(
        df,
        threshold_mps=cfg["sprint_threshold_mps"],
        min_duration_s=cfg["sprint_min_duration_s"],
        min_recovery_s=cfg["sprint_min_recovery_s"],
        frame_rate=FRAME_RATE,
    )
    assert sprints.empty
    assert list(sprints.columns) == physical.SPRINT_COLUMNS
    assert physical.sprint_count(sprints).empty


def test_sprint_count_grid_spread_is_reported(metrics_cfg):
    """``count_sprints_grid`` over a small window x threshold grid returns
    one row per cell and the counts are monotonic in threshold (higher
    threshold -> fewer sprints)."""
    cfg = metrics_cfg["physical"]
    thresholds = cfg["sensitivity"]["sprint_threshold_mps_grid"]
    windows = cfg["sensitivity"]["savgol_window_frames_grid"][:2]
    gap = (2.0, BASE_SPEED)
    raw = segments(gap, (1.5, 6.8), gap, (1.5, 7.2), gap, (1.5, 7.7), gap, (1.5, 8.2), gap)
    smoothing = {windows[0]: 1.0, windows[1]: 0.97}
    by_window = {w: frame_for({"p1": raw * k, "p2": np.full_like(raw, BASE_SPEED)})
                 for w, k in smoothing.items()}

    grid = physical.count_sprints_grid(
        by_window,
        threshold_grid_mps=thresholds,
        min_duration_s=cfg["sprint_min_duration_s"],
        min_recovery_s=cfg["sprint_min_recovery_s"],
        frame_rate=FRAME_RATE,
    )

    assert list(grid.columns) == physical.GRID_COLUMNS
    assert len(grid) == len(windows) * len(thresholds)
    for _, cell in grid.groupby("savgol_window_frames"):
        assert cell["total_sprints"].is_monotonic_decreasing
    counts = grid.set_index(["savgol_window_frames", "sprint_threshold_mps"])["total_sprints"]
    assert list(counts[windows[0]]) == [4, 3, 2, 1]
    assert list(counts[windows[1]]) == [4, 2, 1, 0]
    per_player = grid.set_index(["savgol_window_frames", "sprint_threshold_mps"])
    assert per_player.loc[(windows[0], 6.5), "sprints_per_player_mean"] == pytest.approx(2.0)

    assert physical.sprint_spread_pct(grid) == pytest.approx((4 - 0) / 3 * 100)
    assert physical.sprint_spread_pct(grid, baseline=(windows[1], 7.0)) == pytest.approx(200.0)
