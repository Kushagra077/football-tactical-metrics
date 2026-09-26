"""Synthetic-trajectory tests for ``ftm.kinematics``.

These are the only cheap way to know the physical metrics
are right. CI runs them on every push. The noisy-stationary test is the
important one — it is the difference between measuring player movement
and measuring tracking noise, and it is exactly the failure that makes
naive distance-covered numbers 30% too high.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from ftm.kinematics import (
    KinematicsConfig,
    add_kinematics,
    distance_by_speed_zone,
    total_distance,
)
from ftm.schema import coerce
from tests.conftest import (
    circular_track,
    make_canonical_frame,
    noisy_stationary_track,
    straight_line_track,
)


def _cfg(metrics_cfg: dict, frame_rate: float = 25.0) -> KinematicsConfig:
    return KinematicsConfig.from_metrics_cfg(metrics_cfg, frame_rate)


def _distance(df: pd.DataFrame, cfg: KinematicsConfig) -> float:
    kin, _ = add_kinematics(df, cfg)
    return float(total_distance(kin)["distance_m"].sum())


def test_straight_line_distance(metrics_cfg):
    """Player at exactly 5 m/s for 10 s -> total ``step_dist`` == 50 m
    within tolerance (say +/- 1 m after smoothing lag)."""
    kin, report = add_kinematics(straight_line_track(speed_mps=5.0, duration_s=10.0),
                                 _cfg(metrics_cfg))
    assert total_distance(kin)["distance_m"].sum() == pytest.approx(50.0, abs=1.0)
    assert kin["speed"].to_numpy() == pytest.approx(5.0, abs=0.01)
    assert kin["step_dist"].iloc[0] == 0.0
    assert report.n_clipped == 0
    assert report.n_total == len(kin)
    assert all(kin[c].dtype == "float64" for c in ("vx", "vy", "speed", "step_dist"))


def test_noisy_stationary_distance_is_near_zero(metrics_cfg):
    """Stationary player + Gaussian position noise (sd ~5 cm) -> total
    distance << the summed frame-to-frame noise, and close to 0 (e.g.
    < 2 m over 10 s). A naive ``diff().abs().sum()`` on this input would
    return tens of meters."""
    df = noisy_stationary_track(noise_sd_m=0.05, duration_s=10.0, seed=0)
    naive = float(np.hypot(df["x_pitch"].diff(), df["y_pitch"].diff()).sum())
    assert naive > 15.0

    production = _distance(df, _cfg(metrics_cfg))
    assert production < naive / 5

    widest = max(metrics_cfg["physical"]["sensitivity"]["savgol_window_frames_grid"])
    smooth_cfg = dataclasses.replace(_cfg(metrics_cfg), savgol_window_frames=widest)
    assert _distance(df, smooth_cfg) < 2.0


def test_circular_path_circumference(metrics_cfg):
    """Player on a circle of known radius for one revolution -> total
    distance == 2*pi*r within tolerance."""
    radius = 9.15
    df = circular_track(radius_m=radius, revolutions=1.0, speed_mps=4.0)
    assert _distance(df, _cfg(metrics_cfg)) == pytest.approx(2 * np.pi * radius, rel=0.01)


def test_dt_comes_from_frame_rate_not_hardcoded(metrics_cfg):
    """Same synthetic path sampled at 10 Hz and at 25 Hz yields the same
    total distance (proves ``dt = 1 / frame_rate`` is honoured, not a
    hardcoded 0.04). Radius/speed are chosen so the revolution period is
    long relative to the smoothing window at BOTH rates (the default
    window is 21 frames; at 10 Hz that is 2.1 s, large enough
    relative to a fast lap to bias the two rates apart by more than the
    old 1% tolerance for its own sake, not because dt is wrong)."""
    dists = {}
    for rate in (10.0, 25.0):
        df = circular_track(radius_m=30.0, speed_mps=2.0, frame_rate=rate)
        kin, _ = add_kinematics(df, _cfg(metrics_cfg, rate))
        assert kin["speed"].median() == pytest.approx(2.0, rel=0.01)
        dists[rate] = float(total_distance(kin)["distance_m"].sum())
    assert dists[10.0] == pytest.approx(dists[25.0], rel=0.01)
    assert _cfg(metrics_cfg, 10.0).dt == pytest.approx(0.1)


def test_impossible_speeds_are_clipped_and_counted(metrics_cfg):
    """With glitch rejection off, a 200 m/s jump -> ``speed`` is capped at
    ``max_speed_mps`` and ``ClipReport.n_clipped`` >= 1. Clipping is now
    the backstop behind glitch rejection (which would split the track at
    this jump first), so the backstop is tested with rejection disabled.
    200 m/s, not 40: the 21-frame window smooths a one-frame spike far
    below the ceiling."""
    rate = 25.0
    cfg = dataclasses.replace(_cfg(metrics_cfg, rate), glitch_speed_mps=np.inf)
    df = straight_line_track(speed_mps=5.0, duration_s=10.0, frame_rate=rate)
    jump_m = 200.0 / rate
    df.loc[df.index >= 100, "x_pitch"] += jump_m

    kin, report = add_kinematics(df, cfg)
    assert report.n_clipped >= 1
    assert report.max_speed_seen_mps > cfg.max_speed_mps
    assert report.fraction == pytest.approx(report.n_clipped / report.n_total)
    assert kin["speed"].max() == pytest.approx(cfg.max_speed_mps)
    assert (kin["step_dist"] <= cfg.max_speed_mps * cfg.dt + 1e-12).all()
    assert np.hypot(kin["vx"], kin["vy"]).to_numpy() == pytest.approx(kin["speed"].to_numpy())


def test_single_frame_spike_is_rejected_not_smoothed_in(metrics_cfg):
    """One frame displaced 5 m sideways (125 m/s in and out) on a 5 m/s
    run: that frame gets NaN kinematics, the track is split around it, no
    speed near it exceeds the true 5 m/s, and nothing is clipped."""
    rate = 25.0
    cfg = _cfg(metrics_cfg, rate)
    df = straight_line_track(speed_mps=5.0, duration_s=10.0, frame_rate=rate)
    df.loc[df.index == 100, "y_pitch"] += 5.0

    kin, report = add_kinematics(df, cfg)
    assert report.n_glitch_steps == 2
    assert report.n_glitch_frames == 1
    assert report.n_clipped == 0
    assert np.isnan(kin.loc[100, "speed"]) and np.isnan(kin.loc[100, "step_dist"])
    assert kin["speed"].max() == pytest.approx(5.0, abs=0.01)
    # Only the two steps into/out of the spike are lost (0.2 m each).
    assert total_distance(kin)["distance_m"].sum() == pytest.approx(50.0 - 0.4, abs=0.05)


def test_tracker_slide_splits_the_track(metrics_cfg):
    """A player jogging at 3 m/s whose track 'slides' 20 m over 6 frames
    (~83 m/s per step, the Metrica pattern) and continues from there: the
    slide contributes no distance and no speed, the 5 intermediate frames
    are NaN, and speed on both sides stays at the true 3 m/s."""
    rate = 25.0
    cfg = _cfg(metrics_cfg, rate)
    df = straight_line_track(speed_mps=3.0, duration_s=10.0, frame_rate=rate)
    offset = np.zeros(len(df))
    offset[100:106] = np.linspace(20.0 / 6, 20.0, 6)
    offset[106:] = 20.0
    df["y_pitch"] = df["y_pitch"] + offset

    kin, report = add_kinematics(df, cfg)
    assert report.n_glitch_steps == 6
    assert report.n_glitch_frames == 5
    assert kin.loc[100:104, "speed"].isna().all()
    assert kin["speed"].max() == pytest.approx(3.0, abs=0.01)
    assert total_distance(kin)["distance_m"].sum() == pytest.approx(30.0 - 6 * 0.12, abs=0.05)


def test_gap_is_not_interpolated_into_a_teleport(metrics_cfg):
    """A player missing for 3 s then reappearing 40 m away does not add a
    40 m step (the gap is treated as a fresh sub-track)."""
    rate = 25.0
    cfg = _cfg(metrics_cfg, rate)
    n_before, n_gap, n_after = 125, int(3 * rate), 125
    df = make_canonical_frame(
        n_frames=n_before + n_gap + n_after,
        frame_rate=rate,
        home_xy={"p1": lambda i: (-20.0, 0.0) if i < n_before else (20.0, 0.0)},
        ball_xy=None,
        gk_ids=set(),
    )
    missing = (df["frame_id"] >= n_before) & (df["frame_id"] < n_before + n_gap)
    df = coerce(df.loc[~missing].reset_index(drop=True))

    kin, report = add_kinematics(df, cfg)
    assert total_distance(kin)["distance_m"].sum() < 1.0
    assert report.n_clipped == 0
    assert kin.loc[kin["frame_id"] == n_before + n_gap, "step_dist"].item() == 0.0


def test_speed_zones_partition_total_distance(metrics_cfg):
    """Distance summed across all speed zones == total distance for the
    same player (zones tile the speed axis with no gaps/overlaps)."""
    rate = 25.0
    n = int(20 * rate)
    t = np.arange(n) / rate
    x = -40.0 + np.cumsum(np.abs(8.0 * np.sin(t / 3.0))) / rate

    def xy_home(i: int) -> tuple[float, float]:
        return float(x[i]), 5.0

    df = make_canonical_frame(
        n_frames=n,
        frame_rate=rate,
        home_xy={"p1": xy_home, "gk": (-50.0, 0.0)},
        ball_xy=lambda i: (float(np.sin(i)) * 30.0, 0.0),
    )
    zones = metrics_cfg["physical"]["speed_zones_mps"]
    kin, _ = add_kinematics(df, _cfg(metrics_cfg, rate))
    by_zone = distance_by_speed_zone(kin, zones)
    totals = total_distance(kin).set_index("track_id")["distance_m"]

    assert set(by_zone["track_id"]) == {"p1", "gk"}
    assert len(by_zone) == 2 * len(zones)
    assert list(by_zone.loc[by_zone["track_id"] == "p1", "zone_label"]) == [
        "0-2", "2-4", "4-5.5", "5.5-7", "7-99"
    ]
    zone_sums = by_zone.groupby("track_id")["distance_m"].sum()
    assert zone_sums["p1"] == pytest.approx(totals["p1"])
    assert zone_sums["gk"] == pytest.approx(totals["gk"]) == pytest.approx(0.0, abs=1e-9)
    p1_zones = by_zone[by_zone["track_id"] == "p1"].set_index("zone_label")["distance_m"]
    assert (p1_zones > 0).sum() >= 3
    assert kin.loc[kin["is_ball"], ["vx", "vy", "speed", "step_dist"]].isna().all().all()
    assert "ball" not in totals.index


def test_short_dropout_inside_a_subtrack_does_not_inflate_speed(metrics_cfg):
    """Frames missing for less than ``max_gap_s`` are bridged on the
    ``dt`` grid, not treated as a single ``dt`` step."""
    rate = 25.0
    cfg = _cfg(metrics_cfg, rate)
    df = straight_line_track(speed_mps=2.0, duration_s=10.0, frame_rate=rate)
    n_drop = int(0.8 * rate)
    assert n_drop / rate < cfg.max_gap_s
    df = coerce(df.drop(index=range(100, 100 + n_drop)).reset_index(drop=True))

    kin, report = add_kinematics(df, cfg)
    assert total_distance(kin)["distance_m"].sum() == pytest.approx(20.0, abs=0.5)
    assert kin["speed"].to_numpy() == pytest.approx(2.0, abs=0.05)
    assert report.n_clipped == 0


def test_config_rejects_polyorder_that_zeroes_the_derivative(metrics_cfg):
    with pytest.raises(ValueError, match="polyorder"):
        dataclasses.replace(_cfg(metrics_cfg), savgol_polyorder=0)
