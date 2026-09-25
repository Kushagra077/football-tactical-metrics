"""Tests for ``ftm.pipeline``: config loading, coverage, downsampling, and
the full ``run_match``/``run_all`` orchestration against a fake, in-memory
loader (no network, no kloppy — the abstraction boundary is enforced
separately in ``test_loaders.py``).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ftm import pipeline
from ftm.loaders.base import BaseLoader, MatchMeta
from ftm.schema import coerce
from tests.conftest import METRICS_YAML, flat_back_four


def test_load_config_returns_the_real_yaml():
    cfg = pipeline.load_config(METRICS_YAML)
    assert cfg["pitch"]["length_m"] == 105.0
    assert "savgol_window_frames" in cfg["kinematics"]


def test_load_config_reports_missing_section(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("pitch:\n  length_m: 105.0\n  width_m: 68.0\n")
    with pytest.raises(KeyError, match="kinematics"):
        pipeline.load_config(bad)


def test_load_config_reports_missing_key(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "pitch: {length_m: 105.0, width_m: 68.0}\n"
        "kinematics: {savgol_window_frames: 7}\n"
        "shape: {line_height_n_deepest: 4, min_outfield_players: 8}\n"
        "physical: {hsr_threshold_mps: 5.5, sprint_threshold_mps: 7.0, "
        "sprint_min_duration_s: 1.0, sprint_min_recovery_s: 1.0, speed_zones_mps: []}\n"
        "pressing: {press_radius_m: 5.0, carrier_search_radius_m: 3.0}\n"
        "space: {include_gk: true}\n"
    )
    with pytest.raises(KeyError, match="savgol_polyorder"):
        pipeline.load_config(bad)


def _match_frame(n_frames: int = 130, frame_rate: float = 25.0) -> pd.DataFrame:
    """A flat back four tiled across many identical frames — enough for
    kinematics/shape/pressing/space to run without crashing. Not meant to
    exercise metric correctness (that's ``test_kinematics``/``test_shape``
    etc.), just the pipeline's wiring."""
    base = flat_back_four(x_line=-30.0)
    frames = []
    for i in range(n_frames):
        frame = base.copy()
        frame["frame_id"] = i
        frame["timestamp"] = i / frame_rate
        frames.append(frame)
    return coerce(pd.concat(frames, ignore_index=True))


class _FakeLoader(BaseLoader):
    name = "fake"

    def __init__(self, frame_rate: float = 25.0, n_frames: int = 130):
        self._frame_rate = frame_rate
        self._df = _match_frame(n_frames=n_frames, frame_rate=frame_rate)

    def list_matches(self) -> list[str]:
        return ["m1"]

    def load_meta(self, match_id: str) -> MatchMeta:
        return MatchMeta(
            match_id=match_id,
            provider=self.name,
            frame_rate=self._frame_rate,
            pitch_length_m=105.0,
            pitch_width_m=68.0,
            home_team="Home FC",
            away_team="Away FC",
            periods={1: (0.0, len(self._df) / self._frame_rate)},
        )

    def load(self, match_id: str) -> pd.DataFrame:
        return self._df


def test_coverage_table_full_and_partial_tracking():
    df = _match_frame(n_frames=10)
    # Drop one home player from the second half of the match.
    dropped = df[(df["track_id"] == "h_f1") & (df["frame_id"] >= 5)].index
    df = df.drop(index=dropped).reset_index(drop=True)

    per_period, per_match = pipeline.coverage_table(df)
    full_player = per_period[per_period["track_id"] == "h_gk"]
    assert full_player["coverage_pct"].iloc[0] == pytest.approx(100.0)
    half_player = per_match[per_match["track_id"] == "h_f1"]
    assert half_player["coverage_pct"].iloc[0] == pytest.approx(50.0)
    assert "ball" not in per_match["track_id"].to_numpy()


def test_downsample_keeps_every_nth_frame_and_stays_monotonic():
    df = _match_frame(n_frames=50, frame_rate=25.0)
    out = pipeline.downsample(df, from_hz=25.0, to_hz=5.0)
    kept_ids = sorted(out["frame_id"].unique())
    assert kept_ids == list(range(0, 50, 5))
    assert out.groupby("frame_id").size().nunique() == 1  # same player count every frame


def test_downsample_rejects_upsampling():
    df = _match_frame(n_frames=10)
    with pytest.raises(ValueError, match="upsample"):
        pipeline.downsample(df, from_hz=5.0, to_hz=25.0)


def test_run_match_writes_every_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "get_loader", lambda name: _FakeLoader())
    cfg = pipeline.PipelineConfig(
        metrics_yaml_path=METRICS_YAML, cache_dir=tmp_path, target_hz=5.0
    )
    artifacts = pipeline.run_match("fake", "m1", cfg)

    for path in (
        artifacts.tracking, artifacts.shape, artifacts.pressing,
        artifacts.space, artifacts.space_player, artifacts.physical,
        artifacts.coverage, artifacts.meta,
    ):
        assert path.exists()

    tracking = pd.read_parquet(artifacts.tracking)
    assert {"vx", "vy", "speed", "step_dist"} <= set(tracking.columns)
    # Downsampled from 25 -> 5 Hz: every 5th frame kept.
    assert set(tracking.loc[~tracking["is_ball"], "frame_id"].unique()) == set(range(0, 130, 5))

    physical = pd.read_parquet(artifacts.physical)
    assert set(physical.columns) >= {
        "track_id", "team", "distance_m", "hsr_distance_m", "provider", "match_id",
    }
    assert (physical["provider"] == "fake").all()

    coverage = pd.read_parquet(artifacts.coverage)
    assert set(coverage.columns) >= {"track_id", "team", "coverage_pct", "provider", "match_id"}

    space_team = pd.read_parquet(artifacts.space)
    totals = space_team.groupby(["period", "frame_id"])["area_m2"].sum()
    assert totals.to_numpy() == pytest.approx(105.0 * 68.0, rel=1e-6)

    meta = json.loads(artifacts.meta.read_text())
    assert meta["source_frame_rate_hz"] == 25.0
    assert meta["cached_frame_rate_hz"] == 5.0
    assert meta["match_meta"]["home_team"] == "Home FC"
    assert "n_clipped" in meta["clip_report"]


def test_run_match_refuses_physical_for_skillcorner(tmp_path, monkeypatch):
    loader = _FakeLoader()
    loader.name = "skillcorner"
    monkeypatch.setattr(pipeline, "get_loader", lambda name: loader)
    cfg = pipeline.PipelineConfig(
        metrics_yaml_path=METRICS_YAML, cache_dir=tmp_path, target_hz=5.0
    )
    artifacts = pipeline.run_match("skillcorner", "m1", cfg)
    physical = pd.read_parquet(artifacts.physical)
    assert len(physical) == 0
    assert set(physical.columns) >= {"provider", "match_id"}

    meta = json.loads(artifacts.meta.read_text())
    assert "not reported" in meta["physical_note"]


def test_run_match_can_skip_space(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "get_loader", lambda name: _FakeLoader())
    cfg = pipeline.PipelineConfig(
        metrics_yaml_path=METRICS_YAML, cache_dir=tmp_path, target_hz=5.0, include_space=False,
    )
    artifacts = pipeline.run_match("fake", "m1", cfg)
    assert pd.read_parquet(artifacts.space).empty
    assert pd.read_parquet(artifacts.space_player).empty


def test_run_all_collects_and_reraises_errors(tmp_path, monkeypatch):
    class _BrokenLoader(_FakeLoader):
        def load(self, match_id: str) -> pd.DataFrame:
            raise RuntimeError("boom")

    def _get(name: str):
        return _BrokenLoader() if name == "metrica" else _FakeLoader()

    monkeypatch.setattr(pipeline, "get_loader", _get)
    cfg = pipeline.PipelineConfig(
        metrics_yaml_path=METRICS_YAML, cache_dir=tmp_path, target_hz=5.0
    )
    with pytest.raises(RuntimeError, match="metrica/m1"):
        pipeline.run_all(cfg)
