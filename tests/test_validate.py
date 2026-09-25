"""Tests for ``scripts/validate.py``: synthetic checks against the shipped
metric code, and both report writers against a small cache built by the
real pipeline from a fake loader (offline)."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from scripts import validate

from ftm import pipeline
from ftm.loaders.base import BaseLoader, MatchMeta
from ftm.metrics import physical
from tests.conftest import METRICS_YAML, make_canonical_frame

FRAME_RATE = 25.0
N_FRAMES = 750  # 30 s
SPRINT_S = 4.0


def _runner(v_sprint: float, y: float, x0: float, direction: int):
    """Sprint at ``v_sprint`` for SPRINT_S seconds, then jog at 2 m/s."""

    def pos(i: int):
        t = i / FRAME_RATE
        run = v_sprint * min(t, SPRINT_S) + 2.0 * max(t - SPRINT_S, 0.0)
        return (x0 + direction * run, y)

    return pos


def _moving_match() -> pd.DataFrame:
    speeds = [6.0 + 0.35 * i for i in range(10)]  # 6.0 .. 9.15 m/s straddles every threshold
    home = {"h_gk": (-50.0, 0.0)}
    away = {"a_gk": (50.0, 0.0)}
    for i, v in enumerate(speeds):
        home[f"h_{i}"] = _runner(v, -30.0 + 6.0 * i, -50.0, +1)
        away[f"a_{i}"] = _runner(v, -27.0 + 6.0 * i, 50.0, -1)
    return make_canonical_frame(
        n_frames=N_FRAMES, frame_rate=FRAME_RATE, home_xy=home, away_xy=away,
        gk_ids={"h_gk", "a_gk"},
    )


class _MovingLoader(BaseLoader):
    name = "moving"
    _df = None

    def list_matches(self) -> list[str]:
        return ["g1"]

    def load_meta(self, match_id: str) -> MatchMeta:
        return MatchMeta(
            match_id=match_id, provider=self.name, frame_rate=FRAME_RATE,
            pitch_length_m=105.0, pitch_width_m=68.0, home_team="H", away_team="A",
            periods={1: (0.0, N_FRAMES / FRAME_RATE)},
        )

    def load(self, match_id: str) -> pd.DataFrame:
        if _MovingLoader._df is None:
            _MovingLoader._df = _moving_match()
        return _MovingLoader._df


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cache")
    mp = pytest.MonkeyPatch()
    mp.setattr(pipeline, "get_loader", lambda name: _MovingLoader())
    cfg = pipeline.PipelineConfig(metrics_yaml_path=METRICS_YAML, cache_dir=d)
    pipeline.run_match("metrica", "g1", cfg)
    pipeline.run_match("skillcorner", "s1", cfg)
    mp.undo()
    return d


@pytest.fixture(scope="module")
def cfg():
    return pipeline.load_config(METRICS_YAML)


def test_synthetic_checks_all_pass_on_shipped_code(cfg):
    checks = validate.synthetic_checks(cfg)
    assert len(checks) >= 15
    failed = [c for c in checks if not c["pass"]]
    assert not failed, failed
    for c in checks:
        assert set(c) == {"name", "metric", "expected", "actual", "tol", "pass"}


def test_cached_matches_lists_both_providers(cache_dir):
    assert validate.cached_matches(cache_dir) == [("metrica", "g1"), ("skillcorner", "s1")]


def test_full_match_players_excludes_substitutes():
    df = make_canonical_frame(n_frames=10, home_xy={"gk": (-50, 0), "p": (0, 0)})
    sub = df["track_id"] == "p"
    df = df[~(sub & (df["frame_id"] < 3))]  # "p" comes on late
    assert validate.full_match_players(df) == {"gk"}


def test_physiological_checks_skip_skillcorner_distance(cfg, cache_dir):
    checks, excluded, not_checked = validate.physiological_checks(cfg, cache_dir)
    assert checks and all(c["match"] == "metrica/g1" for c in checks)
    assert {e["match"] for e in excluded} == {"skillcorner/s1"}
    assert any(e["metric"] == "total_distance_km" for e in excluded)
    assert all("not reported" in e["reason"] for e in excluded)
    # config has no compactness band -> reported, not checked
    assert {n["metric"] for n in not_checked} == {"team_compactness_median_m2"}


def test_physiological_pass_flag_matches_band(cfg, cache_dir):
    checks, _, _ = validate.physiological_checks(cfg, cache_dir)
    metrics = {c["metric"] for c in checks}
    assert {"total_distance_km", "top_speed_mps", "hsr_share_of_distance_pct",
            "gk_top_speed_mps"} <= metrics
    for c in checks:
        in_band = (c["low"] is None or c["value"] >= c["low"]) and (
            c["high"] is None or c["value"] <= c["high"])
        assert c["pass"] == in_band
    # 30 s of running is nowhere near a 9 km match: must FAIL, not be massaged.
    dist = [c for c in checks if c["metric"] == "total_distance_km"]
    assert len(dist) == 20 and not any(c["pass"] for c in dist)


def test_write_validation_report(cfg, cache_dir, tmp_path):
    report = validate.write_validation_report(cfg, cache_dir, tmp_path, figures=True)
    on_disk = json.loads((tmp_path / "validation.json").read_text())
    assert on_disk["summary"] == report["summary"]
    s = on_disk["summary"]
    assert s["n_checks"] == len(on_disk["synthetic"]) + len(on_disk["physiological"])
    assert s["synthetic_all_pass"] is True
    assert s["all_pass"] is False and s["failures"]
    assert on_disk["references"]
    assert {c["provider"] for c in on_disk["coverage"]} == {"metrica", "skillcorner"}
    for name in ("coverage_by_provider", "shape_timeseries", "pitch_snapshot"):
        assert (tmp_path / "figures" / f"{name}.png").exists()


def test_write_sensitivity_report(cfg, cache_dir, tmp_path):
    report = validate.write_sensitivity_report(cfg, cache_dir, tmp_path, figures=False)
    grid_cfg = cfg["physical"]["sensitivity"]
    n_cells = len(grid_cfg["savgol_window_frames_grid"]) * len(
        grid_cfg["sprint_threshold_mps_grid"])
    assert len(report["sprint_count"]) == n_cells
    assert len(report["distance_vs_window"]) == len(grid_cfg["savgol_window_frames_grid"])
    assert report["match"] == "metrica/g1"
    assert report["frame_rate_hz"] == 5.0

    grid = pd.DataFrame(report["sprint_count"]).rename(columns={
        "window": "savgol_window_frames", "threshold": "sprint_threshold_mps",
        "total": "total_sprints"})
    head = report["headline"]
    base = (cfg["kinematics"]["savgol_window_frames"], cfg["physical"]["sprint_threshold_mps"])
    assert head["spread_pct"] == pytest.approx(physical.sprint_spread_pct(grid), abs=1e-3)
    assert head["spread_pct_vs_config_default"] == pytest.approx(
        physical.sprint_spread_pct(grid, baseline=base), abs=1e-3)
    assert head["spread_pct"] > 0  # thresholds straddle the runners' speeds
    assert f"{head['spread_pct']:.0f}%" in head["sentence"]
    assert json.loads((tmp_path / "sensitivity.json").read_text())["headline"] == head


def test_sensitivity_requires_metrica(cfg, tmp_path):
    with pytest.raises(FileNotFoundError):
        validate.write_sensitivity_report(cfg, tmp_path, tmp_path, figures=False)


def test_sensitivity_rejects_uncached_match_id(cfg, cache_dir, tmp_path):
    with pytest.raises(FileNotFoundError, match="metrica/nope"):
        validate.write_sensitivity_report(cfg, cache_dir, tmp_path, match_id="nope",
                                          figures=False)


def test_sensitivity_survives_no_full_match_players(cfg, cache_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "full_match_players", lambda df: set())
    report = validate.write_sensitivity_report(cfg, cache_dir, tmp_path, figures=False)
    assert report["headline"]["distance_spread_pct"] is None
    assert all(r["n_players"] == 0 for r in report["distance_vs_window"])


def test_main_exit_code_reflects_failures(cache_dir, tmp_path):
    code = validate.main(["--config", str(METRICS_YAML), "--cache-dir", str(cache_dir),
                          "--reports-dir", str(tmp_path), "--no-figures"])
    assert code == 1  # the toy match fails the 9-12 km band
    assert validate.main(["--cache-dir", str(tmp_path / "empty"),
                          "--reports-dir", str(tmp_path)]) == 1
