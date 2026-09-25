"""Tests for ``scripts/build_cache.py`` against the fake in-memory loader
from ``test_pipeline`` (offline, no kloppy)."""

from __future__ import annotations

import pytest
from scripts import build_cache

from ftm import pipeline
from tests.conftest import METRICS_YAML
from tests.test_pipeline import _FakeLoader


class _BrokenLoader(_FakeLoader):
    def load(self, match_id: str):
        raise RuntimeError("boom")


@pytest.fixture
def fake_loaders(monkeypatch):
    def _install(factory):
        monkeypatch.setattr(pipeline, "get_loader", factory)
        monkeypatch.setattr(build_cache, "get_loader", factory)

    _install(lambda name: _FakeLoader())
    return _install


def _args(tmp_path, *extra: str) -> list[str]:
    return ["--config", str(METRICS_YAML), "--cache-dir", str(tmp_path), "--no-space", *extra]


def test_single_provider_build_prints_summary(tmp_path, fake_loaders, capsys):
    assert build_cache.main(_args(tmp_path, "--provider", "metrica")) == build_cache.EXIT_OK
    out = capsys.readouterr().out
    assert "metrica" in out and "m1" in out
    assert "total cache size" in out
    assert (tmp_path / "metrica__m1__tracking.parquet").exists()
    assert (tmp_path / "metrica__m1__meta.json").exists()


def test_summary_reads_back_rows_players_and_size(tmp_path, fake_loaders):
    cfg = pipeline.PipelineConfig(metrics_yaml_path=METRICS_YAML, cache_dir=tmp_path,
                                  include_space=False)
    s = build_cache.summarize(pipeline.run_match("metrica", "m1", cfg))
    assert (s.provider, s.match_id) == ("metrica", "m1")
    assert s.n_players == 22  # flat back four: 11 v 11
    assert s.rows == 23 * 26  # 130 frames at 25 Hz -> 26 at 5 Hz, 22 players + ball
    assert s.size_mb > 0
    assert 0.0 <= s.clip_fraction <= 1.0
    assert "metrica" in build_cache.format_table([s])


def test_single_match_build(tmp_path, fake_loaders):
    code = build_cache.main(_args(tmp_path, "--provider", "skillcorner", "--match-id", "x9"))
    assert code == build_cache.EXIT_OK
    assert (tmp_path / "skillcorner__x9__physical.parquet").exists()


def test_all_providers_uses_run_all(tmp_path, fake_loaders, monkeypatch):
    calls = []
    real = pipeline.run_all

    def spy(cfg):
        calls.append(cfg)
        return real(cfg)

    monkeypatch.setattr(pipeline, "run_all", spy)
    assert build_cache.main(_args(tmp_path)) == build_cache.EXIT_OK
    assert len(calls) == 1
    assert calls[0].include_space is False


def test_failed_match_gives_nonzero_exit(tmp_path, fake_loaders, capsys):
    fake_loaders(lambda name: _BrokenLoader())
    assert build_cache.main(_args(tmp_path, "--provider", "metrica")) == build_cache.EXIT_FAILED
    assert "metrica/m1" in capsys.readouterr().err


def test_run_all_failure_still_summarizes_good_matches(tmp_path, fake_loaders, capsys):
    fake_loaders(lambda name: _BrokenLoader() if name == "metrica" else _FakeLoader())
    assert build_cache.main(_args(tmp_path)) == build_cache.EXIT_FAILED
    captured = capsys.readouterr()
    assert "skillcorner" in captured.out
    assert "metrica/m1" in captured.err


def test_run_all_failure_ignores_stale_and_failed_metas(tmp_path, fake_loaders, capsys):
    # A previous successful build of metrica/m1 and an unrelated stale match.
    assert build_cache.main(_args(tmp_path, "--provider", "metrica")) == build_cache.EXIT_OK
    stale = tmp_path / "metrica__old__meta.json"
    stale.write_text("{}")
    capsys.readouterr()

    fake_loaders(lambda name: _BrokenLoader() if name == "metrica" else _FakeLoader())
    assert build_cache.main(_args(tmp_path)) == build_cache.EXIT_FAILED
    table = capsys.readouterr().out.split("total cache size")[0]
    rows = [ln.split()[:2] for ln in table.splitlines() if ln.startswith(("metrica", "skill"))]
    assert rows == [["skillcorner", "m1"]]


def test_over_budget_warns_and_exits_nonzero(tmp_path, fake_loaders, capsys):
    code = build_cache.main(_args(tmp_path, "--provider", "metrica", "--max-cache-mb", "1e-9"))
    assert code == build_cache.EXIT_OVER_BUDGET
    assert "--target-hz" in capsys.readouterr().err


def test_match_id_requires_provider(tmp_path):
    with pytest.raises(SystemExit):
        build_cache.main(_args(tmp_path, "--match-id", "1"))


def test_target_hz_defaults_to_config(metrics_cfg):
    assert build_cache.resolve_target_hz(None, metrics_cfg) == metrics_cfg["cache"]["target_hz"]
    assert build_cache.resolve_target_hz(2.5, metrics_cfg) == 2.5
    with pytest.raises(KeyError, match="target-hz"):
        build_cache.resolve_target_hz(None, {})
