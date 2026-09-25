"""Dashboard tests: pure helpers directly, plus AppTest smoke runs on a
hand-built fake cache (no real pipeline, no network)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _load_app_module():
    spec = importlib.util.spec_from_file_location("ftm_streamlit_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


app = _load_app_module()

PLAYERS = [("h1", "home", True), ("h2", "home", False), ("a1", "away", True), ("a2", "away", False)]
BASE_XY = {"h1": (-45.0, 0.0), "h2": (-10.0, 5.0), "a1": (45.0, 0.0), "a2": (10.0, -5.0)}


def _tracking(periods: dict[int, int], hz: float = 5.0, drop: set[tuple[str, int]] = ()):
    rows = []
    for period, n_frames in periods.items():
        for f in range(n_frames):
            for jersey, (tid, team, gk) in enumerate(PLAYERS, start=1):
                if (tid, f) in drop:
                    continue
                x, y = BASE_XY[tid]
                rows.append((f, period, f / hz, tid, team, jersey, x + 0.1 * f, y, False, gk))
            rows.append((f, period, f / hz, "ball", "ball", None, 0.2 * f, 0.0, False, False))
    df = pd.DataFrame(
        rows,
        columns=["frame_id", "period", "timestamp", "track_id", "team", "jersey_number",
                 "x_pitch", "y_pitch", "is_ball", "is_gk"],
    )
    df["is_ball"] = df["track_id"] == "ball"
    df["period"] = df["period"].astype("int8")
    df["track_id"] = df["track_id"].astype("string")
    df["team"] = pd.Categorical(df["team"], categories=["home", "away", "ball"])
    df["jersey_number"] = df["jersey_number"].astype("Int16")
    df["ball_state"] = pd.Categorical(["alive"] * len(df), categories=["alive", "dead"])
    df["speed"] = 1.0
    return df


def _shape(tracking: pd.DataFrame) -> pd.DataFrame:
    keys = tracking.loc[~tracking["is_ball"], ["period", "frame_id", "timestamp", "team"]]
    out = keys.drop_duplicates().reset_index(drop=True)
    out["team"] = out["team"].astype(str)
    n = len(out)
    out["line_height_m"] = [20.0 + i % 5 for i in range(n)]
    out["width_m"] = 30.0
    out["length_m"] = 35.0
    out["hull_area_m2"] = 900.0
    return out


def _space_player(tracking: pd.DataFrame) -> pd.DataFrame:
    players = tracking.loc[~tracking["is_ball"]].copy()
    cells = []
    for row in players.itertuples(index=False):
        x0 = -52.5 if row.team == "home" else 0.0
        cells.append(f"POLYGON (({x0} -34, {x0 + 52.5} -34, {x0 + 52.5} 34, {x0} 34, {x0} -34))")
    out = players[["period", "frame_id", "timestamp", "track_id"]].copy()
    out["team"] = players["team"].astype(str)
    out["area_m2"] = 1785.0
    out["cell_wkt"] = cells
    return out


def _write_match(cache: Path, provider: str, match_id: str, *, broadcast: bool) -> None:
    periods = {1: 20} if broadcast else {1: 20, 2: 15}
    drop = {("a2", f) for f in range(0, 20, 2)} if broadcast else set()
    tracking = _tracking(periods, drop=drop)
    prefix = cache / f"{provider}__{match_id}__"
    tracking.to_parquet(f"{prefix}tracking.parquet", index=False)
    _shape(tracking).to_parquet(f"{prefix}shape.parquet", index=False)
    _space_player(tracking).to_parquet(f"{prefix}space_player.parquet", index=False)

    tids = [p[0] for p in PLAYERS]
    teams = [p[1] for p in PLAYERS]
    cov = pd.DataFrame({"track_id": tids, "team": teams,
                        "coverage_pct": [100.0, 100.0, 100.0, 50.0 if broadcast else 100.0]})
    cov.assign(provider=provider, match_id=match_id).to_parquet(
        f"{prefix}coverage.parquet", index=False
    )
    if broadcast:
        phys = pd.DataFrame(columns=["track_id", "team", "distance_m", "hsr_distance_m",
                                     "n_sprints"])
    else:
        phys = pd.DataFrame({"track_id": tids, "team": teams,
                             "distance_m": [4000.0, 10500.0, 4100.0, 11000.0],
                             "hsr_distance_m": [0.0, 800.0, 0.0, 900.0],
                             "n_sprints": [0, 12, 0, 15]})
    phys.assign(provider=provider, match_id=match_id).to_parquet(
        f"{prefix}physical.parquet", index=False
    )

    meta = {
        "match_meta": {
            "match_id": match_id, "provider": provider, "frame_rate": 5.0,
            "home_team": f"{provider} Home", "away_team": f"{provider} Away",
            # Match-clock bounds like the real loaders: period 2 does not start at 0.
            "periods": {str(p): [100.0 * (p - 1), 100.0 * (p - 1) + (n - 1) / 5.0]
                        for p, n in periods.items()},
        },
        "cached_frame_rate_hz": 5.0,
        "physical_note": "not reported for broadcast" if broadcast else "",
    }
    (cache / f"{provider}__{match_id}__meta.json").write_text(json.dumps(meta))


@pytest.fixture
def fake_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_match(cache, "metrica", "1", broadcast=False)
    _write_match(cache, "skillcorner", "4039", broadcast=True)
    return cache


# ---------------------------------------------------------------------------
# Plain functions
# ---------------------------------------------------------------------------


def test_list_cached_matches_empty_and_missing(tmp_path: Path):
    assert app.list_cached_matches(tmp_path) == []
    assert app.list_cached_matches(tmp_path / "nope") == []


def test_list_cached_matches_parses_meta(fake_cache: Path):
    (fake_cache / "garbage__meta.json").write_text("{}")
    matches = app.list_cached_matches(fake_cache)
    assert [(m["provider"], m["match_id"]) for m in matches] == [
        ("metrica", "1"), ("skillcorner", "4039"),
    ]
    metrica = matches[0]
    assert metrica["label"] == "Metrica · 1 · metrica Home vs metrica Away"
    assert metrica["periods"] == {1: (0.0, 3.8), 2: (100.0, 102.8)}
    assert metrica["target_hz"] == 5.0
    assert matches[1]["physical_note"]


def test_resolve_cache_dir_env_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(app.CACHE_DIR_ENV, str(tmp_path))
    assert app.resolve_cache_dir() == tmp_path
    monkeypatch.delenv(app.CACHE_DIR_ENV)
    assert app.resolve_cache_dir() == app.CACHE_DIR


def test_load_tidy(fake_cache: Path):
    shape = app.load_tidy(fake_cache, "metrica", "1", "shape")
    assert {"line_height_m", "team", "period"} <= set(shape.columns)
    assert app.load_tidy(fake_cache, "metrica", "1", "pressing").empty
    with pytest.raises(ValueError, match="unknown tidy artifact"):
        app.load_tidy(fake_cache, "metrica", "1", "../meta")


def test_nearest_frame_picks_closest_timestamp(fake_cache: Path):
    tracking = app.load_tidy(fake_cache, "metrica", "1", "tracking")
    frame = app.nearest_frame(tracking, 2, 1.13)  # frames every 0.2 s -> frame 6 (1.2 s)
    assert frame["frame_id"].unique().tolist() == [6]
    assert (frame["period"] == 2).all()
    assert len(frame) == len(PLAYERS) + 1
    assert app.nearest_frame(tracking, 3, 0.0).empty


def test_frame_cells(fake_cache: Path):
    sp = app.load_tidy(fake_cache, "metrica", "1", "space_player")
    cells = app.frame_cells(sp, 1, 4)
    assert len(cells) == len(PLAYERS)
    assert app.frame_cells(pd.DataFrame(), 1, 4).empty


def test_period_bounds_are_period_relative(fake_cache: Path):
    match = app.list_cached_matches(fake_cache)[0]
    ranges = app.tracking_period_ranges(fake_cache, "metrica", "1")
    assert ranges == {1: (0.0, 3.8), 2: (0.0, 2.8)}
    assert app.period_bounds(match, 2, ranges) == (0.0, 2.8)
    # meta fallback is shifted to start at 0, never match-clock seconds
    assert app.period_bounds(match, 2) == pytest.approx((0.0, 2.8))
    assert app.period_bounds({**match, "periods": {}}, 2) == (0.0, 1.0)
    assert app.tracking_period_ranges(fake_cache, "metrica", "missing") == {}


def test_player_table_metrica_shows_coverage_next_to_each_value(fake_cache: Path):
    phys = app.load_tidy(fake_cache, "metrica", "1", "physical")
    cov = app.load_tidy(fake_cache, "metrica", "1", "coverage")
    table = app.build_player_table(phys, cov, broadcast=False)
    assert len(table) == len(PLAYERS)
    row = table.set_index("Player").loc["h2"]
    assert row["Distance (m)"] == "10,500 (100% cov)"
    assert row["HSR distance (m)"] == "800 (100% cov)"
    assert row["Sprints"] == "12 (100% cov)"
    assert row["Coverage %"] == 100.0


def test_player_table_skillcorner_reads_na_not_zero(fake_cache: Path):
    match = app.list_cached_matches(fake_cache)[1]
    phys = app.load_tidy(fake_cache, "skillcorner", "4039", "physical")
    cov = app.load_tidy(fake_cache, "skillcorner", "4039", "coverage")
    assert app.is_broadcast(match)
    assert not app.is_broadcast(app.list_cached_matches(fake_cache)[0])
    table = app.build_player_table(phys, cov, broadcast=True)
    assert len(table) == len(PLAYERS)
    for label in app.PHYSICAL_COLUMNS.values():
        assert (table[label] == app.NA_BROADCAST).all()
    assert table.set_index("Player").loc["a2", "Coverage %"] == 50.0


def test_player_table_empty_physical_non_broadcast_is_dash(fake_cache: Path):
    cov = app.load_tidy(fake_cache, "metrica", "1", "coverage")
    table = app.build_player_table(pd.DataFrame(), cov, broadcast=False)
    assert (table["Distance (m)"] == "—").all()


# ---------------------------------------------------------------------------
# AppTest smoke runs
# ---------------------------------------------------------------------------


def _run_app(monkeypatch, cache_dir: Path) -> AppTest:
    monkeypatch.setenv(app.CACHE_DIR_ENV, str(cache_dir))
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    return at


def test_app_without_cache_shows_hint(monkeypatch, tmp_path: Path):
    at = _run_app(monkeypatch, tmp_path / "empty")
    assert not at.exception
    assert any("No cached matches" in i.value for i in at.info)
    assert len(at.sidebar.selectbox) == 0


def test_app_renders_panels_for_each_match(monkeypatch, fake_cache: Path):
    at = _run_app(monkeypatch, fake_cache)
    assert not at.exception, at.exception
    assert len(at.sidebar.selectbox) == 1
    assert len(at.sidebar.slider) == 1
    assert len(at.sidebar.toggle) == 1
    assert len(at.sidebar.multiselect) == 1
    headers = [h.value for h in at.subheader]
    assert headers == ["Pitch snapshot", "Team shape over the window", "Players"]
    table = at.dataframe[0].value
    assert table["Distance (m)"].str.contains("% cov").all()

    at.sidebar.radio[0].set_value(2)
    at.sidebar.toggle[0].set_value(False)
    at.run()
    assert not at.exception, at.exception
    assert at.sidebar.slider[0].value == (0.0, 2.8)
    assert not any("No shape rows" in i.value for i in at.info)
    at.sidebar.slider[0].set_value((1.0, 2.0))
    at.run()
    assert not at.exception, at.exception

    at.sidebar.selectbox[0].set_value(1)
    at.run()
    assert not at.exception, at.exception
    table = at.dataframe[0].value
    assert (table["Distance (m)"] == app.NA_BROADCAST).all()
    assert any("broadcast" in c.value for c in at.caption)


def test_app_empty_series_selection(monkeypatch, fake_cache: Path):
    at = _run_app(monkeypatch, fake_cache)
    at.sidebar.multiselect[0].set_value([])
    at.run()
    assert not at.exception, at.exception
    assert any("at least one shape series" in i.value for i in at.info)
