"""Loader tests.

The real ``metrica.load`` / ``skillcorner.load`` need downloads, so they
live behind a marker (``@pytest.mark.integration``) and are skipped in
the default CI run. What we can test cheaply:

* the factory
* the abstraction boundary (a fake loader's output flows through every
  metric unchanged; metrics never import provider code)
* the loader helpers: download cache, GK heuristic, ball_state rule and
  the raw-file sanitizers, on small synthetic inputs
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ftm.metrics
from ftm.loaders import BaseLoader, MetricaLoader, SkillCornerLoader, get_loader
from ftm.loaders.metrica import derive_ball_state, sanitize_epts_lines, sanitize_metrica_csv

FORBIDDEN_IMPORTS = ("kloppy", "ftm.loaders", "ftm.kinematics")


def test_get_loader_returns_right_types():
    """``get_loader("metrica")`` -> ``MetricaLoader``; ``"SkillCorner"``
    (any case) -> ``SkillCornerLoader``; unknown name -> ``KeyError``
    listing the valid names."""
    assert type(get_loader("metrica")) is MetricaLoader
    assert type(get_loader("METRICA")) is MetricaLoader
    assert type(get_loader("SkillCorner")) is SkillCornerLoader
    assert type(get_loader("skillcorner")) is SkillCornerLoader
    assert isinstance(get_loader("metrica"), BaseLoader)

    with pytest.raises(KeyError) as excinfo:
        get_loader("statsbomb")
    message = str(excinfo.value)
    assert "metrica" in message and "skillcorner" in message


def test_fake_loader_output_passes_schema_and_every_metric():
    """Build a synthetic canonical frame (via ``conftest``), run it
    through ``kinematics`` + ``shape`` + ``pressing`` + ``space`` +
    ``physical`` with no code under ``ftm/metrics`` knowing which
    'provider' it came from. This is the executable form of the
    source-agnostic rule."""
    raise NotImplementedError


def _imported_modules(tree: ast.Module, module_name: str) -> set[str]:
    package = module_name.rsplit(".", 1)[0]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)]
                prefix = ".".join(base + ([node.module] if node.module else []))
            else:
                prefix = node.module or ""
            names.add(prefix)
            # ``from ftm import loaders`` imports ftm.loaders too.
            names.update(f"{prefix}.{alias.name}" for alias in node.names)
    return names


def test_metrics_package_never_imports_kloppy_or_loaders():
    """Static check: parse every module under ``ftm.metrics`` and assert
    none imports ``kloppy``, ``ftm.loaders`` or ``ftm.kinematics``
    (absolute or relative). Uses ``ast`` so docstrings that merely mention
    kloppy don't trip it. Fails loudly if the abstraction leaks."""
    metrics_dir = Path(ftm.metrics.__file__).parent
    files = sorted(metrics_dir.rglob("*.py"))
    assert files, "found no modules under ftm/metrics"

    leaks: list[str] = []
    for path in files:
        rel = path.relative_to(metrics_dir).with_suffix("")
        parts = ["ftm", "metrics", *rel.parts]
        if parts[-1] == "__init__":
            parts = [*parts[:-1], "__init__"]
        module_name = ".".join(parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name in _imported_modules(tree, module_name):
            if any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN_IMPORTS):
                leaks.append(f"{path.name}: {name}")
    assert not leaks, f"ftm.metrics imports provider/kinematics code: {leaks}"


def test_import_checker_catches_relative_and_absolute_leaks():
    src = "import kloppy\nfrom .. import loaders\nfrom ..kinematics import speed\n"
    names = _imported_modules(ast.parse(src), "ftm.metrics.shape")
    assert {"kloppy", "ftm.loaders", "ftm.kinematics"} <= names
    doc_only = '"""Uses kloppy upstream."""\nimport pandas as pd\n'
    assert _imported_modules(ast.parse(doc_only), "ftm.metrics.shape") == {"pandas"}


# ----- BaseLoader helpers ----------------------------------------------------


class _DummyLoader(BaseLoader):
    name = "dummy"

    def list_matches(self):
        return []

    def load_meta(self, match_id):
        raise NotImplementedError

    def load(self, match_id):
        raise NotImplementedError


def test_download_cached_fetches_once_into_provider_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FTM_DATA_DIR", str(tmp_path / "data"))
    src = tmp_path / "remote" / "match" / "tracking.txt"
    src.parent.mkdir(parents=True)
    src.write_text("v1")

    loader = _DummyLoader()
    path = loader._download_cached(src.as_uri())
    assert path.read_text() == "v1"
    assert (tmp_path / "data" / "raw" / "dummy") in path.parents
    assert path.name == "tracking.txt"

    src.write_text("v2 (changed upstream)")
    assert loader._download_cached(src.as_uri()) == path
    assert path.read_text() == "v1", "second call must hit the cache, not re-download"
    assert not list(path.parent.glob("*.part")), "temp files must not be left behind"


def test_download_cached_failure_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.setenv("FTM_DATA_DIR", str(tmp_path / "data"))
    missing = (tmp_path / "nope" / "missing.txt").as_uri()
    with pytest.raises(OSError):
        _DummyLoader()._download_cached(missing)
    raw = tmp_path / "data" / "raw" / "dummy"
    assert not [p for p in raw.rglob("*") if p.is_file()]


def _long_frame(positions: dict[tuple[str, str], list[float]], periods=(1,)) -> pd.DataFrame:
    rows = []
    for period in periods:
        for (team, track), xs in positions.items():
            for frame, x in enumerate(xs):
                rows.append((period, frame, track, team, x))
    return pd.DataFrame(rows, columns=["period", "frame_id", "track_id", "team", "x_pitch"])


def test_infer_is_gk_picks_deepest_regular_player_per_team():
    df = _long_frame(
        {
            ("home", "h_gk"): [-50.0, -48.0, -49.0, -50.0],
            ("home", "h_def"): [-30.0, -35.0, -30.0, -25.0],
            ("home", "h_fwd"): [5.0, 10.0, 0.0, 5.0],
            ("away", "a_gk"): [51.0, 49.0, 50.0, 50.0],
            ("away", "a_def"): [30.0, 20.0, 25.0, 30.0],
            ("ball", "ball"): [-52.0, 0.0, 52.0, 0.0],
        }
    )
    is_gk = _DummyLoader()._infer_is_gk(df)
    assert is_gk.index.equals(df.index)
    assert set(df.loc[is_gk, "track_id"]) == {"h_gk", "a_gk"}


def test_infer_is_gk_ignores_cameo_tracks_near_goal_line():
    positions = {
        ("home", "gk"): [-45.0] * 40,
        ("home", "def"): [-20.0] * 40,
        # present in 2 of 40 frames, right on the goal line: not the keeper
        ("home", "cameo"): [-52.0, -52.0],
    }
    df = _long_frame(positions)
    is_gk = _DummyLoader()._infer_is_gk(df)
    assert set(df.loc[is_gk, "track_id"]) == {"gk"}


# ----- Metrica helpers --------------------------------------------------------


def test_derive_ball_state_alive_only_when_tracked_and_on_pitch():
    x = pd.Series([0.0, 52.5, 52.6, np.nan, -10.0, 10.0])
    y = pd.Series([0.0, 34.0, 0.0, 0.0, np.nan, -34.1])
    state = derive_ball_state(x, y, x_bounds=(-52.5, 52.5), y_bounds=(-34.0, 34.0))
    assert state.tolist() == ["alive", "alive", "dead", "dead", "dead", "dead"]
    assert state.index.equals(x.index)


def _epts_specs():
    # Two format blocks: frames 1-3 have 2 players, frames 4-6 have 1.
    two = re.compile(r"^(?P<f>\d+):[^,;:]*,[^,;:]*;[^,;:]*,[^,;:]*:[^,;:]*,[^,;:]*$")
    one = re.compile(r"^(?P<f>\d+):[^,;:]*,[^,;:]*:[^,;:]*,[^,;:]*$")
    return [(1, 3, two), (4, 6, one)]


def test_sanitize_epts_lines_drops_malformed_and_counts():
    lines = [
        b"1:0.1,0.2;0.3,0.4:0.5,0.5\n",
        b"2:0.1,0.2;0.3:0.5,0.5\n",  # missing a coordinate
        b"3:0.1,0.2;0.3,0.4:NaN,NaN\n",
        b"4:0.1,0.2:0.5,0.5\n",
        b"garbage\n",
        b"\n",
        b"5:0.1,0.",  # truncated last line of an interrupted download
    ]
    kept, dropped = sanitize_epts_lines(lines, _epts_specs())
    assert dropped == 3
    assert [k.split(b":")[0] for k in kept] == [b"1", b"3", b"4"]
    assert all(k.endswith(b"\n") for k in kept)


def test_sanitize_epts_lines_refuses_to_drop_block_boundary():
    lines = [b"1:0.1,0.2;0.3,0.4:0.5,0.5\n", b"3:broken\n", b"4:0.1,0.2:0.5,0.5\n"]
    with pytest.raises(ValueError, match="frame 3"):
        sanitize_epts_lines(lines, _epts_specs())


def _csv(rows: list[str]) -> bytes:
    header = [
        ",,,Home,,Home,,,",
        ",,,1,,2,,,",
        "Period,Frame,Time [s],P1,,P2,,Ball,",
    ]
    return ("\n".join(header + rows) + "\n").encode()


def test_sanitize_metrica_csv_drops_bad_frames_from_both_files():
    home = _csv(
        [
            "1,1,0.04,0.1,0.2,0.3,0.4,0.5,0.5",
            "1,2,0.08,0.1,0.2,0.3,0.4,0.5,0.5",
            "1,3,0.12,0.1,0.2,0.3,0.4,NaN,NaN",
            "1,4,0.16,0.1,0.2,0.3,0.4,0.6,0.6",
        ]
    )
    away = _csv(
        [
            "1,1,0.04,0.7,0.2,0.8,0.4,0.5,0.5",
            "1,2,0.08,0.7,0.2",  # short row (truncated)
            "1,3,0.12,0.7,0.2,0.8,0.4,NaN,NaN",
            "1,4,0.16,0.7,0.2,0.8,0.4,0.9,0.9",  # ball disagrees with home
        ]
    )
    new_home, new_away, dropped = sanitize_metrica_csv(home, away)
    assert dropped == 2

    def frames(data: bytes) -> list[int]:
        return [int(line.split(b",")[1]) for line in data.splitlines()[3:]]

    assert frames(new_home) == frames(new_away) == [1, 3]
    assert new_home.splitlines()[:3] == home.splitlines()[:3]


def test_sanitize_metrica_csv_is_identity_on_clean_input():
    rows = ["1,1,0.04,0.1,0.2,0.3,0.4,0.5,0.5", "2,2,0.08,0.1,0.2,0.3,0.4,NaN,NaN"]
    home = away = _csv(rows)
    new_home, new_away, dropped = sanitize_metrica_csv(home, away)
    assert dropped == 0
    assert new_home == home and new_away == away


def test_metrica_unknown_match_raises_keyerror():
    with pytest.raises(KeyError, match="known"):
        MetricaLoader().load("99")


# ----- integration (network) -----------------------------------------------


@pytest.mark.integration
def test_metrica_game3_loads_and_validates():
    """(integration, network) Full, unlimited ``MetricaLoader().load("3")``
    passes ``validate``, has 22 players + 1 ball in a typical frame,
    ``x_pitch`` spanning ~+/-52.5, ``frame_rate`` read from metadata == 25,
    and a plausible share of ball-alive frames."""
    from ftm.schema import COORD_TOLERANCE_M, validate

    loader = MetricaLoader()
    assert "3" in loader.list_matches()
    df = validate(loader.load("3"))

    meta = loader.load_meta("3")
    assert meta.frame_rate == 25
    assert set(df["period"].unique()) == {1, 2}

    players_per_frame = df[~df["is_ball"]].groupby(["period", "frame_id"]).size()
    assert players_per_frame.median() == 22
    ball_per_frame = df[df["is_ball"]].groupby(["period", "frame_id"]).size()
    assert ball_per_frame.max() == 1
    # Substitutes add tracks over a match, but never more than 22 at once.
    assert df.loc[~df["is_ball"], "track_id"].nunique() >= 22

    assert df["x_pitch"].abs().max() <= 52.5 + COORD_TOLERANCE_M
    assert df["x_pitch"].min() < -45 and df["x_pitch"].max() > 45

    gk_per_frame = df[df["is_gk"]].groupby(["period", "frame_id"]).size()
    assert gk_per_frame.median() == 2

    frames = df.drop_duplicates(["period", "frame_id"])
    alive_share = (frames["ball_state"] == "alive").mean()
    assert 0.4 < alive_share < 0.9

