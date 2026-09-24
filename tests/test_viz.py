from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from shapely.geometry import box  # noqa: E402

from ftm import viz  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _players_frame() -> pd.DataFrame:
    rows = [
        ("h_gk", "home", -48.0, 0.0, True, False, 1),
        ("h_1", "home", -30.0, -20.0, False, False, 2),
        ("h_2", "home", -30.0, 20.0, False, False, None),
        ("a_gk", "away", 48.0, 0.0, True, False, 1),
        ("a_1", "away", 10.0, 5.0, False, False, 9),
        ("ball", "ball", 0.0, 0.0, False, True, None),
    ]
    df = pd.DataFrame(
        rows, columns=["track_id", "team", "x_pitch", "y_pitch", "is_gk", "is_ball",
                       "jersey_number"]
    )
    df["jersey_number"] = df["jersey_number"].astype("Int16")
    df["team"] = pd.Categorical(df["team"], categories=["home", "away", "ball"])
    return df


def _voronoi_frame() -> pd.DataFrame:
    cells = [
        ("h_gk", "home", box(-52.5, -34, -40, 34)),
        ("h_1", "home", box(-40, -34, -20, 0)),
        ("h_2", "home", box(-40, 0, -20, 34)),
        ("a_1", "away", box(-20, -34, 40, 34)),
        ("a_gk", "away", box(40, -34, 52.5, 34)),
    ]
    return pd.DataFrame(
        {
            "track_id": [c[0] for c in cells],
            "team": [c[1] for c in cells],
            "area_m2": [c[2].area for c in cells],
            "cell_wkt": [c[2].wkt for c in cells],
        }
    )


def _shape_frame(periods=(1, 2), n=20) -> pd.DataFrame:
    rows = []
    for period in periods:
        for i in range(n):
            for team, sign in (("home", -1), ("away", 1)):
                rows.append(
                    {
                        "period": period, "frame_id": i, "timestamp": i * 0.5, "team": team,
                        "line_height_m": sign * (25 + np.sin(i / 3)), "width_m": 40 + i % 3,
                        "length_m": 35.0, "hull_area_m2": 1200.0 + i, "cx_m": sign * 10.0,
                        "cy_m": 0.0, "n_outfield": 10, "low_outfield_count": False,
                    }
                )
    return pd.DataFrame(rows)


def _grid() -> pd.DataFrame:
    rows = [
        {"savgol_window_frames": w, "sprint_threshold_mps": t,
         "total_sprints": 100 - 5 * t + w, "sprints_per_player_mean": (100 - 5 * t + w) / 22}
        for w in (5, 7, 9, 11, 13) for t in (6.0, 6.5, 7.0, 7.5)
    ]
    return pd.DataFrame(rows)


def _coverage() -> pd.DataFrame:
    rows = []
    for provider, base in (("metrica", 99.0), ("skillcorner", 40.0)):
        for k in range(6):
            rows.append({"provider": provider, "track_id": f"p{k}",
                         "team": "home" if k < 3 else "away", "coverage_pct": base - 5 * k})
    return pd.DataFrame(rows)


def test_pitch_snapshot_returns_figure_with_overlay():
    fig = viz.pitch_snapshot(_players_frame(), voronoi_frame=_voronoi_frame(), title="t")
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].patches) >= 5
    assert fig.axes[0].get_title() == "t"


def test_pitch_snapshot_without_overlay():
    fig = viz.pitch_snapshot(_players_frame().drop(columns="jersey_number"))
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_pitch_snapshot_duplicate_index_and_missing_wkt():
    df = pd.concat([_players_frame().iloc[:3], _players_frame().iloc[3:]])
    df.index = [0, 1, 0, 1, 0, 1]
    assert isinstance(viz.pitch_snapshot(df), Figure)
    with pytest.raises(ValueError, match="cell_wkt"):
        viz.pitch_snapshot(df, voronoi_frame=_voronoi_frame().drop(columns="cell_wkt"))


def test_time_series_one_axis_per_column_and_no_cross_period_line():
    fig = viz.time_series(_shape_frame(), columns=["line_height_m", "width_m"])
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    # 2 teams x 2 periods = 4 separate lines per subplot (+1 period rule)
    data_lines = [ln for ln in fig.axes[0].get_lines() if len(ln.get_xdata()) > 2]
    assert len(data_lines) == 4
    home_p2 = max(data_lines, key=lambda ln: np.min(ln.get_xdata()))
    assert np.min(home_p2.get_xdata()) >= 9.5  # offset by period-1 duration


def test_time_series_window_slices():
    fig = viz.time_series(_shape_frame(periods=(1,)), columns=["hull_area_m2"], window=(2, 5))
    assert len(fig.axes) == 1
    for ln in fig.axes[0].get_lines():
        x = np.asarray(ln.get_xdata(), dtype=float)
        assert x.min() >= 2 and x.max() <= 5


def test_time_series_rejects_unknown_column():
    with pytest.raises(KeyError):
        viz.time_series(_shape_frame(), columns=["nope"])


def test_sensitivity_heatmap_annotates_cells_and_spread():
    grid = _grid()
    fig = viz.sensitivity_heatmap(grid)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2  # heatmap + colorbar
    ax = fig.axes[0]
    assert len(ax.texts) == 20
    vals = grid["total_sprints"]
    expected = (vals.max() - vals.min()) / vals.median() * 100
    assert f"{expected:.0f}% spread" in ax.get_title()


def test_coverage_bar_one_axis_per_provider():
    fig = viz.coverage_bar(_coverage())
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    heights = [p.get_height() for p in fig.axes[1].patches]
    assert heights == sorted(heights, reverse=True)


def test_save_figure_writes_png(tmp_path):
    fig = viz.pitch_snapshot(_players_frame())
    out = viz.save_figure(fig, tmp_path / "nested" / "snap.png")
    assert out == tmp_path / "nested" / "snap.png"
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not plt.fignum_exists(fig.number)
