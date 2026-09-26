"""mplsoccer wrappers — the only place plotting code lives.

Used by both ``scripts/validate.py`` (report figures under
``reports/figures/``) and ``app/streamlit_app.py`` (dashboard panels).
Keeping it in one module means the README still-frame and the live
dashboard look identical.

Every function takes already-computed tidy data (never a raw match) and
returns a Matplotlib ``Figure``. No file writing here except the explicit
``save_figure`` helper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon
from mplsoccer import Pitch
from shapely import wkt as shapely_wkt

from ftm.metrics.physical import sprint_spread_pct
from ftm.schema import PITCH_LENGTH_M, PITCH_WIDTH_M

TEAM_COLORS: dict[str, str] = {"home": "#d62728", "away": "#1f77b4", "ball": "#111111"}
PROVIDER_COLORS: dict[str, str] = {"metrica": "#2ca02c", "skillcorner": "#9467bd"}
_FALLBACK_COLOR = "#7f7f7f"


def _team_color(team) -> str:
    return TEAM_COLORS.get(str(team), _FALLBACK_COLOR)


def _make_pitch() -> Pitch:
    # "secondspectrum" in mplsoccer 1.8 is centred: x in [-L/2, L/2], y in [-W/2, W/2],
    # y pointing up, which matches the canonical frame with no shifting.
    return Pitch(
        pitch_type="secondspectrum",
        pitch_length=PITCH_LENGTH_M,
        pitch_width=PITCH_WIDTH_M,
        pitch_color="#f4f7f2",
        line_color="#8a8a8a",
        line_zorder=1,
    )


def _draw_cells(ax, voronoi_frame: pd.DataFrame) -> None:
    if "cell_wkt" not in voronoi_frame.columns:
        raise ValueError("voronoi_frame needs a 'cell_wkt' column (polygon WKT per player)")
    for row in voronoi_frame.itertuples(index=False):
        cell_wkt = row.cell_wkt
        if cell_wkt is None or pd.isna(cell_wkt):
            continue
        geom = shapely_wkt.loads(cell_wkt)
        if geom.is_empty:
            continue
        polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        color = _team_color(row.team)
        for poly in polys:
            if poly.geom_type != "Polygon" or poly.is_empty:
                continue
            coords = np.asarray(poly.exterior.coords)
            ax.add_patch(
                MplPolygon(
                    coords, closed=True, facecolor=color, edgecolor=color,
                    alpha=0.18, linewidth=0.6, zorder=2,
                )
            )


def _draw_centroid_trace(
    ax, centroid_trace: pd.DataFrame, *, frame_id: int | None, max_gap_s: float,
) -> None:
    missing = {"team", "timestamp", "cx_m", "cy_m"} - set(centroid_trace.columns)
    if missing:
        raise ValueError(f"centroid_trace is missing columns: {sorted(missing)}")
    for team in ("home", "away"):
        rows = centroid_trace[centroid_trace["team"].astype(str) == team]
        rows = rows.sort_values("timestamp")
        if rows.empty:
            continue
        x = rows["cx_m"].to_numpy(dtype=float)
        y = rows["cy_m"].to_numpy(dtype=float)
        # Shape rows exist only while the ball is alive: break the line at
        # stoppages instead of drawing a straight jump across them.
        breaks = np.flatnonzero(np.diff(rows["timestamp"].to_numpy(dtype=float)) > max_gap_s) + 1
        color = TEAM_COLORS[team]
        ax.plot(np.insert(x, breaks, np.nan), np.insert(y, breaks, np.nan), color=color,
                linewidth=1.0, alpha=0.55, zorder=3, label=f"{team} centroid")
        if frame_id is not None and "frame_id" in rows.columns:
            now = rows[rows["frame_id"] == frame_id]
            if not now.empty:
                ax.scatter(now["cx_m"], now["cy_m"], s=140, marker="X", color=color,
                           edgecolors="white", linewidth=1.2, zorder=5)


def pitch_snapshot(
    players_frame: pd.DataFrame,
    *,
    voronoi_frame: pd.DataFrame | None = None,
    centroid_trace: pd.DataFrame | None = None,
    title: str | None = None,
    centroid_max_gap_s: float = 1.0,
) -> Figure:
    """Draw player positions for one frame, optional Voronoi overlay.

    Parameters
    ----------
    players_frame:
        One frame's rows: track_id, team, x_pitch, y_pitch, is_gk, is_ball
        (optional ``jersey_number``, drawn inside the marker when present).
    voronoi_frame:
        Optional per-player rows (track_id, team, area_m2, cell_wkt) for the
        same frame; ``cell_wkt`` is a shapely polygon WKT in the same centred
        coordinates. Each cell is shaded by team colour at low alpha.
    centroid_trace:
        Optional shape rows (team, timestamp, cx_m, cy_m, optional frame_id)
        for a time window: each team's centroid path is drawn as a line,
        broken wherever consecutive rows are more than ``centroid_max_gap_s``
        apart (stoppages), with an X at this frame's centroid.
    title:
        e.g. "Metrica Game 3 - 62:14 - ball alive".

    Uses ``mplsoccer.Pitch(pitch_type="secondspectrum")`` at 105x68, which
    is centred on the centre spot, so canonical coordinates are plotted
    as-is. Goalkeepers are drawn as squares, outfielders as circles, the
    ball as a small white-faced dot. Returns the ``Figure``.
    """
    pitch = _make_pitch()
    fig, ax = pitch.draw(figsize=(10, 6.8))

    if voronoi_frame is not None and len(voronoi_frame):
        _draw_cells(ax, voronoi_frame)

    players_frame = players_frame.reset_index(drop=True)
    if centroid_trace is not None and len(centroid_trace):
        frame_id = (int(players_frame["frame_id"].iloc[0])
                    if "frame_id" in players_frame.columns and len(players_frame) else None)
        _draw_centroid_trace(ax, centroid_trace, frame_id=frame_id,
                             max_gap_s=centroid_max_gap_s)
    is_ball = players_frame["is_ball"].astype(bool)
    players = players_frame[~is_ball]
    has_jersey = "jersey_number" in players_frame.columns

    for team in ("home", "away"):
        team_rows = players[players["team"].astype(str) == team]
        if team_rows.empty:
            continue
        color = TEAM_COLORS[team]
        for gk, marker, label in ((False, "o", team), (True, "s", f"{team} GK")):
            sel = team_rows[team_rows["is_gk"].astype(bool) == gk]
            if sel.empty:
                continue
            pitch.scatter(
                sel["x_pitch"], sel["y_pitch"], ax=ax, s=260, marker=marker,
                color=color, edgecolors="black" if gk else "white",
                linewidth=2.0 if gk else 1.2, zorder=4, label=label,
            )
        if has_jersey:
            for row in team_rows.itertuples(index=False):
                if pd.isna(row.jersey_number):
                    continue
                ax.text(
                    row.x_pitch, row.y_pitch, str(int(row.jersey_number)),
                    color="white", fontsize=7, fontweight="bold",
                    ha="center", va="center", zorder=5,
                )

    ball = players_frame[is_ball]
    if not ball.empty:
        pitch.scatter(
            ball["x_pitch"], ball["y_pitch"], ax=ax, s=80, color="white",
            edgecolors=TEAM_COLORS["ball"], linewidth=1.5, zorder=6, label="ball",
        )

    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=7, frameon=False,
                  fontsize=9)
    if title:
        ax.set_title(title, fontsize=12)
    return fig


def _elapsed_time(df: pd.DataFrame) -> pd.Series:
    periods = sorted(df["period"].unique())
    offsets: dict = {}
    running = 0.0
    for p in periods:
        offsets[p] = running
        ts = np.sort(df.loc[df["period"] == p, "timestamp"].unique())
        step = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.0
        running += float(ts[-1]) + step
    return df["timestamp"] + df["period"].map(offsets).astype(float)


def time_series(
    shape_frame: pd.DataFrame,
    *,
    columns: list[str],
    window: tuple[float, float] | None = None,
) -> Figure:
    """Line-height / width / compactness over time, both teams.

    ``shape_frame`` is the per-(frame, team) output of
    ``metrics.shape.compute_all``. ``columns`` picks which metrics to
    draw (one subplot each, shared x). ``window`` optionally restricts to
    a (start_s, end_s) slice — the dashboard passes its slider value here.

    Period handling: ``timestamp`` restarts every period, so the x axis is
    elapsed match time = ``timestamp`` + the summed durations (max
    timestamp plus one frame step) of earlier periods present in
    ``shape_frame``. With a single period this is just ``timestamp``.
    ``window`` is applied on that x axis. Each (team, period) is drawn as
    its own line, so nothing is connected across a period boundary;
    boundaries get a dashed vertical rule.

    Returns the ``Figure``.
    """
    if not columns:
        raise ValueError("columns must name at least one metric")
    missing = [c for c in columns if c not in shape_frame.columns]
    if missing:
        raise KeyError(f"shape_frame is missing columns: {missing}")

    df = shape_frame.copy()
    df["_t"] = _elapsed_time(df) if len(df) else df["timestamp"]
    boundaries = df.groupby("period")["_t"].min().sort_values().iloc[1:].tolist()
    if window is not None:
        start, end = window
        df = df[(df["_t"] >= start) & (df["_t"] <= end)]
        boundaries = [b for b in boundaries if start <= b <= end]
    df = df.sort_values(["team", "period", "_t"])

    fig, axes = plt.subplots(
        len(columns), 1, sharex=True, figsize=(10, 2.4 * len(columns) + 0.6), squeeze=False
    )
    axes = axes[:, 0]
    for ax, col in zip(axes, columns, strict=True):
        for team in ("home", "away"):
            team_df = df[df["team"].astype(str) == team]
            for i, (_, seg) in enumerate(team_df.groupby("period", sort=True)):
                ax.plot(
                    seg["_t"], seg[col], color=TEAM_COLORS[team], linewidth=1.2,
                    label=team if i == 0 else None,
                )
        for b in boundaries:
            ax.axvline(b, color="#999999", linestyle="--", linewidth=0.8)
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(loc="upper right", fontsize=9)
    if window is not None:
        axes[-1].set_xlim(window)
    axes[-1].set_xlabel("elapsed match time (s)")
    fig.tight_layout()
    return fig


def sensitivity_heatmap(grid: pd.DataFrame) -> Figure:
    """Heatmap of sprint count over the (window x threshold) grid.

    ``grid`` is ``metrics.physical.count_sprints_grid`` output (columns
    savgol_window_frames, sprint_threshold_mps, total_sprints, ...).
    Rows = Savitzky-Golay window, columns = sprint threshold. Each cell is
    annotated with its count; the title carries the headline spread from
    ``metrics.physical.sprint_spread_pct`` (the same number the sensitivity
    report writes), so the figure and ``reports/sensitivity.json`` never
    disagree.
    """
    table = grid.pivot_table(
        index="savgol_window_frames", columns="sprint_threshold_mps",
        values="total_sprints", aggfunc="first",
    ).sort_index().sort_index(axis=1)
    values = table.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]

    fig, ax = plt.subplots(figsize=(1.4 * table.shape[1] + 3, 0.8 * table.shape[0] + 2))
    im = ax.imshow(values, cmap="viridis", aspect="auto", origin="upper")
    ax.set_xticks(range(table.shape[1]), [f"{c:g}" for c in table.columns])
    ax.set_yticks(range(table.shape[0]), [f"{r:g}" for r in table.index])
    ax.set_xlabel("sprint threshold (m/s)")
    ax.set_ylabel("Savitzky-Golay window (frames)")

    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        mid = (lo + hi) / 2
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                v = values[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(
                    j, i, f"{v:g}", ha="center", va="center", fontsize=10,
                    color="black" if v > mid else "white",
                )
        try:
            spread = sprint_spread_pct(grid)
        except ValueError:
            ax.set_title(f"Total sprints across settings: {lo:g}-{hi:g}")
        else:
            ax.set_title(f"Total sprints across settings: {spread:.0f}% spread")
    else:
        ax.set_title("Total sprints across settings")

    fig.colorbar(im, ax=ax, label="total sprints")
    fig.tight_layout()
    return fig


def coverage_bar(coverage: pd.DataFrame) -> Figure:
    """Per-player coverage % bar chart, Metrica vs SkillCorner side by side.

    ``coverage`` has columns provider, track_id, team, coverage_pct
    (0-100). One subplot per provider (sorted by coverage, descending),
    bars coloured by team, shared 0-100 y axis. Makes the "broadcast data
    drops players" point visually for the README coverage section.
    """
    providers = list(dict.fromkeys(coverage["provider"].astype(str)))
    if not providers:
        raise ValueError("coverage is empty")
    fig, axes = plt.subplots(
        1, len(providers), sharey=True, figsize=(6 * len(providers), 4.5), squeeze=False
    )
    axes = axes[0]
    for ax, provider in zip(axes, providers, strict=True):
        sub = coverage[coverage["provider"].astype(str) == provider].sort_values(
            "coverage_pct", ascending=False
        )
        x = np.arange(len(sub))
        ax.bar(x, sub["coverage_pct"], color=[_team_color(t) for t in sub["team"]], width=0.8)
        ax.set_xticks(x, sub["track_id"].astype(str), rotation=90, fontsize=7)
        mean = float(sub["coverage_pct"].mean()) if len(sub) else float("nan")
        ax.axhline(mean, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_title(f"{provider} (mean {mean:.0f}%)")
        ax.set_xlabel("player (track_id)")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("frames tracked (%)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=TEAM_COLORS[t]) for t in ("home", "away")
    ]
    axes[-1].legend(handles, ["home", "away"], loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: Path, *, dpi: int = 120) -> Path:
    """Save ``fig`` to ``path`` (creating parents), close it, return path.

    The one sanctioned file-writing helper in this module so scripts stay
    tidy.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
