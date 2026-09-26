"""Streamlit dashboard — reads parquet only, NEVER computes.

Computing 25 Hz metrics inside a callback means every slider drag
recomputes the match and the HF free-CPU Space feels broken. All heavy
work happened in ``scripts/build_cache.py``; this file loads tidy parquet
from ``data/cache/`` and draws it.

Gate (spec step 7): a slider drag re-renders in under a second locally.

Deploy: HF Spaces, Streamlit SDK, CPU Basic. Free Spaces sleep after
inactivity — say so in the Space description.

--------------------------------------------------------------------------
Layout
--------------------------------------------------------------------------
Sidebar:
  * match selector — every cached match, provider shown in the label
    e.g. "SkillCorner · 1234 · Team A vs Team B"
  * time-window slider — (start_s, end_s) within the selected period
  * period selector (1 / 2)
  * toggles: Voronoi overlay on/off, which shape series to show

Main:
  1. Pitch snapshot at the window midpoint — player positions + Voronoi
     overlay (``ftm.viz.pitch_snapshot``)
  2. Time series — line height / width / compactness for both teams over
     the window (``ftm.viz.time_series``)
  3. Per-player table — physical metrics with coverage % shown next to
     EVERY physical number; for SkillCorner the distance columns read
     "n/a (broadcast)" rather than a misleading value

Caching:
  * ``@st.cache_data`` on every parquet load, keyed by file path + mtime
  * ``@st.cache_resource`` for the list-of-matches index

Cache dir: ``$FTM_CACHE_DIR`` if set, else ``<repo>/data/cache``.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ftm import viz
from ftm.metrics.physical import REFUSED_PROVIDERS

CACHE_DIR_ENV = "FTM_CACHE_DIR"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"

TIDY_NAMES = frozenset(
    {"tracking", "shape", "pressing", "space", "space_player", "physical", "coverage"}
)
PROVIDER_LABELS = {"metrica": "Metrica", "skillcorner": "SkillCorner"}
SHAPE_SERIES = {
    "line_height_m": "Line height (m)",
    "width_m": "Width (m)",
    "length_m": "Length (m)",
    "hull_area_m2": "Compactness: hull area (m²)",
}
DEFAULT_SERIES = ["line_height_m", "width_m", "hull_area_m2"]
PHYSICAL_COLUMNS = {
    "distance_m": "Distance (m)",
    "hsr_distance_m": "HSR distance (m)",
    "n_sprints": "Sprints",
}
NA_BROADCAST = "n/a (broadcast)"
NO_MATCHES_MSG = (
    "No cached matches found in `{cache_dir}` — run `uv run python scripts/build_cache.py` "
    "first (or point `$FTM_CACHE_DIR` at a cache directory)."
)


def resolve_cache_dir() -> Path:
    override = os.environ.get(CACHE_DIR_ENV)
    return Path(override).expanduser() if override else CACHE_DIR


# ---------------------------------------------------------------------------
# Loading (cached)
# ---------------------------------------------------------------------------


def _meta_fingerprint(cache_dir: Path) -> tuple[tuple[str, float], ...]:
    return tuple(
        sorted((p.name, p.stat().st_mtime) for p in Path(cache_dir).glob("*__meta.json"))
    )


def _match_label(provider: str, match_id: str, home: str, away: str) -> str:
    name = PROVIDER_LABELS.get(provider, provider)
    return f"{name} · {match_id} · {home} vs {away}"


def _parse_periods(raw) -> dict[int, tuple[float, float]]:
    periods: dict[int, tuple[float, float]] = {}
    for key, bounds in (raw or {}).items():
        try:
            start, end = float(bounds[0]), float(bounds[1])
            periods[int(key)] = (start, end)
        except (TypeError, ValueError, IndexError):
            continue
    return dict(sorted(periods.items()))


def scan_matches(cache_dir: Path) -> list[dict]:
    """Uncached body of ``list_cached_matches``."""
    matches: list[dict] = []
    for meta_path in sorted(Path(cache_dir).glob("*__meta.json")):
        parts = meta_path.name[: -len("__meta.json")].split("__", 1)
        if len(parts) != 2 or not all(parts):
            continue
        provider, match_id = parts
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        mm = meta.get("match_meta") or {}
        home = str(mm.get("home_team") or "home")
        away = str(mm.get("away_team") or "away")
        matches.append(
            {
                "provider": provider,
                "match_id": match_id,
                "home_team": home,
                "away_team": away,
                "periods": _parse_periods(mm.get("periods")),
                "target_hz": meta.get("cached_frame_rate_hz"),
                "physical_note": meta.get("physical_note") or "",
                "label": _match_label(provider, match_id, home, away),
            }
        )
    return matches


@st.cache_resource(show_spinner=False)
def _cached_index(cache_dir: str, fingerprint: tuple) -> list[dict]:
    return scan_matches(Path(cache_dir))


def list_cached_matches(cache_dir: Path) -> list[dict]:
    """Scan ``cache_dir`` for ``*__meta.json`` and return one dict per
    match: provider, match_id, teams, periods, target_hz, label. Cached
    with ``@st.cache_resource``, re-scanned when any meta file changes."""
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return []
    fingerprint = _meta_fingerprint(cache_dir)
    if not fingerprint:
        return []
    return _cached_index(str(cache_dir), fingerprint)


@st.cache_data(show_spinner=False, max_entries=32)
def _read_parquet(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_tidy(cache_dir: Path, provider: str, match_id: str, name: str) -> pd.DataFrame:
    """Read one tidy parquet (``<provider>__<match_id>__<name>.parquet``)
    into a DataFrame. Cached via ``@st.cache_data`` keyed by path + mtime.
    ``name`` is one of {tracking, shape, pressing, space, space_player,
    physical, coverage}. A missing file yields an empty frame."""
    if name not in TIDY_NAMES:
        raise ValueError(f"unknown tidy artifact {name!r}; expected one of {sorted(TIDY_NAMES)}")
    path = Path(cache_dir) / f"{provider}__{match_id}__{name}.parquet"
    if not path.is_file():
        return pd.DataFrame()
    return _read_parquet(str(path), path.stat().st_mtime)


@st.cache_data(show_spinner=False, max_entries=32)
def _period_ranges(path: str, mtime: float) -> dict[int, tuple[float, float]]:
    ts = pd.read_parquet(path, columns=["period", "timestamp"])
    agg = ts.groupby("period")["timestamp"].agg(["min", "max"]).sort_index()
    return {int(p): (float(r["min"]), float(r["max"])) for p, r in agg.iterrows()}


def tracking_period_ranges(
    cache_dir: Path, provider: str, match_id: str
) -> dict[int, tuple[float, float]]:
    """period -> (min, max) tracking timestamp, read from two columns only."""
    path = Path(cache_dir) / f"{provider}__{match_id}__tracking.parquet"
    if not path.is_file():
        return {}
    return _period_ranges(str(path), path.stat().st_mtime)


def load_all_coverage(cache_dir: Path, matches: list[dict]) -> pd.DataFrame:
    frames = [load_tidy(cache_dir, m["provider"], m["match_id"], "coverage") for m in matches]
    frames = [f for f in frames if len(f) and "provider" in f.columns]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Pure data shaping
# ---------------------------------------------------------------------------


def period_bounds(
    match: dict, period: int, ranges: dict[int, tuple[float, float]] | None = None
) -> tuple[float, float]:
    """(start_s, end_s) slider bounds in period-relative seconds.

    Tracking timestamps restart every period, but ``meta.periods`` holds
    match-clock bounds (period 2 starts ~2700 s), so the tracking ranges
    win; meta is only a fallback, shifted to start at 0.
    """
    if ranges and period in ranges:
        lo, hi = ranges[period]
        return lo, max(hi, lo + 1.0)
    bounds = match.get("periods", {}).get(period)
    if bounds and bounds[1] > bounds[0]:
        return 0.0, float(bounds[1] - bounds[0])
    return 0.0, 1.0


def nearest_frame(tracking: pd.DataFrame, period: int, t: float) -> pd.DataFrame:
    """All rows of the frame in ``period`` whose timestamp is nearest ``t``."""
    rows = tracking[tracking["period"] == period]
    if rows.empty:
        return rows
    frames = rows.drop_duplicates("frame_id")[["frame_id", "timestamp"]]
    fid = frames.loc[(frames["timestamp"] - t).abs().idxmin(), "frame_id"]
    return rows[rows["frame_id"] == fid].reset_index(drop=True)


def frame_cells(space_player: pd.DataFrame, period: int, frame_id: int) -> pd.DataFrame:
    if space_player.empty or "cell_wkt" not in space_player.columns:
        return pd.DataFrame()
    sel = (space_player["period"] == period) & (space_player["frame_id"] == frame_id)
    return space_player[sel].reset_index(drop=True)


def is_broadcast(match: dict) -> bool:
    return bool(match.get("physical_note")) or match.get("provider") in REFUSED_PROVIDERS


def _fmt_cov(value: float, cov: float, *, integer: bool = False) -> str:
    if pd.isna(value):
        return "—"
    num = f"{int(value):d}" if integer else f"{value:,.0f}"
    cov_txt = "?" if pd.isna(cov) else f"{cov:.0f}%"
    return f"{num} ({cov_txt} cov)"


def build_player_table(
    physical: pd.DataFrame, coverage: pd.DataFrame, *, broadcast: bool
) -> pd.DataFrame:
    """Per-player display table: coverage % beside every physical value.

    Rows come from ``coverage`` (every tracked player) left-joined with
    ``physical`` on (track_id, team). For broadcast providers every
    physical column reads ``NA_BROADCAST`` — never 0, which would look real.
    """
    keys = ["track_id", "team"]
    cov = coverage.copy() if len(coverage) else pd.DataFrame(columns=[*keys, "coverage_pct"])
    cov = cov[[*keys, "coverage_pct"]]
    for k in keys:
        cov[k] = cov[k].astype(str)

    if broadcast or physical.empty:
        table = cov.copy()
        for label in PHYSICAL_COLUMNS.values():
            table[label] = NA_BROADCAST if broadcast else "—"
    else:
        phys = physical[[*keys, *[c for c in PHYSICAL_COLUMNS if c in physical.columns]]].copy()
        for k in keys:
            phys[k] = phys[k].astype(str)
        table = cov.merge(phys, on=keys, how="outer")
        for col, label in PHYSICAL_COLUMNS.items():
            if col not in table.columns:
                table[label] = "—"
                continue
            table[label] = [
                _fmt_cov(v, c, integer=col == "n_sprints")
                for v, c in zip(table[col], table["coverage_pct"], strict=True)
            ]
            table = table.drop(columns=col)

    table = table.sort_values(["team", "coverage_pct"], ascending=[False, False])
    table = table.rename(
        columns={"track_id": "Player", "team": "Team", "coverage_pct": "Coverage %"}
    )
    table["Coverage %"] = table["Coverage %"].astype(float).round(1)
    return table.reset_index(drop=True)


def _mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def sidebar_controls(matches: list[dict], cache_dir: Path | None = None) -> dict:
    """Render the sidebar widgets, return the current selection dict
    (provider, match_id, period, window, show_voronoi, series)."""
    cache_dir = Path(cache_dir) if cache_dir is not None else resolve_cache_dir()
    sb = st.sidebar
    sb.header("Match")
    idx = sb.selectbox(
        "Cached match", range(len(matches)), format_func=lambda i: matches[i]["label"],
        key="match_idx",
    )
    match = matches[idx]
    provider, match_id = match["provider"], match["match_id"]

    ranges = tracking_period_ranges(cache_dir, provider, match_id)
    periods = list(ranges) or list(match["periods"]) or [1]
    period = sb.radio("Period", periods, horizontal=True, key=f"period-{provider}-{match_id}")
    lo, hi = period_bounds(match, period, ranges)
    window = sb.slider(
        "Time window (s into period)", min_value=lo, max_value=hi, value=(lo, hi),
        step=1.0, key=f"window-{provider}-{match_id}-{period}",
    )
    sb.caption(f"{_mmss(window[0])} – {_mmss(window[1])}")

    sb.header("Display")
    show_voronoi = sb.toggle("Voronoi overlay", value=True, key="show_voronoi")
    series = sb.multiselect(
        "Shape series", list(SHAPE_SERIES), default=DEFAULT_SERIES,
        format_func=SHAPE_SERIES.get, key="series",
    )
    if match.get("target_hz"):
        sb.caption(f"Cached at {float(match['target_hz']):g} Hz.")

    return {
        "cache_dir": cache_dir,
        "match": match,
        "provider": provider,
        "match_id": match_id,
        "period": int(period),
        "window": (float(window[0]), float(window[1])),
        "show_voronoi": bool(show_voronoi),
        "series": list(series),
    }


def _show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def render_pitch(selection: dict) -> None:
    """Load ``tracking`` + ``space_player`` for the window midpoint frame and
    ``st.pyplot(ftm.viz.pitch_snapshot(...))``."""
    st.subheader("Pitch snapshot")
    cache_dir, provider, match_id = (
        selection["cache_dir"], selection["provider"], selection["match_id"],
    )
    period = selection["period"]
    start, end = selection["window"]
    tracking = load_tidy(cache_dir, provider, match_id, "tracking")
    frame = nearest_frame(tracking, period, (start + end) / 2) if len(tracking) else tracking
    if frame.empty:
        st.info("No tracking rows for this period.")
        return

    cells = None
    if selection["show_voronoi"]:
        space_player = load_tidy(cache_dir, provider, match_id, "space_player")
        cells = frame_cells(space_player, period, int(frame["frame_id"].iloc[0]))
        if cells.empty:
            st.caption("No Voronoi cells cached for this frame.")
            cells = None

    t = float(frame["timestamp"].iloc[0])
    state = str(frame["ball_state"].iloc[0]) if "ball_state" in frame.columns else ""
    name = PROVIDER_LABELS.get(provider, provider)
    title = f"{name} {match_id} - P{period} {_mmss(t)}" + (f" - ball {state}" if state else "")
    _show(viz.pitch_snapshot(frame, voronoi_frame=cells, title=title))


def render_time_series(selection: dict) -> None:
    """Load ``shape``, slice to the window, ``st.pyplot(
    ftm.viz.time_series(...))``."""
    st.subheader("Team shape over the window")
    if not selection["series"]:
        st.info("Pick at least one shape series in the sidebar.")
        return
    shape = load_tidy(selection["cache_dir"], selection["provider"], selection["match_id"], "shape")
    if shape.empty:
        st.info("No shape metrics cached for this match.")
        return
    # Single period, so viz's elapsed-time axis equals the period timestamp.
    shape = shape[shape["period"] == selection["period"]]
    columns = [c for c in selection["series"] if c in shape.columns]
    start, end = selection["window"]
    in_window = shape[(shape["timestamp"] >= start) & (shape["timestamp"] <= end)]
    if in_window.empty or not columns:
        st.info("No shape rows in this window.")
        return
    _show(viz.time_series(shape, columns=columns, window=(start, end)))


def render_player_table(selection: dict) -> None:
    """Join ``physical`` + ``coverage``; show coverage % beside every
    physical column; distance columns read "n/a (broadcast)" for SkillCorner."""
    st.subheader("Players")
    cache_dir, provider, match_id = (
        selection["cache_dir"], selection["provider"], selection["match_id"],
    )
    physical = load_tidy(cache_dir, provider, match_id, "physical")
    coverage = load_tidy(cache_dir, provider, match_id, "coverage")
    if coverage.empty and physical.empty:
        st.info("No physical or coverage data cached for this match.")
        return
    broadcast = is_broadcast(selection["match"])
    st.dataframe(
        build_player_table(physical, coverage, broadcast=broadcast),
        hide_index=True,
    )
    st.caption(
        "Coverage % = share of the match's frames in which the player was tracked; "
        "it is shown beside every physical value because a low-coverage player's "
        "totals are undercounts."
    )
    if broadcast:
        st.caption(selection["match"].get("physical_note") or (
            "Physical metrics are not reported for broadcast tracking."
        ))


@st.cache_data(show_spinner=False, max_entries=4)
def _coverage_png(cache_dir: str, fingerprint: tuple) -> bytes | None:
    # Independent of every sidebar widget, and the slowest figure (~0.2 s):
    # render once per cache state, not on every slider drag. The expander
    # body runs on each rerun even when collapsed.
    coverage = load_all_coverage(Path(cache_dir), _cached_index(cache_dir, fingerprint))
    if coverage.empty:
        return None
    fig = viz.coverage_bar(coverage)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")  # st.pyplot's defaults
    plt.close(fig)
    return buf.getvalue()


def render_coverage_overview(cache_dir: Path) -> None:
    with st.expander("Coverage across all cached matches"):
        png = _coverage_png(str(cache_dir), _meta_fingerprint(cache_dir))
        if png is None:
            st.info("No coverage data cached.")
            return
        st.image(png, width="stretch")


def main() -> None:
    """``st.set_page_config`` -> load index -> sidebar -> the three
    render_* panels. No metric computation anywhere in this call tree."""
    st.set_page_config(page_title="Football tactical metrics", layout="wide")
    st.title("Football tactical metrics")
    cache_dir = resolve_cache_dir()
    matches = list_cached_matches(cache_dir)
    if not matches:
        st.info(NO_MATCHES_MSG.format(cache_dir=cache_dir))
        return
    selection = sidebar_controls(matches, cache_dir)
    st.caption(selection["match"]["label"])
    render_pitch(selection)
    render_time_series(selection)
    render_player_table(selection)
    render_coverage_overview(cache_dir)


if __name__ == "__main__":
    main()
