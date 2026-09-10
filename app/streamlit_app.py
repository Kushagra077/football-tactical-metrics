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
"""

from __future__ import annotations

from pathlib import Path

CACHE_DIR = Path("data/cache")


def list_cached_matches(cache_dir: Path) -> list[dict]:
    """Scan ``cache_dir`` for ``*__meta.json`` and return one dict per
    match: provider, match_id, teams, periods, target_hz, label. Cached
    with ``@st.cache_resource``."""
    raise NotImplementedError


def load_tidy(cache_dir: Path, provider: str, match_id: str, name: str):
    """Read one tidy parquet (``<provider>__<match_id>__<name>.parquet``)
    into a DataFrame. Decorated with ``@st.cache_data``. ``name`` is one
    of {tracking, shape, pressing, space, physical, coverage}."""
    raise NotImplementedError


def sidebar_controls(matches: list[dict]) -> dict:
    """Render the sidebar widgets, return the current selection dict
    (provider, match_id, period, window, show_voronoi, series)."""
    raise NotImplementedError


def render_pitch(selection: dict) -> None:
    """Load ``tracking`` + ``space`` for the window midpoint frame and
    ``st.pyplot(ftm.viz.pitch_snapshot(...))``."""
    raise NotImplementedError


def render_time_series(selection: dict) -> None:
    """Load ``shape``, slice to the window, ``st.pyplot(
    ftm.viz.time_series(...))``."""
    raise NotImplementedError


def render_player_table(selection: dict) -> None:
    """Join ``physical`` + ``coverage``; show coverage % beside every
    physical column; blank the distance columns for SkillCorner."""
    raise NotImplementedError


def main() -> None:
    """``st.set_page_config`` -> load index -> sidebar -> the three
    render_* panels. No metric computation anywhere in this call tree."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
