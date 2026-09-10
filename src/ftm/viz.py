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

import pandas as pd


def pitch_snapshot(
    players_frame: pd.DataFrame,
    *,
    voronoi_frame: pd.DataFrame | None = None,
    title: str | None = None,
):
    """Draw player positions for one frame, optional Voronoi overlay.

    Parameters
    ----------
    players_frame:
        One frame's rows: track_id, team, x_pitch, y_pitch, is_gk, is_ball.
    voronoi_frame:
        Optional per-player ``area`` + cell polygons for the same frame;
        shade each cell by team colour at low alpha.
    title:
        e.g. "Metrica Game 3 - 62:14 - ball alive".

    Use ``mplsoccer.Pitch`` with ``pitch_type="secondspectrum"`` (or
    "custom" 105x68, origin center) so the coordinates line up with the
    canonical frame without rescaling. Home/away/ball get 3 fixed colours
    defined once at module level.

    Returns the ``Figure``. This is README figure #1.
    """
    raise NotImplementedError


def time_series(
    shape_frame: pd.DataFrame,
    *,
    columns: list[str],
    window: tuple[float, float] | None = None,
):
    """Line-height / width / compactness over time, both teams.

    ``shape_frame`` is the per-(frame, team) output of
    ``metrics.shape.compute_all``. ``columns`` picks which metrics to
    draw (one subplot each, shared x = timestamp). ``window`` optionally
    restricts to a (start_s, end_s) slice — the dashboard passes its
    slider value here.

    Returns the ``Figure``.
    """
    raise NotImplementedError


def sensitivity_heatmap(grid: pd.DataFrame):
    """Heatmap of sprint count over the (window x threshold) grid.

    ``grid`` is ``metrics.physical.count_sprints_grid`` output. Annotate
    each cell with the count; title carries the headline "N% spread".
    Saved to ``reports/figures/`` and embedded in the README next to the
    sensitivity table.
    """
    raise NotImplementedError


def coverage_bar(coverage: pd.DataFrame):
    """Per-player coverage % bar chart, Metrica vs SkillCorner side by side.

    Makes the "broadcast data drops players" point visually for the
    README coverage section.
    """
    raise NotImplementedError


def save_figure(fig, path: Path, *, dpi: int = 120) -> Path:
    """Save ``fig`` to ``path`` (creating parents), close it, return path.

    The one sanctioned file-writing helper in this module so scripts stay
    tidy.
    """
    raise NotImplementedError
