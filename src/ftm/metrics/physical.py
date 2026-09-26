"""Physical metrics: distance covered, speed zones, high-speed running,
sprints — and the inputs to B1's headline honest result.

Requires kinematics columns (``speed``, ``step_dist``) already added by
``ftm.kinematics.add_kinematics``.

Thresholds are CONVENTIONAL, not universal — providers disagree on them.
That disagreement is the point of the sensitivity analysis. All values
from ``metrics.yaml``:
    physical:
      hsr_threshold_mps: 5.5
      sprint_threshold_mps: 7.0
      sprint_min_duration_s: 1.0
      sprint_min_recovery_s: 1.0     # gap required between two sprints
      speed_zones_mps: [[0,2],[2,4],[4,5.5],[5.5,7.0],[7.0,99]]

Nothing here knows which provider a frame came from. Whether a player's
totals are reported is decided by tracking coverage instead
(:func:`apply_coverage_rule`, ``physical.min_coverage_pct``): a distance
built from a track seen for 40% of the match is an undercount that looks
like a real number, whether the gaps come from a broadcast camera, a
substitution or any future tracking source.

Filters applied by every function here: ball rows (``is_ball``) are
dropped; goalkeepers and dead-ball frames are kept (a player runs whether
or not the ball is in play). NaN ``speed`` / ``step_dist`` (track gaps)
count as not moving: zero distance, never above a threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PHYSICAL_METRIC_COLUMNS = ["distance_m", "hsr_distance_m", "n_sprints"]

SPRINT_COLUMNS = [
    "track_id",
    "team",
    "period",
    "start_ts",
    "end_ts",
    "duration_s",
    "peak_speed_mps",
    "mean_speed_mps",
    "distance_m",
]

GRID_COLUMNS = [
    "savgol_window_frames",
    "sprint_threshold_mps",
    "total_sprints",
    "sprints_per_player_mean",
]


def _players(df_with_kin: pd.DataFrame) -> pd.DataFrame:
    """Non-ball rows with NaN speed/step_dist replaced by 0 (not moving)."""
    missing = {"speed", "step_dist"} - set(df_with_kin.columns)
    if missing:
        raise KeyError(
            f"missing kinematics columns {sorted(missing)}; run ftm.kinematics.add_kinematics"
        )
    out = df_with_kin.loc[~df_with_kin["is_ball"].astype(bool)].copy()
    out["speed"] = out["speed"].astype("float64").fillna(0.0)
    out["step_dist"] = out["step_dist"].astype("float64").fillna(0.0)
    return out


def _sum_by(df: pd.DataFrame, keys: list[str], value: pd.Series, name: str) -> pd.DataFrame:
    return (
        df.assign(**{name: value})
        .groupby(keys, observed=True, sort=True)[name]
        .sum()
        .reset_index()
    )


def distance_covered(df_with_kin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Total distance per player, per period and for the whole input.

    Distance is the sum of ``step_dist`` (m).

    Returns a tuple ``(per_period, per_match)``:

    * ``per_period``: (track_id, team, period, distance_m), one row per
      player per period they appear in.
    * ``per_match``: (track_id, team, distance_m), the sum over periods.

    Sanity band: full-match outfield totals of roughly 9-12 km
    (``physiological_ranges.total_distance_km``).
    """
    players = _players(df_with_kin)
    per_period = _sum_by(
        players, ["track_id", "team", "period"], players["step_dist"], "distance_m"
    )
    per_match = _sum_by(players, ["track_id", "team"], players["step_dist"], "distance_m")
    return per_period, per_match


def zone_label(low: float, high: float, *, last: bool) -> str:
    """Label for a ``[low, high)`` speed band: ``"4-5.5"``, or ``"7+"`` for
    the open-ended last band. Numbers use ``%g`` formatting (m/s)."""
    return f"{low:g}+" if last else f"{low:g}-{high:g}"


def speed_zones(
    df_with_kin: pd.DataFrame,
    *,
    zones_mps: list[tuple[float, float]],
) -> pd.DataFrame:
    """Distance per player in each speed band.

    Bands are ``[low, high)`` and must tile the speed axis: sorted, each
    ``low`` equal to the previous ``high``. The last band's upper bound is
    treated as infinite, so every frame lands in exactly one band and the
    zone distances sum to ``distance_covered``. Binning is done here
    rather than via ``ftm.kinematics`` because metrics may only import
    ``ftm.schema``. Labels come from :func:`zone_label` (e.g. ``"0-2"``,
    ``"2-4"``, ``"4-5.5"``, ``"5.5-7"``, ``"7+"``).

    Returns: (track_id, team, zone_label, distance_m), every zone present
    for every player (0.0 when unused), rows ordered by player then zone.
    """
    zones = [(float(lo), float(hi)) for lo, hi in zones_mps]
    if not zones:
        raise ValueError("zones_mps must contain at least one band")
    for (lo, hi), (next_lo, _) in zip(zones, zones[1:] + [(None, None)], strict=True):
        if hi <= lo:
            raise ValueError(f"speed zone [{lo}, {hi}) is empty or reversed")
        if next_lo is not None and next_lo != hi:
            raise ValueError(f"speed zones must tile the axis; gap/overlap at {hi} vs {next_lo}")

    labels = [zone_label(lo, hi, last=i == len(zones) - 1) for i, (lo, hi) in enumerate(zones)]
    edges = [lo for lo, _ in zones[1:]]
    players = _players(df_with_kin)
    zone_idx = np.searchsorted(edges, players["speed"].to_numpy(), side="right")
    zone_cat = pd.Categorical.from_codes(zone_idx, categories=labels, ordered=True)

    summed = (
        players.assign(zone_label=zone_cat)
        .groupby(["track_id", "team", "zone_label"], observed=False, sort=True)["step_dist"]
        .sum()
    )
    present = players[["track_id", "team"]].drop_duplicates()
    index = pd.MultiIndex.from_frame(present).sort_values()
    full = pd.MultiIndex.from_tuples(
        [(t, tm, z) for t, tm in index for z in labels],
        names=["track_id", "team", "zone_label"],
    )
    out = summed.reindex(full, fill_value=0.0).rename("distance_m").reset_index()
    out["zone_label"] = pd.Categorical(out["zone_label"], categories=labels, ordered=True)
    return out


def high_speed_running(
    df_with_kin: pd.DataFrame,
    *,
    threshold_mps: float,
) -> pd.DataFrame:
    """Distance covered at ``speed >= threshold_mps`` (default 5.5) per player.

    Returns: (track_id, team, hsr_distance_m); players who never reach the
    threshold get 0.0.
    """
    players = _players(df_with_kin)
    fast = players["step_dist"].where(players["speed"] >= threshold_mps, 0.0)
    return _sum_by(players, ["track_id", "team"], fast, "hsr_distance_m")


def _true_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and end (exclusive) indices of True runs."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def detect_sprints(
    df_with_kin: pd.DataFrame,
    *,
    threshold_mps: float,
    min_duration_s: float,
    min_recovery_s: float,
    frame_rate: float,
) -> pd.DataFrame:
    """Find sprint efforts per player.

    A sprint = a run of consecutive frames with ``speed >= threshold_mps``
    lasting at least ``min_duration_s``. Two candidate runs separated by
    less than ``min_recovery_s`` below threshold are merged into one
    effort (prevents double-counting a single sprint that dips for one
    noisy frame).

    Algorithm per (track_id, period), sorted by timestamp:
    1. Boolean mask ``speed >= threshold`` (NaN speed is False).
    2. Close interior below-threshold gaps shorter than
       ``min_recovery_s * frame_rate`` frames (only gaps with a True run
       on both sides).
    3. Label contiguous True runs; keep those with duration
       >= ``min_duration_s``.
    4. One row per surviving run.

    Duration is ``n_frames / frame_rate`` where ``n_frames`` counts every
    frame in the (merged) run, including closed gap frames. ``start_ts`` /
    ``end_ts`` are the timestamps of the first / last frame of the run.
    Peak, mean speed and distance (sum of ``step_dist``) cover the same
    frames. Rows are assumed consecutive frames at ``frame_rate``.

    Returns: (track_id, team, period, start_ts, end_ts, duration_s,
    peak_speed_mps, mean_speed_mps, distance_m) — an empty frame with
    these columns when nothing qualifies.

    A ``sprint_count`` per player is just a group-by size on this. That
    count is what the sensitivity sweep perturbs — see
    ``count_sprints_grid``.
    """
    players = _players(df_with_kin).sort_values(["track_id", "period", "timestamp"])
    max_gap_frames = min_recovery_s * frame_rate
    rows: list[dict] = []
    for (track_id, period), grp in players.groupby(
        ["track_id", "period"], observed=True, sort=True
    ):
        speed = grp["speed"].to_numpy()
        starts, ends = _true_runs(speed >= threshold_mps)
        if len(starts) == 0:
            continue
        keep_break = (starts[1:] - ends[:-1]) >= max_gap_frames
        starts = starts[np.concatenate([[True], keep_break])]
        ends = ends[np.concatenate([keep_break, [True]])]

        ts = grp["timestamp"].to_numpy()
        step = grp["step_dist"].to_numpy()
        team = grp["team"].iloc[0]
        for s, e in zip(starts, ends, strict=True):
            duration = (e - s) / frame_rate
            if duration < min_duration_s:
                continue
            rows.append(
                {
                    "track_id": track_id,
                    "team": team,
                    "period": period,
                    "start_ts": float(ts[s]),
                    "end_ts": float(ts[e - 1]),
                    "duration_s": float(duration),
                    "peak_speed_mps": float(speed[s:e].max()),
                    "mean_speed_mps": float(speed[s:e].mean()),
                    "distance_m": float(step[s:e].sum()),
                }
            )
    return pd.DataFrame(rows, columns=SPRINT_COLUMNS)


def sprint_count(sprints: pd.DataFrame) -> pd.DataFrame:
    """Collapse ``detect_sprints`` output to counts per player.

    Only players with at least one sprint appear (the input carries no
    record of players who never sprinted).

    Returns: (track_id, team, n_sprints, total_sprint_distance_m).
    """
    return (
        sprints.groupby(["track_id", "team"], observed=True, sort=True)
        .agg(n_sprints=("distance_m", "size"), total_sprint_distance_m=("distance_m", "sum"))
        .reset_index()
    )


def apply_coverage_rule(
    physical: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    min_coverage_pct: float,
) -> pd.DataFrame:
    """Blank the physical metrics of tracks with too little coverage.

    ``physical`` has one row per (track_id, team) with the
    :data:`PHYSICAL_METRIC_COLUMNS`; ``coverage`` is (track_id, team,
    coverage_pct), e.g. ``pipeline.coverage_table``'s per-match frame. A
    track below ``min_coverage_pct`` keeps its row, but its distance, HSR
    and sprint count become missing (NaN / ``<NA>``), and
    ``below_coverage`` is True. Tracks absent from ``coverage`` count as
    0% covered.

    Returns ``physical`` plus ``coverage_pct`` and ``below_coverage``;
    ``n_sprints`` becomes a nullable ``Int64``.
    """
    missing = set(PHYSICAL_METRIC_COLUMNS) - set(physical.columns)
    if missing:
        raise KeyError(f"physical frame is missing {sorted(missing)}")
    out = physical.drop(columns=["coverage_pct", "below_coverage"], errors="ignore").merge(
        coverage[["track_id", "team", "coverage_pct"]], on=["track_id", "team"], how="left"
    )
    out["coverage_pct"] = out["coverage_pct"].astype("float64").fillna(0.0)
    below = out["coverage_pct"] < min_coverage_pct
    out["below_coverage"] = below.astype(bool)
    out["distance_m"] = out["distance_m"].astype("float64").mask(below)
    out["hsr_distance_m"] = out["hsr_distance_m"].astype("float64").mask(below)
    out["n_sprints"] = out["n_sprints"].astype("Int64").mask(below)
    return out


def count_sprints_grid(
    df_with_kin_by_window: dict[int, pd.DataFrame],
    *,
    threshold_grid_mps: list[float],
    min_duration_s: float,
    min_recovery_s: float,
    frame_rate: float,
    mean_over_track_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Sprint counts across a (smoothing window x sprint threshold) grid.

    This produces the numbers behind B1's headline sentence:
    *"sprint count varies by N% across defensible threshold choices, so I
    report the parameters alongside the number."*

    Parameters
    ----------
    df_with_kin_by_window:
        Map ``savgol_window_frames -> canonical frame re-run through
        ``add_kinematics`` with that window``. Built by
        ``scripts/validate.py``.
    threshold_grid_mps:
        Sprint-speed thresholds to sweep, e.g. [6.5, 7.0, 7.5, 8.0].

    Returns
    -------
    tidy frame: (savgol_window_frames, sprint_threshold_mps,
    total_sprints, sprints_per_player_mean), one row per grid cell,
    sorted by window then threshold. ``total_sprints`` counts every
    non-ball track. ``sprints_per_player_mean`` is the mean over
    ``mean_over_track_ids`` (e.g. full-match outfield players, counting
    only their sprints and including those with zero), or over every
    non-ball track when that is None. ``scripts/validate.py`` turns this into
    ``reports/sensitivity.json`` and the README table, including the
    spread from :func:`sprint_spread_pct` (the "N%").
    """
    rows = []
    for window in sorted(df_with_kin_by_window):
        df = df_with_kin_by_window[window]
        if mean_over_track_ids is None:
            mean_ids = set(df.loc[~df["is_ball"].astype(bool), "track_id"].astype(str))
        else:
            mean_ids = {str(t) for t in mean_over_track_ids}
        for threshold in sorted(threshold_grid_mps):
            sprints = detect_sprints(
                df,
                threshold_mps=threshold,
                min_duration_s=min_duration_s,
                min_recovery_s=min_recovery_s,
                frame_rate=frame_rate,
            )
            n_mean = int(sprints["track_id"].astype(str).isin(mean_ids).sum())
            rows.append(
                {
                    "savgol_window_frames": int(window),
                    "sprint_threshold_mps": float(threshold),
                    "total_sprints": int(len(sprints)),
                    "sprints_per_player_mean": n_mean / len(mean_ids) if mean_ids else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=GRID_COLUMNS)


def _lower_median(values: pd.Series) -> float:
    uniq = np.sort(values.unique())
    return uniq[(len(uniq) - 1) // 2]


def sprint_spread_pct(
    grid: pd.DataFrame,
    *,
    baseline: tuple[int, float] | None = None,
) -> float:
    """Spread of ``total_sprints`` across the grid as a % of a reference cell.

    ``(max - min) / reference * 100``. The reference cell is
    ``baseline = (savgol_window_frames, sprint_threshold_mps)`` when given
    (e.g. the config defaults), else the mid-grid cell: the lower median
    of the distinct windows x the lower median of the distinct thresholds
    (lower median = element ``(n - 1) // 2`` of the sorted values, so it is
    always a real grid value). Raises ``ValueError`` if the reference cell
    is missing or has zero sprints.
    """
    if baseline is None:
        baseline = (
            _lower_median(grid["savgol_window_frames"]),
            _lower_median(grid["sprint_threshold_mps"]),
        )
    window, threshold = baseline
    ref = grid.loc[
        (grid["savgol_window_frames"] == window)
        & np.isclose(grid["sprint_threshold_mps"], threshold),
        "total_sprints",
    ]
    if ref.empty:
        raise ValueError(f"reference cell {baseline} not in grid")
    ref_value = float(ref.iloc[0])
    if ref_value == 0:
        raise ValueError(f"reference cell {baseline} has zero sprints; spread undefined")
    totals = grid["total_sprints"]
    return float((totals.max() - totals.min()) / ref_value * 100.0)
