"""Write ``reports/validation.json`` and ``reports/sensitivity.json``.

These two committed files are half of B1's "definition of done". A1
committed latency numbers but not accuracy; do not repeat that.

    python -m scripts.validate --config configs/metrics.yaml \
        --cache-dir data/cache --reports-dir reports

Depends on the cache existing (run ``build_cache.py`` first).

--------------------------------------------------------------------------
reports/validation.json  — metrics vs. ground truth
--------------------------------------------------------------------------
1. Synthetic ground truth. The trajectory generators and hand-placed
   formations from ``tests/conftest.py`` are run through the REAL
   kinematics / physical / shape / space functions and compared with
   answers derived by hand (e.g. the flat back four's hull is 908 m^2 by
   the shoelace formula). This proves the shipped code, not the doubles.

2. Published physiological ranges (``physiological_ranges`` in the YAML),
   read from the cached Metrica matches:
     * total distance per full-match outfield player
     * top speed per full-match outfield player; keepers upper bound only
     * HSR share of total distance per full-match outfield player
     * team compactness — only if ``team_compactness_m2`` is configured;
       otherwise the observed medians go under ``not_checked``
   "Full match" = present in the first and last frame of every period, so
   substitutes are never compared against a 90-minute band.
   SkillCorner distance metrics are never computed: those matches appear
   under ``excluded`` with the pipeline's reason.

Shape:
    {generated_at, config_hash, synthetic: [...], physiological: [...],
     excluded: [...], not_checked: [...], coverage: [...], references: [...],
     summary: {n_checks, n_pass, n_fail, all_pass, failures}}

--------------------------------------------------------------------------
reports/sensitivity.json — how much each metric moves with parameters
--------------------------------------------------------------------------
For one Metrica match, the match is RE-LOADED at its native frame rate
(local raw cache under ``data/raw/``, no network) and run through
``add_kinematics`` for every ``savgol_window_frames_grid`` value, then
``count_sprints_grid`` over ``sprint_threshold_mps_grid``. Windows are in
frames at the NATIVE rate — the same rate the pipeline now smooths at
(DD-030): sweeping the cached 5 Hz positions would answer a question the
pipeline no longer asks, since kinematics runs before downsampling.
Spread is ``sprint_spread_pct`` relative to the mid-grid cell (as the
heatmap title); the config-default-referenced spread is recorded too.

Both writers drop figures via ``ftm.viz`` into ``reports/figures/``.

Exit non-zero if any validation check fails (so CI can gate on it).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tests.conftest import (
    circular_track,
    flat_back_four,
    noisy_stationary_track,
    straight_line_track,
)

from ftm import pipeline, viz
from ftm.kinematics import KinematicsConfig, add_kinematics
from ftm.loaders import get_loader
from ftm.metrics import physical, shape, space

# Validation tolerances (not metric thresholds); they mirror tests/test_kinematics.py.
TOL_LINEAR_DISTANCE_M = 1.0
TOL_SPEED_MPS = 0.01
TOL_REL = 0.01
TOL_NOISE_DISTANCE_M = 2.0
TOL_GEOMETRY = 1e-6

# Hand-derived answers for conftest.flat_back_four(x_line=-30): outfield
# points (-30,+-20), (-30,+-7), (-15,+-18), (-15,+-6), (-2,+-8).
FBF_X_LINE = -30.0
FBF_WIDTH_M = 40.0
FBF_LENGTH_M = 28.0
FBF_HULL_M2 = 908.0  # shoelace over the 6 hull vertices
FBF_CENTROID_X = -18.4

REFERENCES = [
    {
        "metrics": ["total_distance_km", "hsr_share_of_distance_pct"],
        "source": "Bradley, P. S. et al. (2009). High-intensity running in English FA Premier "
        "League soccer matches. Journal of Sports Sciences, 27(2), 159-168.",
    },
    {
        "metrics": ["total_distance_km"],
        "source": "Di Salvo, V. et al. (2007). Performance characteristics according to "
        "playing position in elite soccer. International Journal of Sports Medicine, "
        "28(3), 222-227.",
    },
    {
        "metrics": ["top_speed_mps"],
        "source": "Standard sports-science ranges for elite match play (peak outfield "
        "speeds of roughly 8-11 m/s); band as configured in configs/metrics.yaml.",
    },
    {
        "metrics": ["*"],
        "source": "Bands are conventions, not laws: HSR share depends on the HSR "
        "threshold (physical.hsr_threshold_mps). Thresholds are NOT tuned to make "
        "checks pass; see README.",
    },
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def config_hash(metrics_cfg: dict) -> str:
    blob = json.dumps(metrics_cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _num(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else round(x, 4)


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_num, allow_nan=False) + "\n")


def cached_matches(cache_dir: Path) -> list[tuple[str, str]]:
    """(provider, match_id) for every match with a meta.json in the cache."""
    out = []
    for meta in sorted(Path(cache_dir).glob("*__meta.json")):
        provider, match_id = meta.name.split("__")[:2]
        out.append((provider, match_id))
    return out


def _path(cache_dir: Path, provider: str, match_id: str, name: str) -> Path:
    ext = "json" if name == "meta" else "parquet"
    return Path(cache_dir) / f"{provider}__{match_id}__{name}.{ext}"


def _read(cache_dir: Path, provider: str, match_id: str, name: str, **kw) -> pd.DataFrame:
    return pd.read_parquet(_path(cache_dir, provider, match_id, name), **kw)


def _meta(cache_dir: Path, provider: str, match_id: str) -> dict:
    return json.loads(_path(cache_dir, provider, match_id, "meta").read_text())


def full_match_players(tracking: pd.DataFrame) -> set[str]:
    """Tracks present in the first and last player frame of every period."""
    players = tracking.loc[~tracking["is_ball"].astype(bool)]
    keep: set[str] | None = None
    for _, grp in players.groupby("period", observed=True):
        first, last = grp["frame_id"].min(), grp["frame_id"].max()
        at_first = set(grp.loc[grp["frame_id"] == first, "track_id"].astype(str))
        at_last = set(grp.loc[grp["frame_id"] == last, "track_id"].astype(str))
        both = at_first & at_last
        keep = both if keep is None else keep & both
    return keep or set()


def goalkeepers(tracking: pd.DataFrame) -> set[str]:
    gk = tracking.loc[tracking["is_gk"].astype(bool), "track_id"].astype(str)
    return set(gk.unique())


def _check(name: str, metric: str, expected, actual, tol, ok: bool) -> dict:
    return {
        "name": name, "metric": metric, "expected": _num(expected),
        "actual": _num(actual), "tol": tol, "pass": bool(ok),
    }


def _close(actual: float, expected: float, *, abs_tol: float = 0.0, rel_tol: float = 0.0):
    return math.isclose(actual, expected, abs_tol=abs_tol, rel_tol=rel_tol)


# ---------------------------------------------------------------------------
# synthetic ground truth
# ---------------------------------------------------------------------------


def _kin(df: pd.DataFrame, metrics_cfg: dict, frame_rate: float, **overrides) -> pd.DataFrame:
    cfg = KinematicsConfig.from_metrics_cfg(metrics_cfg, frame_rate)
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return add_kinematics(df, cfg)[0]


def _distance(df_kin: pd.DataFrame) -> float:
    _, per_match = physical.distance_covered(df_kin, provider="metrica")
    return float(per_match["distance_m"].sum())


def _sprints(df_kin: pd.DataFrame, metrics_cfg: dict, frame_rate: float) -> pd.DataFrame:
    phys = metrics_cfg["physical"]
    return physical.detect_sprints(
        df_kin,
        threshold_mps=phys["sprint_threshold_mps"],
        min_duration_s=phys["sprint_min_duration_s"],
        min_recovery_s=phys["sprint_min_recovery_s"],
        frame_rate=frame_rate,
    )


def _synthetic_kinematics(metrics_cfg: dict) -> list[dict]:
    fr = 25.0  # conftest generators' default rate; dt derives from it
    checks = []

    speed, duration = 5.0, 10.0
    line = _kin(straight_line_track(speed_mps=speed, duration_s=duration, frame_rate=fr),
                metrics_cfg, fr)
    d = _distance(line)
    checks.append(_check("straight_line_5mps_10s", "distance_m", speed * duration, d,
                         TOL_LINEAR_DISTANCE_M,
                         _close(d, speed * duration, abs_tol=TOL_LINEAR_DISTANCE_M)))
    max_dev = float(np.abs(line["speed"] - speed).max())
    checks.append(_check("straight_line_5mps_10s", "speed_mps_max_abs_error", 0.0, max_dev,
                         TOL_SPEED_MPS, max_dev <= TOL_SPEED_MPS))

    noisy = noisy_stationary_track(noise_sd_m=0.05, duration_s=10.0, frame_rate=fr, seed=0)
    naive = float(np.hypot(noisy["x_pitch"].diff(), noisy["y_pitch"].diff()).sum())
    widest = max(metrics_cfg["physical"]["sensitivity"]["savgol_window_frames_grid"])
    d_default = _distance(_kin(noisy, metrics_cfg, fr))
    d_widest = _distance(_kin(noisy, metrics_cfg, fr, savgol_window_frames=int(widest)))
    checks.append(_check("noisy_stationary_default_window", "distance_vs_naive_ratio",
                         0.0, d_default / naive, "< 0.2", d_default < naive / 5))
    checks.append(_check(f"noisy_stationary_window_{widest}", "distance_m", 0.0, d_widest,
                         TOL_NOISE_DISTANCE_M, d_widest < TOL_NOISE_DISTANCE_M))

    radius = 9.15
    circle = _kin(circular_track(radius_m=radius, frame_rate=fr), metrics_cfg, fr)
    d = _distance(circle)
    expected = 2 * math.pi * radius
    checks.append(_check("circle_r9.15_one_lap", "distance_m", expected, d, f"rel {TOL_REL}",
                         _close(d, expected, rel_tol=TOL_REL)))
    zones = physical.speed_zones(
        circle, zones_mps=metrics_cfg["physical"]["speed_zones_mps"], provider="metrica"
    )
    z = float(zones["distance_m"].sum())
    checks.append(_check("circle_r9.15_one_lap", "speed_zones_sum_equals_distance", d, z,
                         TOL_GEOMETRY, _close(z, d, abs_tol=TOL_GEOMETRY)))

    phys = metrics_cfg["physical"]
    kin = metrics_cfg["kinematics"]
    sprint_speed = (phys["sprint_threshold_mps"] + kin["max_speed_mps"]) / 2
    sprint = _kin(straight_line_track(speed_mps=sprint_speed, duration_s=duration, frame_rate=fr),
                  metrics_cfg, fr)
    n = len(_sprints(sprint, metrics_cfg, fr))
    checks.append(_check(f"straight_line_{sprint_speed:g}mps_10s", "n_sprints", 1, n, 0, n == 1))

    hsr_speed = (phys["hsr_threshold_mps"] + phys["sprint_threshold_mps"]) / 2
    hsr_track = _kin(straight_line_track(speed_mps=hsr_speed, duration_s=duration, frame_rate=fr),
                     metrics_cfg, fr)
    n = len(_sprints(hsr_track, metrics_cfg, fr))
    checks.append(_check(f"straight_line_{hsr_speed:g}mps_10s", "n_sprints", 0, n, 0, n == 0))
    hsr = float(physical.high_speed_running(
        hsr_track, threshold_mps=phys["hsr_threshold_mps"], provider="metrica"
    )["hsr_distance_m"].sum())
    d = _distance(hsr_track)
    checks.append(_check(f"straight_line_{hsr_speed:g}mps_10s", "hsr_distance_equals_distance",
                         d, hsr, f"rel {TOL_REL}", _close(hsr, d, rel_tol=TOL_REL)))

    # Clipping is the backstop behind glitch rejection, so test it with
    # rejection off (with it on, an 18 m/s track is all glitch steps).
    too_fast = kin["max_speed_mps"] * 1.5
    clipped = _kin(straight_line_track(speed_mps=too_fast, duration_s=duration, frame_rate=fr),
                   metrics_cfg, fr, glitch_speed_mps=math.inf)
    top = float(clipped["speed"].max())
    checks.append(_check(f"straight_line_{too_fast:g}mps_clipped", "top_speed_mps",
                         kin["max_speed_mps"], top, TOL_GEOMETRY,
                         _close(top, kin["max_speed_mps"], abs_tol=TOL_GEOMETRY)))

    spiked = straight_line_track(speed_mps=speed, duration_s=duration, frame_rate=fr)
    spiked.loc[spiked.index == len(spiked) // 2, "y_pitch"] += 5.0
    spiked = _kin(spiked, metrics_cfg, fr)
    top = float(spiked["speed"].max())
    checks.append(_check(f"straight_line_{speed:g}mps_with_5m_spike", "top_speed_mps",
                         speed, top, TOL_SPEED_MPS, _close(top, speed, abs_tol=TOL_SPEED_MPS)))
    return checks


def _synthetic_shape_space(metrics_cfg: dict) -> list[dict]:
    fbf = flat_back_four(x_line=FBF_X_LINE)
    out = shape.compute_all(fbf, metrics_cfg).set_index("team")
    checks = []
    for team in ("home", "away"):
        lh = float(out.loc[team, "line_height_m"])
        checks.append(_check(f"flat_back_four_{team}", "line_height_m", FBF_X_LINE, lh,
                             TOL_GEOMETRY, _close(lh, FBF_X_LINE, abs_tol=TOL_GEOMETRY)))
    home = out.loc["home"]
    for metric, expected in (("width_m", FBF_WIDTH_M), ("length_m", FBF_LENGTH_M),
                             ("hull_area_m2", FBF_HULL_M2), ("cx_m", FBF_CENTROID_X),
                             ("cy_m", 0.0)):
        v = float(home[metric])
        checks.append(_check("flat_back_four_home", metric, expected, v, TOL_GEOMETRY,
                             _close(v, expected, abs_tol=TOL_GEOMETRY)))

    team_frame, _ = space.team_space_control(fbf, metrics_cfg, with_cells=False)
    pitch_area = metrics_cfg["pitch"]["length_m"] * metrics_cfg["pitch"]["width_m"]
    total = float(team_frame["area_m2"].sum())
    checks.append(_check("flat_back_four_voronoi", "total_area_m2", pitch_area, total,
                         TOL_GEOMETRY, _close(total, pitch_area, abs_tol=TOL_GEOMETRY)))
    # The formation is a point mirror of itself, so each team owns exactly half.
    home_share = float(team_frame.loc[team_frame["team"].astype(str) == "home",
                                      "area_share"].iloc[0])
    checks.append(_check("flat_back_four_voronoi", "home_area_share", 0.5, home_share,
                         TOL_GEOMETRY, _close(home_share, 0.5, abs_tol=TOL_GEOMETRY)))
    return checks


def _synthetic_refusal(metrics_cfg: dict) -> list[dict]:
    df = _kin(straight_line_track(), metrics_cfg, 25.0)
    checks = []
    for fn_name, call in (
        ("distance_covered", lambda: physical.distance_covered(df, provider="skillcorner")),
        ("high_speed_running", lambda: physical.high_speed_running(
            df, threshold_mps=metrics_cfg["physical"]["hsr_threshold_mps"],
            provider="SkillCorner")),
        ("speed_zones", lambda: physical.speed_zones(
            df, zones_mps=metrics_cfg["physical"]["speed_zones_mps"], provider="skillcorner")),
    ):
        checks.append(_check("skillcorner_refusal", f"{fn_name}_raises", 1,
                             int(_raises(call, ValueError)), 0, _raises(call, ValueError)))
    return checks


def _raises(call: Callable[[], object], exc: type[Exception]) -> bool:
    try:
        call()
    except exc:
        return True
    return False


def synthetic_checks(metrics_cfg: dict) -> list[dict]:
    return [
        *_synthetic_kinematics(metrics_cfg),
        *_synthetic_shape_space(metrics_cfg),
        *_synthetic_refusal(metrics_cfg),
    ]


# ---------------------------------------------------------------------------
# physiological ranges on real data
# ---------------------------------------------------------------------------


def _range_check(match: str, metric: str, player: str | None, value: float,
                 low: float | None, high: float | None) -> dict:
    ok = (low is None or value >= low) and (high is None or value <= high)
    return {"match": match, "metric": metric, "player": player, "value": _num(value),
            "low": low, "high": high, "pass": bool(ok)}


def _tracking_cols() -> list[str]:
    return ["track_id", "team", "period", "frame_id", "is_ball", "is_gk", "speed"]


def physiological_checks(
    metrics_cfg: dict, cache_dir: Path
) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns ``(checks, excluded, not_checked)``."""
    ranges = metrics_cfg["physiological_ranges"]
    d_lo, d_hi = ranges["total_distance_km"]
    s_lo, s_hi = ranges["top_speed_mps"]
    h_lo, h_hi = ranges["hsr_share_of_distance_pct"]
    compact = ranges.get("team_compactness_m2")

    checks: list[dict] = []
    excluded: list[dict] = []
    not_checked: list[dict] = []
    for provider, match_id in cached_matches(cache_dir):
        match = f"{provider}/{match_id}"
        if provider in physical.REFUSED_PROVIDERS:
            note = _meta(cache_dir, provider, match_id).get("physical_note", "")
            for metric in ("total_distance_km", "top_speed_mps", "hsr_share_of_distance_pct"):
                excluded.append({"match": match, "metric": metric, "reason": note})
            continue

        tracking = _read(cache_dir, provider, match_id, "tracking", columns=_tracking_cols())
        full = full_match_players(tracking)
        gks = goalkeepers(tracking)
        phys_df = _read(cache_dir, provider, match_id, "physical")
        phys_df["track_id"] = phys_df["track_id"].astype(str)
        top = (
            tracking.loc[~tracking["is_ball"].astype(bool)]
            .assign(track_id=lambda d: d["track_id"].astype(str))
            .groupby("track_id")["speed"].max()
        )

        for row in phys_df.sort_values("track_id").itertuples(index=False):
            tid = row.track_id
            if tid in gks:
                if tid in top.index and np.isfinite(top[tid]):
                    checks.append(_range_check(match, "gk_top_speed_mps", tid, top[tid],
                                               None, s_hi))
                continue
            if tid not in full:
                continue
            km = row.distance_m / 1000.0
            checks.append(_range_check(match, "total_distance_km", tid, km, d_lo, d_hi))
            checks.append(_range_check(match, "top_speed_mps", tid, top.get(tid, np.nan),
                                       s_lo, s_hi))
            share = 100.0 * row.hsr_distance_m / row.distance_m if row.distance_m else np.nan
            checks.append(_range_check(match, "hsr_share_of_distance_pct", tid, share,
                                       h_lo, h_hi))

        shape_df = _read(cache_dir, provider, match_id, "shape")
        medians = shape_df.groupby("team")["hull_area_m2"].median()
        for team, med in medians.items():
            if compact is not None:
                checks.append(_range_check(match, "team_compactness_median_m2", str(team),
                                           med, compact[0], compact[1]))
            else:
                not_checked.append({
                    "match": match, "metric": "team_compactness_median_m2",
                    "team": str(team), "value": _num(med),
                    "reason": "no physiological_ranges.team_compactness_m2 in config",
                })
    ceiling = float(metrics_cfg["kinematics"]["max_speed_mps"])
    for c in checks:
        # A NaN value can never pass, so a missing top speed shows up as a failure.
        if c["value"] is None:
            c["pass"] = False
        elif c["metric"].endswith("top_speed_mps"):
            # Speed pinned at the clip ceiling = a tracking glitch, not a real sprint.
            c["at_clip_ceiling"] = math.isclose(c["value"], ceiling)
    return checks, excluded, not_checked


def coverage_summary(cache_dir: Path) -> list[dict]:
    rows = []
    for provider, match_id in cached_matches(cache_dir):
        cov = _read(cache_dir, provider, match_id, "coverage")
        rows.append({
            "provider": provider, "match_id": match_id, "n_players": int(len(cov)),
            "mean_coverage_pct": _num(cov["coverage_pct"].mean()),
            "min_coverage_pct": _num(cov["coverage_pct"].min()),
        })
    return rows


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _pick(matches: list[tuple[str, str]], provider: str,
          preferred: str | None = None) -> str | None:
    ids = [m for p, m in matches if p == provider]
    if preferred is not None and preferred in ids:
        return preferred
    return ids[0] if ids else None


def validation_figures(cache_dir: Path, figures_dir: Path) -> list[str]:
    matches = cached_matches(cache_dir)
    written: list[str] = []
    metrica = _pick(matches, "metrica")
    skillcorner = _pick(matches, "skillcorner")

    cov_parts = [
        _read(cache_dir, p, m, "coverage")
        for p, m in (("metrica", metrica), ("skillcorner", skillcorner)) if m is not None
    ]
    if cov_parts:
        fig = viz.coverage_bar(pd.concat(cov_parts, ignore_index=True))
        written.append(str(viz.save_figure(fig, figures_dir / "coverage_by_provider.png")))

    if metrica is None:
        return written
    shape_df = _read(cache_dir, "metrica", metrica, "shape")
    if len(shape_df):
        t0 = float(shape_df.loc[shape_df["period"] == shape_df["period"].min(),
                                "timestamp"].min())
        # First ten minutes; figure is a still for the README.
        fig = viz.time_series(shape_df, columns=["line_height_m", "width_m", "hull_area_m2"],
                              window=(t0, t0 + 600.0))
        written.append(str(viz.save_figure(fig, figures_dir / "shape_timeseries.png")))

    tracking = _read(cache_dir, "metrica", metrica, "tracking")
    space_player = _read(cache_dir, "metrica", metrica, "space_player")
    source = space_player if len(space_player) else tracking[tracking["ball_state"] == "alive"]
    keys = source[["period", "frame_id"]].drop_duplicates().reset_index(drop=True)
    period, frame_id = (int(v) for v in keys.iloc[len(keys) // 2])
    cells = None
    if len(space_player):
        cells = space_player[(space_player["period"] == period)
                             & (space_player["frame_id"] == frame_id)]
    frame = tracking[(tracking["period"] == period) & (tracking["frame_id"] == frame_id)]
    fig = viz.pitch_snapshot(frame, voronoi_frame=cells,
                             title=f"Metrica {metrica} - period {period}, frame {frame_id}")
    written.append(str(viz.save_figure(fig, figures_dir / "pitch_snapshot.png")))
    return written


# ---------------------------------------------------------------------------
# report writers
# ---------------------------------------------------------------------------


def write_validation_report(
    metrics_cfg: dict, cache_dir: Path, reports_dir: Path, *, figures: bool = True
) -> dict:
    """Run synthetic + physiological checks, write ``validation.json``,
    return the dict."""
    synthetic = synthetic_checks(metrics_cfg)
    physio, excluded, not_checked = physiological_checks(metrics_cfg, cache_dir)
    all_checks = [*synthetic, *physio]
    failures = [
        {k: c.get(k) for k in ("name", "match", "metric", "player", "value", "actual",
                               "at_clip_ceiling")
         if c.get(k) is not None}
        for c in all_checks if not c["pass"]
    ]
    n_pass = sum(c["pass"] for c in all_checks)
    report = {
        "generated_at": _now(),
        "config_hash": config_hash(metrics_cfg),
        "full_match_rule": "outfield players present in the first and last frame of "
                           "every period; substitutes are not range-checked",
        "synthetic": synthetic,
        "physiological": physio,
        "excluded": excluded,
        "not_checked": not_checked,
        "coverage": coverage_summary(cache_dir),
        "references": REFERENCES,
        "summary": {
            "n_checks": len(all_checks),
            "n_pass": int(n_pass),
            "n_fail": len(all_checks) - int(n_pass),
            "synthetic_all_pass": all(c["pass"] for c in synthetic),
            "physiological_all_pass": all(c["pass"] for c in physio),
            "all_pass": bool(all_checks) and n_pass == len(all_checks),
            "n_top_speed_at_clip_ceiling": sum(bool(c.get("at_clip_ceiling")) for c in physio),
            "failures": failures,
        },
    }
    if figures:
        report["figures"] = validation_figures(cache_dir, Path(reports_dir) / "figures")
    _write_json(report, Path(reports_dir) / "validation.json")
    return report


def _spread(values: pd.Series, reference: float) -> float | None:
    if not reference:
        return None
    return float((values.max() - values.min()) / reference * 100.0)


def write_sensitivity_report(
    metrics_cfg: dict,
    cache_dir: Path,
    reports_dir: Path,
    *,
    match_id: str | None = None,
    figures: bool = True,
) -> dict:
    """Run the parameter sweep, write ``sensitivity.json``, return the dict."""
    matches = cached_matches(cache_dir)
    if match_id is not None and ("metrica", str(match_id)) not in matches:
        raise FileNotFoundError(f"metrica/{match_id} is not in {cache_dir}")
    mid = _pick(matches, "metrica", match_id)
    if mid is None:
        raise FileNotFoundError(f"no cached Metrica match in {cache_dir}; run build_cache first")

    phys = metrics_cfg["physical"]
    grid_cfg = phys["sensitivity"]
    windows = [int(w) for w in grid_cfg["savgol_window_frames_grid"]]
    thresholds = [float(t) for t in grid_cfg["sprint_threshold_mps_grid"]]
    # Native rate, not the cache rate: the pipeline now smooths BEFORE
    # downsampling (DD-030), so sweeping the window at the cached 5 Hz
    # rate would test a knob the pipeline doesn't actually turn anymore.
    frame_rate = float(_meta(cache_dir, "metrica", mid)["source_frame_rate_hz"])

    base_df = get_loader("metrica").load(mid)
    full = full_match_players(base_df)
    gks = goalkeepers(base_df)
    outfield_full = full - gks

    base_kin = KinematicsConfig.from_metrics_cfg(metrics_cfg, frame_rate)
    by_window: dict[int, pd.DataFrame] = {}
    distance_rows = []
    for w in windows:
        df_kin, clip = add_kinematics(
            base_df, dataclasses.replace(base_kin, savgol_window_frames=w)
        )
        by_window[w] = df_kin
        _, dist = physical.distance_covered(df_kin, provider="metrica")
        hsr = physical.high_speed_running(
            df_kin, threshold_mps=phys["hsr_threshold_mps"], provider="metrica"
        )
        merged = dist.merge(hsr, on=["track_id", "team"])
        merged = merged[merged["track_id"].astype(str).isin(outfield_full)]
        distance_rows.append({
            "window": w,
            "n_players": int(len(merged)),
            "mean_km": _num(merged["distance_m"].mean() / 1000.0),
            "mean_hsr_km": _num(merged["hsr_distance_m"].mean() / 1000.0),
            "n_clipped": int(clip.n_clipped),
        })

    grid = physical.count_sprints_grid(
        by_window,
        threshold_grid_mps=thresholds,
        min_duration_s=phys["sprint_min_duration_s"],
        min_recovery_s=phys["sprint_min_recovery_s"],
        frame_rate=frame_rate,
    )
    # Mid-grid reference (the spec's rule, and what viz.sensitivity_heatmap titles
    # with), so the figure and this file quote the same N%.
    ref_w = sorted(windows)[(len(windows) - 1) // 2]
    ref_t = sorted(thresholds)[(len(thresholds) - 1) // 2]
    spread = physical.sprint_spread_pct(grid)
    base = (int(metrics_cfg["kinematics"]["savgol_window_frames"]),
            float(phys["sprint_threshold_mps"]))
    vs_default = (physical.sprint_spread_pct(grid, baseline=base)
                  if base[0] in windows and base[1] in thresholds else None)
    ref_total = float(grid.loc[(grid["savgol_window_frames"] == ref_w)
                               & np.isclose(grid["sprint_threshold_mps"], ref_t),
                               "total_sprints"].iloc[0])
    thr_only = _spread(grid.loc[grid["savgol_window_frames"] == ref_w, "total_sprints"],
                       ref_total)
    win_only = _spread(grid.loc[np.isclose(grid["sprint_threshold_mps"], ref_t),
                                "total_sprints"], ref_total)
    dist_df = pd.DataFrame(distance_rows).set_index("window")
    ref_km = dist_df.loc[ref_w, "mean_km"]
    dist_spread = None if pd.isna(ref_km) else _spread(dist_df["mean_km"].astype(float),
                                                        float(ref_km))

    sentence = (
        f"Sprint count varies by {spread:.0f}% across defensible parameter choices "
        f"(Savitzky-Golay window {min(windows)}-{max(windows)} frames at {frame_rate:g} Hz, "
        f"sprint threshold {min(thresholds):g}-{max(thresholds):g} m/s; "
        f"{thr_only:.0f}% from the threshold alone), so I report the parameters "
        "alongside the number."
    )
    report = {
        "generated_at": _now(),
        "config_hash": config_hash(metrics_cfg),
        "match": f"metrica/{mid}",
        "frame_rate_hz": frame_rate,
        "grids": {"savgol_window_frames": windows, "sprint_threshold_mps": thresholds},
        "fixed_params": {
            "savgol_polyorder": int(metrics_cfg["kinematics"]["savgol_polyorder"]),
            "sprint_min_duration_s": phys["sprint_min_duration_s"],
            "sprint_min_recovery_s": phys["sprint_min_recovery_s"],
            "hsr_threshold_mps": phys["hsr_threshold_mps"],
        },
        "sprint_count": [
            {"window": int(r.savgol_window_frames), "threshold": float(r.sprint_threshold_mps),
             "total": int(r.total_sprints), "per_player_mean": _num(r.sprints_per_player_mean)}
            for r in grid.itertuples(index=False)
        ],
        "distance_vs_window": distance_rows,
        "headline": {
            "metric": "sprint_count",
            "spread_pct": _num(spread),
            "threshold_only_spread_pct": _num(thr_only),
            "window_only_spread_pct": _num(win_only),
            "distance_spread_pct": _num(dist_spread),
            "spread_pct_vs_config_default": _num(vs_default),
            "config_default_params": {"savgol_window_frames": base[0],
                                      "sprint_threshold_mps": base[1]},
            "baseline_params": {"savgol_window_frames": int(ref_w),
                                "sprint_threshold_mps": float(ref_t),
                                "total_sprints": int(ref_total)},
            "sentence": sentence,
        },
    }
    if figures:
        fig = viz.sensitivity_heatmap(grid)
        path = viz.save_figure(fig, Path(reports_dir) / "figures" / "sensitivity_heatmap.png")
        report["figures"] = [str(path)]
    _write_json(report, Path(reports_dir) / "sensitivity.json")
    return report


def main(argv: list[str] | None = None) -> int:
    """Parse args, build both reports, print the headline sentence,
    return an exit code (non-zero if any check failed)."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/metrics.yaml"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--reports-dir", type=Path, default=Path("reports"))
    p.add_argument("--sensitivity-match", default=None, help="Metrica match id to sweep")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    if not cached_matches(args.cache_dir):
        print(f"no cached matches in {args.cache_dir}; run scripts.build_cache first",
              file=sys.stderr)
        return 1
    metrics_cfg = pipeline.load_config(args.config)
    figures = not args.no_figures

    validation = write_validation_report(metrics_cfg, args.cache_dir, args.reports_dir,
                                         figures=figures)
    summary = validation["summary"]
    print(f"validation: {summary['n_pass']}/{summary['n_checks']} checks pass "
          f"(synthetic_all_pass={summary['synthetic_all_pass']}, "
          f"physiological_all_pass={summary['physiological_all_pass']})")
    for f in summary["failures"]:
        print(f"  FAIL {f}")

    sensitivity = write_sensitivity_report(metrics_cfg, args.cache_dir, args.reports_dir,
                                           match_id=args.sensitivity_match, figures=figures)
    print(sensitivity["headline"]["sentence"])
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
