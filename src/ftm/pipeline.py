"""match_id -> tidy metrics parquet.

Orchestrates: loader -> validate -> kinematics -> every metric -> tidy
frames written to ``data/cache/``. The Streamlit app reads those parquet
files and computes NOTHING, so this module is where all the cost lives.

Config resolution happens here: read ``configs/metrics.yaml`` once, pass
the resolved dict down. Nothing below this layer opens the YAML.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yaml

from ftm.kinematics import KinematicsConfig, add_kinematics
from ftm.loaders import get_loader
from ftm.metrics import pressing, shape, space
from ftm.metrics.physical import (
    REFUSED_PROVIDERS,
    detect_sprints,
    distance_covered,
    high_speed_running,
)

_REQUIRED_KEYS: dict[str, list[str]] = {
    "pitch": ["length_m", "width_m"],
    "kinematics": ["savgol_window_frames", "savgol_polyorder", "max_speed_mps", "max_gap_s"],
    "shape": ["line_height_n_deepest", "min_outfield_players"],
    "physical": [
        "hsr_threshold_mps",
        "sprint_threshold_mps",
        "sprint_min_duration_s",
        "sprint_min_recovery_s",
        "speed_zones_mps",
    ],
    "pressing": ["press_radius_m", "carrier_search_radius_m"],
    "space": ["include_gk"],
}


@dataclass(frozen=True)
class PipelineConfig:
    """Everything ``run_match`` needs beyond the raw match.

    metrics_yaml_path   Path to ``configs/metrics.yaml``.
    cache_dir           Output dir (``data/cache``).
    target_hz           Downsample rate for the cache (e.g. 5.0). Full
                        25 Hz cache is too big for HF Spaces; the README
                        must state the chosen rate.
    include_space       Voronoi is slow — allow skipping it in dev runs.
    """

    metrics_yaml_path: Path
    cache_dir: Path
    target_hz: float = 5.0
    include_space: bool = True


@dataclass(frozen=True)
class MatchArtifacts:
    """Paths written for one match, so callers/tests can find them.

    tracking      Downsampled canonical frame + kinematics columns.
    shape         Per-(frame, team) shape metrics.
    pressing      Per-frame pressing metrics.
    space         Per-(frame, team) team space control.
    space_player  Per-(frame, player) space control detail, for the
                  pitch-snapshot overlay. Empty when ``include_space`` is
                  False.
    physical      Per-player physical metrics (empty/flagged for
                  SkillCorner).
    coverage      Per-player coverage % for the match.
    meta          JSON dump of ``MatchMeta`` + config + clip report.
    """

    tracking: Path
    shape: Path
    pressing: Path
    space: Path
    space_player: Path
    physical: Path
    coverage: Path
    meta: Path


def load_config(path: Path) -> dict:
    """Parse ``configs/metrics.yaml`` into a plain nested dict.

    Validate presence of the keys the metrics need (kinematics.*,
    shape.*, physical.*, pressing.*, space.*); raise ``KeyError`` with a
    helpful message if one is missing. This is the ONLY place the YAML is
    read.
    """
    with Path(path).open() as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise KeyError(f"{path}: expected a top-level mapping, got {type(cfg).__name__}")
    for section, keys in _REQUIRED_KEYS.items():
        if section not in cfg:
            raise KeyError(f"{path}: missing required section {section!r}")
        missing = [k for k in keys if k not in cfg[section]]
        if missing:
            raise KeyError(f"{path}: section {section!r} missing keys {missing}")
    return cfg


def coverage_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-player frame coverage for a match.

    For each (track_id, period): frames_present / total_frames_in_period,
    where ``total_frames_in_period`` is the number of DISTINCT frame_ids
    seen by ANY track in that period (the match's actual frame grid), not
    an assumption from frame_rate x duration. Roll up to a per-player
    match coverage (frames present across all periods / frames in the
    match). This feeds ``reports/validation.json`` (the Metrica-vs-
    SkillCorner gap) and the dashboard's per-player table, where coverage
    % sits next to every physical number.

    Returns
    -------
    ``(per_period, per_match)``:
      * ``per_period``: (track_id, team, period, coverage_pct)
      * ``per_match``: (track_id, team, coverage_pct)

    Ball rows are excluded — coverage is a tracking-quality metric for
    players, not the ball.
    """
    players = df.loc[~df["is_ball"].astype(bool)]
    frames_per_period = players.groupby("period")["frame_id"].nunique()

    present = (
        players.groupby(["track_id", "team", "period"], observed=True)["frame_id"]
        .nunique()
        .rename("frames_present")
        .reset_index()
    )
    present["frames_total"] = present["period"].map(frames_per_period)
    present["coverage_pct"] = 100.0 * present["frames_present"] / present["frames_total"]
    per_period = present[["track_id", "team", "period", "coverage_pct"]]

    total_frames = int(frames_per_period.sum())
    per_match = (
        present.groupby(["track_id", "team"], observed=True)
        .agg(frames_present=("frames_present", "sum"))
        .reset_index()
    )
    per_match["coverage_pct"] = (
        100.0 * per_match["frames_present"] / total_frames if total_frames else 0.0
    )
    per_match = per_match[["track_id", "team", "coverage_pct"]]
    return per_period, per_match


def downsample(df: pd.DataFrame, *, from_hz: float, to_hz: float) -> pd.DataFrame:
    """Keep every Nth frame per period so the cache is small.

    ``N = round(from_hz / to_hz)``. Slice by frame index within each
    period (not by time rounding) to keep ``frame_id`` monotonic. Recompute
    nothing here — downsample BEFORE kinematics so smoothing windows are
    defined in terms of the cached rate, and record both rates in
    ``MatchArtifacts.meta``.
    """
    if to_hz <= 0 or from_hz <= 0:
        raise ValueError(f"rates must be positive, got from_hz={from_hz}, to_hz={to_hz}")
    if to_hz > from_hz:
        raise ValueError(f"cannot upsample: to_hz={to_hz} > from_hz={from_hz}")
    n = round(from_hz / to_hz)
    if n <= 1:
        return df.reset_index(drop=True)

    keep_ids = (
        df.loc[~df["is_ball"].astype(bool)]
        .groupby("period")["frame_id"]
        .apply(lambda s: pd.Index(sorted(s.unique()))[::n])
    )
    keep = set()
    for period, ids in keep_ids.items():
        keep.update((period, int(fid)) for fid in ids)

    mask = [(p, f) in keep for p, f in zip(df["period"], df["frame_id"], strict=True)]
    return df.loc[mask].reset_index(drop=True)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def run_match(
    provider: str,
    match_id: str,
    cfg: PipelineConfig,
) -> MatchArtifacts:
    """Full pipeline for one match. Writes parquet, returns the paths.

    Steps:
    1. ``loader = ftm.loaders.get_loader(provider)``.
    2. ``meta = loader.load_meta(match_id)`` ; ``df = loader.load(match_id)``
       (``df`` is already schema-valid).
    3. ``df = downsample(df, from_hz=meta.frame_rate, to_hz=cfg.target_hz)``.
    4. Build ``KinematicsConfig`` from the YAML + ``meta.frame_rate``
       (use the DOWNSAMPLED rate), then
       ``df, clip = ftm.kinematics.add_kinematics(df, kin_cfg)``.
    5. ``shape.compute_all`` / ``pressing.compute_all`` /
       (``space.team_space_control`` if ``cfg.include_space``).
    6. Physical metrics: run them for Metrica; for SkillCorner produce
       the flagged/empty physical frame and rely on ``coverage_table``.
    7. ``coverage_table(df)``.
    8. Write each tidy frame to ``cfg.cache_dir`` as
       ``<provider>__<match_id>__<name>.parquet``; dump ``meta`` JSON
       including ``clip`` and the resolved config.

    Idempotent: re-running overwrites cleanly.
    """
    metrics_cfg = load_config(cfg.metrics_yaml_path)
    loader = get_loader(provider)
    meta = loader.load_meta(match_id)
    df = loader.load(match_id)

    effective_hz = min(cfg.target_hz, meta.frame_rate)
    df = downsample(df, from_hz=meta.frame_rate, to_hz=effective_hz)

    kin_cfg = KinematicsConfig.from_metrics_cfg(metrics_cfg, effective_hz)
    df, clip = add_kinematics(df, kin_cfg)

    shape_df = shape.compute_all(df, metrics_cfg)
    pressing_df = pressing.compute_all(df, metrics_cfg)

    if cfg.include_space:
        space_team, space_player = space.team_space_control(df, metrics_cfg)
    else:
        space_team, space_player = (
            pd.DataFrame(columns=space.TEAM_FRAME_COLUMNS),
            pd.DataFrame(columns=space.PLAYER_FRAME_COLUMNS),
        )

    provider_key = provider.strip().lower()
    if provider_key in REFUSED_PROVIDERS:
        physical_df = pd.DataFrame(
            columns=["track_id", "team", "distance_m", "hsr_distance_m", "n_sprints"]
        )  # provider/match_id added below, same as the non-empty branch
        physical_note = (
            f"physical (distance/HSR/sprints) not reported for provider={provider_key!r}: "
            "broadcast tracking is not comparable to full-pitch optical tracking."
        )
    else:
        physical_cfg = metrics_cfg["physical"]
        _, per_match_dist = distance_covered(df, provider=provider_key)
        hsr = high_speed_running(
            df, threshold_mps=physical_cfg["hsr_threshold_mps"], provider=provider_key
        )
        sprints = detect_sprints(
            df,
            threshold_mps=physical_cfg["sprint_threshold_mps"],
            min_duration_s=physical_cfg["sprint_min_duration_s"],
            min_recovery_s=physical_cfg["sprint_min_recovery_s"],
            frame_rate=effective_hz,
        )
        n_sprints = (
            sprints.groupby(["track_id", "team"], observed=True, sort=True)
            .size()
            .rename("n_sprints")
            .reset_index()
        )
        physical_df = (
            per_match_dist.merge(hsr, on=["track_id", "team"], how="left")
            .merge(n_sprints, on=["track_id", "team"], how="left")
        )
        physical_df["hsr_distance_m"] = physical_df["hsr_distance_m"].fillna(0.0)
        physical_df["n_sprints"] = physical_df["n_sprints"].fillna(0).astype(int)
        physical_note = ""

    _, coverage_per_match = coverage_table(df)
    # provider/match_id are only implicit in the filename otherwise; a report or
    # dashboard combining coverage/physical across many cached matches needs them
    # as real columns, not something parsed back out of a path.
    coverage_per_match = coverage_per_match.assign(provider=provider_key, match_id=str(match_id))
    physical_df = physical_df.assign(provider=provider_key, match_id=str(match_id))

    prefix = f"{provider_key}__{match_id}__"
    paths = MatchArtifacts(
        tracking=cfg.cache_dir / f"{prefix}tracking.parquet",
        shape=cfg.cache_dir / f"{prefix}shape.parquet",
        pressing=cfg.cache_dir / f"{prefix}pressing.parquet",
        space=cfg.cache_dir / f"{prefix}space.parquet",
        space_player=cfg.cache_dir / f"{prefix}space_player.parquet",
        physical=cfg.cache_dir / f"{prefix}physical.parquet",
        coverage=cfg.cache_dir / f"{prefix}coverage.parquet",
        meta=cfg.cache_dir / f"{prefix}meta.json",
    )
    _write_parquet(df, paths.tracking)
    _write_parquet(shape_df, paths.shape)
    _write_parquet(pressing_df, paths.pressing)
    _write_parquet(space_team, paths.space)
    _write_parquet(space_player, paths.space_player)
    _write_parquet(physical_df, paths.physical)
    _write_parquet(coverage_per_match, paths.coverage)

    meta_dict = {
        "match_meta": {**asdict(meta), "periods": {str(k): v for k, v in meta.periods.items()}},
        "source_frame_rate_hz": meta.frame_rate,
        "cached_frame_rate_hz": effective_hz,
        "target_hz_requested": cfg.target_hz,
        "clip_report": asdict(clip),
        "physical_note": physical_note,
        "include_space": cfg.include_space,
    }
    paths.meta.parent.mkdir(parents=True, exist_ok=True)
    paths.meta.write_text(json.dumps(meta_dict, indent=2, default=str))
    return paths


def run_all(cfg: PipelineConfig) -> list[MatchArtifacts]:
    """Run ``run_match`` for every match from every provider.

    Iterates ``get_loader(p).list_matches()`` for p in
    {"metrica", "skillcorner"}. Used by ``scripts/build_cache.py``.
    Collect and re-raise errors with the offending (provider, match_id)
    attached so a single bad match is easy to find.
    """
    results: list[MatchArtifacts] = []
    errors: list[str] = []
    for provider in ("metrica", "skillcorner"):
        loader = get_loader(provider)
        for match_id in loader.list_matches():
            try:
                results.append(run_match(provider, match_id, cfg))
            except Exception as exc:  # noqa: BLE001 - re-raised with context below
                errors.append(f"{provider}/{match_id}: {exc!r}")
    if errors:
        raise RuntimeError("run_all failed for:\n" + "\n".join(errors))
    return results
