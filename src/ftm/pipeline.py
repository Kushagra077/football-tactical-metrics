"""match_id -> tidy metrics parquet.

Orchestrates: loader -> validate -> kinematics -> every metric -> tidy
frames written to ``data/cache/``. The Streamlit app reads those parquet
files and computes NOTHING, so this module is where all the cost lives.

Config resolution happens here: read ``configs/metrics.yaml`` once, pass
the resolved dict down. Nothing below this layer opens the YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


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
    space         Per-frame team space control (+ per-player detail).
    physical      Per-player physical metrics (empty/flagged for
                  SkillCorner).
    coverage      Per-player coverage % for the match.
    meta          JSON dump of ``MatchMeta`` + config + clip report.
    """

    tracking: Path
    shape: Path
    pressing: Path
    space: Path
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
    raise NotImplementedError


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-player frame coverage for a match.

    For each (track_id, period): frames_present / total_frames_in_period.
    Roll up to a per-player match coverage. This feeds
    ``reports/validation.json`` (the Metrica-vs-SkillCorner gap) and the
    dashboard's per-player table, where coverage % sits next to every
    physical number.

    Returns: (track_id, team, period, coverage_pct) and a per-player roll-up.
    """
    raise NotImplementedError


def downsample(df: pd.DataFrame, *, from_hz: float, to_hz: float) -> pd.DataFrame:
    """Keep every Nth frame per period so the cache is small.

    ``N = round(from_hz / to_hz)``. Slice by frame index within each
    period (not by time rounding) to keep ``frame_id`` monotonic. Recompute
    nothing here — downsample BEFORE kinematics so smoothing windows are
    defined in terms of the cached rate, and record both rates in
    ``MatchArtifacts.meta``.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def run_all(cfg: PipelineConfig) -> list[MatchArtifacts]:
    """Run ``run_match`` for every match from every provider.

    Iterates ``get_loader(p).list_matches()`` for p in
    {"metrica", "skillcorner"}. Used by ``scripts/build_cache.py``.
    Collect and re-raise errors with the offending (provider, match_id)
    attached so a single bad match is easy to find.
    """
    raise NotImplementedError
