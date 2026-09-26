"""Precompute every match into tidy parquet under ``data/cache/``.

Run once locally. The Streamlit app then only ever READS these files.

    python -m scripts.build_cache --config configs/metrics.yaml \
        --cache-dir data/cache --target-hz 5

CLI outline (use argparse):
  --config       path to metrics.yaml           (default configs/metrics.yaml)
  --cache-dir    output dir                      (default data/cache)
  --target-hz    downsample rate for the cache  (default: ``cache.target_hz``
                                                 from the YAML)
  --provider     optional filter: metrica|skillcorner (default: both)
  --match-id     optional single match to (re)build (requires --provider)
  --no-space     skip Voronoi for a fast dev run
  --max-cache-mb size budget for the whole cache dir (default 300)

Behaviour:
  1. Build ``PipelineConfig`` from the args.
  2. If --provider/--match-id given, call ``pipeline.run_match`` per
     selected match; else ``pipeline.run_all``.
  3. Print a summary table: provider, match_id, rows, #players, clip
     fraction, output size on disk.
  4. Check the total cache size is under ``--max-cache-mb``; warn loudly
     if not and suggest a lower --target-hz (the README must state
     whichever rate you settle on).
  5. Exit codes: 0 ok, 1 any match failed, 2 cache over budget.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ftm import pipeline
from ftm.loaders import get_loader

PROVIDERS = ("metrica", "skillcorner")
DEFAULT_MAX_CACHE_MB = 300.0
EXIT_OK, EXIT_FAILED, EXIT_OVER_BUDGET = 0, 1, 2


@dataclass(frozen=True)
class MatchSummary:
    provider: str
    match_id: str
    rows: int
    n_players: int
    clip_fraction: float
    size_mb: float


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/metrics.yaml"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--target-hz", type=float, default=None)
    p.add_argument("--provider", choices=PROVIDERS, default=None)
    p.add_argument("--match-id", default=None)
    p.add_argument("--no-space", action="store_true")
    p.add_argument("--max-cache-mb", type=float, default=DEFAULT_MAX_CACHE_MB)
    args = p.parse_args(argv)
    if args.match_id is not None and args.provider is None:
        p.error("--match-id requires --provider")
    return args


def resolve_target_hz(cli_value: float | None, metrics_cfg: dict) -> float:
    """CLI wins; else ``cache.target_hz`` from the YAML."""
    if cli_value is not None:
        return float(cli_value)
    try:
        return float(metrics_cfg["cache"]["target_hz"])
    except KeyError as exc:
        raise KeyError("pass --target-hz or set cache.target_hz in metrics.yaml") from exc


def match_files(cache_dir: Path, provider: str, match_id: str) -> list[Path]:
    return sorted(Path(cache_dir).glob(f"{provider}__{match_id}__*"))


def summarize(artifacts: pipeline.MatchArtifacts) -> MatchSummary:
    """Read back what ``run_match`` wrote — the summary reflects the disk."""
    meta = json.loads(artifacts.meta.read_text())
    tracking = pd.read_parquet(artifacts.tracking, columns=["track_id", "is_ball"])
    provider, match_id = artifacts.meta.name.split("__")[:2]
    files = match_files(artifacts.meta.parent, provider, match_id)
    return MatchSummary(
        provider=provider,
        match_id=match_id,
        rows=len(tracking),
        n_players=int(tracking.loc[~tracking["is_ball"].astype(bool), "track_id"].nunique()),
        clip_fraction=float(meta["clip_report"]["fraction"]),
        size_mb=sum(f.stat().st_size for f in files) / 1e6,
    )


def format_table(summaries: list[MatchSummary]) -> str:
    header = f"{'provider':<12} {'match_id':<10} {'rows':>10} {'players':>8} " \
             f"{'clip_frac':>10} {'size_mb':>9}"
    lines = [header, "-" * len(header)]
    for s in summaries:
        lines.append(
            f"{s.provider:<12} {s.match_id:<10} {s.rows:>10,} {s.n_players:>8} "
            f"{s.clip_fraction:>10.5f} {s.size_mb:>9.2f}"
        )
    return "\n".join(lines)


def cache_size_mb(cache_dir: Path) -> float:
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return 0.0
    files = [*cache_dir.glob("*.parquet"), *cache_dir.glob("*.json")]
    return sum(f.stat().st_size for f in files) / 1e6


def _targets(provider: str, match_id: str | None) -> list[tuple[str, str]]:
    if match_id is not None:
        return [(provider, str(match_id))]
    return [(provider, str(m)) for m in get_loader(provider).list_matches()]


def build(
    cfg: pipeline.PipelineConfig, provider: str | None, match_id: str | None
) -> tuple[list[pipeline.MatchArtifacts], list[str]]:
    """Run the pipeline; return (artifacts, error strings). Never raises for a
    single bad match."""
    if provider is None:
        started = time.time()
        try:
            return pipeline.run_all(cfg), []
        except RuntimeError as exc:
            # run_all keeps going past bad matches, so this run's good ones are on
            # disk: keep metas written since we started, minus the failed keys.
            errors = str(exc).splitlines()[1:] or [str(exc)]
            failed = {e.split(":", 1)[0] for e in errors}
            done = [
                _artifacts_from_meta(m)
                for m in sorted(Path(cfg.cache_dir).glob("*__meta.json"))
                if m.stat().st_mtime >= started
                and "/".join(m.name.split("__")[:2]) not in failed
            ]
            return done, errors

    artifacts: list[pipeline.MatchArtifacts] = []
    errors: list[str] = []
    for prov, mid in _targets(provider, match_id):
        print(f"building {prov}/{mid} ...", flush=True)
        try:
            artifacts.append(pipeline.run_match(prov, mid, cfg))
        except Exception as exc:  # noqa: BLE001 - reported per match, exit code set below
            errors.append(f"{prov}/{mid}: {exc!r}")
    return artifacts, errors


def _artifacts_from_meta(meta_path: Path) -> pipeline.MatchArtifacts:
    prefix = meta_path.name[: -len("meta.json")]
    d = meta_path.parent
    return pipeline.MatchArtifacts(
        **{
            name: d / f"{prefix}{name}.parquet"
            for name in ("tracking", "shape", "pressing", "space", "space_player",
                         "physical", "coverage")
        },
        meta=meta_path,
    )


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the pipeline, print the summary, return an exit code."""
    args = _parse_args(argv)
    metrics_cfg = pipeline.load_config(args.config)
    cfg = pipeline.PipelineConfig(
        metrics_yaml_path=args.config,
        cache_dir=args.cache_dir,
        target_hz=resolve_target_hz(args.target_hz, metrics_cfg),
        include_space=not args.no_space,
    )
    print(f"cache_dir={cfg.cache_dir} target_hz={cfg.target_hz} include_space={cfg.include_space}")

    artifacts, errors = build(cfg, args.provider, args.match_id)
    summaries = []
    for a in artifacts:
        try:
            summaries.append(summarize(a))
        except (OSError, KeyError, ValueError) as exc:
            errors.append(f"{a.meta.name}: summary failed: {exc!r}")
    if summaries:
        print(format_table(summaries))

    total_mb = cache_size_mb(cfg.cache_dir)
    print(f"total cache size: {total_mb:.1f} MB (budget {args.max_cache_mb:.0f} MB)")

    code = EXIT_OK
    if errors:
        print(f"\n{len(errors)} match(es) FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        code = EXIT_FAILED
    if total_mb > args.max_cache_mb:
        print(
            f"\nWARNING: cache is {total_mb:.1f} MB, over the {args.max_cache_mb:.0f} MB budget. "
            f"Rebuild with a lower --target-hz (currently {cfg.target_hz:g}) and state the "
            "chosen rate in the README.",
            file=sys.stderr,
        )
        code = code or EXIT_OVER_BUDGET
    return code


if __name__ == "__main__":
    raise SystemExit(main())
