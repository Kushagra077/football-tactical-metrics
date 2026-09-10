"""Precompute every match into tidy parquet under ``data/cache/``.

Run once locally (and in the HF Space build, or ship the parquet). The
Streamlit app then only ever READS these files.

    python -m scripts.build_cache --config configs/metrics.yaml \
        --cache-dir data/cache --target-hz 5

CLI outline (use argparse):
  --config       path to metrics.yaml           (default configs/metrics.yaml)
  --cache-dir    output dir                      (default data/cache)
  --target-hz    downsample rate for the cache  (default 5.0)
  --provider     optional filter: metrica|skillcorner (default: both)
  --match-id     optional single match to (re)build
  --no-space     skip Voronoi for a fast dev run

Behaviour:
  1. Build ``PipelineConfig`` from the args.
  2. If --provider/--match-id given, call ``pipeline.run_match`` once;
     else ``pipeline.run_all``.
  3. Print a summary table: provider, match_id, rows, #players, clip
     fraction, output size on disk.
  4. Assert the total cache size is under a few hundred MB; warn loudly
     if not and suggest a lower --target-hz (the README must state
     whichever rate you settle on).
  5. Non-zero exit if any match failed.
"""

from __future__ import annotations


def main() -> int:
    """Parse args, run the pipeline, print the summary, return an exit code."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
