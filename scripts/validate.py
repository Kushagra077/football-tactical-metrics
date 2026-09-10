"""Write ``reports/validation.json`` and ``reports/sensitivity.json``.

These two committed files are half of B1's "definition of done". A1
committed latency numbers but not accuracy; do not repeat that.

    python -m scripts.validate --config configs/metrics.yaml \
        --cache-dir data/cache --reports-dir reports

Depends on the cache existing (run ``build_cache.py`` first).

--------------------------------------------------------------------------
reports/validation.json  — metrics vs. ground truth
--------------------------------------------------------------------------
Two kinds of check, both recorded with pass/fail + the actual number:

1. Synthetic ground truth. Reuse the trajectory generators from
   ``tests/test_kinematics.py`` (straight line, noisy stationary, circle)
   and the hand-placed formations from ``tests/test_metrics.py``. Run the
   real metric functions on them and assert the known answers. This
   proves the shipped code, not just the test doubles.

2. Published physiological ranges. For each Metrica full match:
     * total distance per player within 9–12 km
     * top speeds plausible (<= ~11 m/s outfield, keeper lower)
     * HSR share of total distance in a sane band
     * team compactness within a plausible m^2 range
   Cite the ranges' sources in a "references" field.

Shape of the file (suggested):
    {
      "generated_at": ...,
      "config_hash": ...,
      "synthetic": [ {name, metric, expected, actual, tol, pass} ... ],
      "physiological": [ {match, metric, player?, value, low, high, pass} ... ],
      "references": [ ... ],
      "summary": {"n_checks": .., "n_pass": .., "all_pass": bool}
    }

--------------------------------------------------------------------------
reports/sensitivity.json — how much each metric moves with parameters
--------------------------------------------------------------------------
Sweep the two knobs that analysts actually argue about:
  * smoothing window: ``kinematics.savgol_window_frames`` grid
  * sprint threshold: ``physical.sprint_threshold_mps`` grid
(optionally HSR threshold too).

For one representative Metrica match:
  1. For each window in the grid: re-run ``add_kinematics`` -> get a
     ``df_with_kin``. Collect into ``{window: df}``.
  2. ``metrics.physical.count_sprints_grid(...)`` over that map x the
     threshold grid.
  3. Also record how total distance and HSR distance move across the
     window grid.
  4. Compute the spread as a % of the mid-grid value — this is the "N" in
     *"sprint count varies by N% across defensible threshold choices, so
     I report the parameters alongside the number."*

Shape (suggested):
    {
      "generated_at": ..., "match": ...,
      "grids": {"savgol_window_frames": [...], "sprint_threshold_mps": [...]},
      "sprint_count": [ {window, threshold, total, per_player_mean} ... ],
      "distance_vs_window": [ {window, mean_km} ... ],
      "headline": {"metric": "sprint_count", "spread_pct": ...,
                   "baseline_params": {...}, "sentence": "..."},
    }

Both writers also drop their figures via ``ftm.viz`` into
``reports/figures/`` for the README.

Exit non-zero if any validation check fails (so CI can gate on it).
"""

from __future__ import annotations


def write_validation_report(*args, **kwargs) -> dict:
    """Run synthetic + physiological checks, write ``validation.json``,
    return the dict."""
    raise NotImplementedError


def write_sensitivity_report(*args, **kwargs) -> dict:
    """Run the parameter sweep, write ``sensitivity.json``, return the dict."""
    raise NotImplementedError


def main() -> int:
    """Parse args, build both reports, print the headline sentence,
    return an exit code (non-zero if any check failed)."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
