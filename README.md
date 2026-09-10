<!--
README build order (spec step 8) — TABLES BEFORE PROSE. Fill each section
as the corresponding pipeline step starts producing real numbers. Do not
write prose that the committed JSON reports don't back up.

  1. One Voronoi + positions still frame  (reports/figures/…)
  2. Metrics table — definitions + thresholds stated inline
  3. Validation table — synthetic ground truth + physiological ranges
  4. Sensitivity table — the honest result, give it room
  5. Coverage comparison — Metrica vs SkillCorner
  6. Schema contract — with the note that Track A output plugs in here
  7. Attribution for both data sources (Metrica has no license file —
     the credit is doing real work)
-->

# Football Tactical Metrics (B1)

Tracking data in, validated tactical + physical metrics out. Two providers,
one schema, one metrics layer that does not know or care which provider a
frame came from.

**Live dashboard:** _(HF Space link — Streamlit SDK, CPU Basic; free Spaces
sleep after inactivity)_

---

## 1 · Snapshot

_(embed `reports/figures/pitch_snapshot.png` — player positions + Voronoi
overlay for one frame)_

## 2 · Metrics

| Metric | Definition | Threshold(s) | Source of threshold |
|---|---|---|---|
| Defensive line height | mean `x_pitch` of the N deepest outfield players, signed toward own goal | N = 4 | `configs/metrics.yaml` |
| Width / Length | `y.max() − y.min()` / `x.max() − x.min()`, outfield only | — | — |
| Compactness | convex-hull area (m²) of outfield players | — | — |
| Distance covered | Σ smoothed step distance (Metrica only) | Sav-Gol window = 7 frames | `configs/metrics.yaml` |
| High-speed running | distance above 5.5 m/s | 5.5 m/s | conventional |
| Sprints | ≥ 1 s sustained above 7.0 m/s, ≥ 1 s recovery between | 7.0 m/s / 1 s / 1 s | conventional — _see §4_ |
| Space control | Voronoi cell area per player, clipped to pitch | — | — |
| Pressing | opponents within 5 m of the ball carrier | 5 m | `configs/metrics.yaml` |

Frames clipped for impossible speed (> 12 m/s): _N (fill from ClipReport)_.

## 3 · Validation

_(table from `reports/validation.json` — synthetic known-answer checks and
published physiological ranges, pass/fail + actual numbers)_

## 4 · Sensitivity — the honest result

> _"sprint count varies by **N%** across defensible threshold choices, so I
> report the parameters alongside the number."_

_(table + heatmap from `reports/sensitivity.json`)_

## 5 · Coverage: Metrica vs SkillCorner

_(table from `reports/validation.json` — per-player frame coverage. Metrica
≈ 100%; SkillCorner broadcast tracking drops players. Distance-covered
metrics are **not** reported for SkillCorner — not comparable.)_

## 6 · The schema contract

`src/ftm/schema.py` defines the canonical DataFrame; `validate()` enforces
it and raises on any violation. Nothing under `src/ftm/metrics/` imports
kloppy — loaders convert provider formats into the canonical frame, metrics
consume only that frame. A future Track A pipeline that emits a DataFrame
passing `validate()` runs through every metric here unchanged.

This is the same move as `backends/base.py` in
[project A1](https://github.com/Kushagra077/football-detect-serve), one level up.

## 7 · Licensing & attribution

- **This repo:** MIT. Every dependency is BSD or MIT — no AGPL anywhere, so
  MIT ships cleanly. (A1 cannot: Ultralytics is AGPL-3.0. Knowing the
  difference is the point.)
- **Metrica Sample Data:** no license stated — used with attribution to
  Metrica Sports. Files are downloaded at build time, never redistributed here.
- **SkillCorner Open Data:** MIT. Credit to SkillCorner.

## Development

```bash
pip install -e ".[dev,app]"
pytest -m "not integration" -q          # fast, offline
python -m scripts.build_cache           # provider data -> data/cache/*.parquet
python -m scripts.validate              # -> reports/validation.json + sensitivity.json
streamlit run app/streamlit_app.py
```
