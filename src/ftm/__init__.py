"""ftm — Football Tactical Metrics.

A source-agnostic library that turns professional tracking data into
team-shape and physical metrics.

The whole repo hangs off one rule:

    ``schema.py`` defines the canonical DataFrame. Loaders convert each
    provider's native format into that frame. Everything under
    ``ftm.metrics`` consumes ONLY the canonical frame and is forbidden
    from importing kloppy (or any provider SDK).

That restriction is what makes a future Track A pipeline droppable in:
if its output passes ``schema.validate()``, every metric here runs on it
unchanged.

Sub-packages
------------
loaders   Provider adapters: ``load(match_id) -> canonical DataFrame``.
metrics   Pure functions over the canonical frame (shape, physical,
          pressing, space).

Modules
-------
schema      The column contract plus ``validate()``.
kinematics  Smoothing, velocity, speed, distance.
pipeline    match_id -> tidy metrics parquet.
viz         mplsoccer wrappers for the dashboard and report figures.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
