"""Tactical metrics — pure functions over the canonical DataFrame.

RULES for everything in this package:

* Import only: pandas, numpy, scipy, shapely, and ``ftm.schema``.
  Never import kloppy, a loader, or a provider SDK. If a metric needs
  provider knowledge, the abstraction has leaked — push the fix into the
  loader.
* Each function takes a validated canonical frame (optionally already
  carrying kinematics columns) and returns a tidy result frame. No
  in-place mutation of the input.
* Read every threshold from the resolved config dict (originating in
  ``configs/metrics.yaml``), passed in as an argument. No magic numbers
  in function bodies.
* Standard filters, applied by the caller or documented per function:
  outfield only (``~is_gk``), ball alive only (``ball_state == "alive"``),
  exclude the ball (``~is_ball``).

Modules
-------
shape     line height, width, length, compactness, centroid
physical  distance covered, speed zones, HSR, sprints
pressing  opponents near the ball carrier
space     Voronoi area controlled per player / per team
"""

from __future__ import annotations

__all__ = ["shape", "physical", "pressing", "space"]

from ftm.metrics import physical, pressing, shape, space  # noqa: E402,F401
