"""SkillCorner Open Data loader — the abstraction test.

SkillCorner publishes 10 A-League 24/25 matches, ~10 Hz, BROADCAST
tracking: only players in frame are tracked, so players drop in and out.
This source proves the metrics survive incomplete data, and that the
loader abstraction actually abstracts.

Hard rule (spec step 5): adding this loader must not require changing a
single line under ``ftm/metrics``. If you feel the urge to special-case
SkillCorner inside a metric, stop — the fix belongs here or in how the
pipeline flags low-coverage frames.

License: MIT (data + code). Files load from the SkillCorner
opendata GitHub repo; cache raw JSON under gitignored ``data/raw/``.

Incomplete-data handling this loader is responsible for
------------------------------------------------------
* Emit every tracked (frame, player) row you have; do NOT forward-fill
  missing players with stale positions.
* Per-player coverage % (frames present / frames in period) is computed
  downstream in the pipeline, but this loader must preserve the true
  gaps so that computation is honest.
* ``ball_state``: SkillCorner ball tracking is sparse; when the ball
  position is unknown, still emit period/possession-derived state where
  possible, else "dead".
"""

from __future__ import annotations

import pandas as pd

from ftm.loaders.base import BaseLoader, MatchMeta


class SkillCornerLoader(BaseLoader):
    """Load SkillCorner open broadcast tracking into the canonical frame."""

    name = "skillcorner"

    #: Base URL of the open-data repo (raw.githubusercontent...). The
    #: match index lists available ids and their metadata/tracking paths.
    OPENDATA_BASE_URL: str = "https://raw.githubusercontent.com/SkillCorner/opendata/master/"

    def list_matches(self) -> list[str]:
        """Read the open-data match index and return available match ids.

        Fetch the index JSON via ``_download_cached``; return the ids as
        strings. Do not hardcode the 10 ids — let the index drive it.
        """
        raise NotImplementedError

    def load_meta(self, match_id: str) -> MatchMeta:
        """Build :class:`MatchMeta` from the match's ``match_data.json``.

        ``frame_rate`` is ~10 Hz but READ it from the file. ``pitch_length_m``
        / ``pitch_width_m`` come from the file too (they vary by venue).
        ``provider="skillcorner"``; ``notes`` should record "broadcast
        tracking, players drop out — distance covered NOT reported".
        """
        raise NotImplementedError

    def load(self, match_id: str) -> pd.DataFrame:
        """Load one SkillCorner match as the canonical DataFrame.

        Steps:

        1. Try kloppy's ``skillcorner`` module first
           (``skillcorner.load`` / ``load_open_data``); it already knows
           this format. If you go raw instead, parse
           ``structured_data.json`` (per-frame tracking) + ``match_data.json``
           (rosters, pitch, teams).
        2. Transform to meters + fixed orientation, same arguments as
           Metrica (``secondspectrum`` + ``STATIC_HOME_AWAY``). If parsing
           raw, replicate that: scale normalized coords by pitch size,
           translate origin to center, and flip the second-period frame
           for the team that changed ends so orientation stays static.
        3. Long format: one row per tracked (period, frame_id, timestamp,
           track_id). MISSING players simply have no row for that frame.
        4. ``is_gk`` from roster position in ``match_data.json``.
        5. ``jersey_number`` from roster (usually present here).
        6. ``ball_state`` as described in the module docstring.
        7. ``df = ftm.schema.coerce(df)`` ; ``return ftm.schema.validate(df)``.

        Gate (spec step 5): ``validate`` passes on this output, the full
        metrics pipeline runs on it unchanged, and ``reports/validation.json``
        gains a coverage table showing the Metrica vs. SkillCorner gap.
        """
        raise NotImplementedError
