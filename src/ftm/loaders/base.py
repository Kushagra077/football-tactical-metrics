"""The loader contract.

``BaseLoader`` is the interface. Concrete loaders (``metrica.py``,
``skillcorner.py``) subclass it. The metrics layer never sees a loader —
it only ever receives the DataFrame a loader returns — so this file is
where "source-agnostic" is actually enforced.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MatchMeta:
    """Everything about a match that is NOT per-frame tracking data.

    Populate this from the provider's metadata, never from assumptions.
    ``frame_rate`` in particular must be read from the provider — kinematics
    computes ``dt`` from it, and a hardcoded 25 Hz silently corrupts every
    physical metric when the source is actually 10 Hz.

    Fields
    ------
    match_id        Provider-native id, as a string.
    provider        "metrica" | "skillcorner".
    frame_rate      Hz, from provider metadata.
    pitch_length_m  Provider's stated pitch length (usually 105).
    pitch_width_m   Provider's stated pitch width (usually 68).
    home_team       Display name.
    away_team       Display name.
    periods         Mapping period -> (start_timestamp, end_timestamp) in
                    seconds, useful for trimming and for building the
                    dashboard time slider.
    notes           Free text: licensing reminder, known quirks, etc.
    """

    match_id: str
    provider: str
    frame_rate: float
    pitch_length_m: float
    pitch_width_m: float
    home_team: str
    away_team: str
    periods: dict[int, tuple[float, float]]
    notes: str = ""


class BaseLoader(abc.ABC):
    """Convert one provider's match into the canonical DataFrame.

    Contract for subclasses:

    * ``load(match_id)`` returns a frame that passes
      ``ftm.schema.validate()`` — meters, fixed orientation, all 11
      canonical columns.
    * ``load`` must be pure w.r.t. its input: same ``match_id`` in, same
      frame out. Any network fetch should go through a local cache.
    * The class must NOT leak provider types. Return plain pandas; do not
      return kloppy datasets.
    * ``list_matches()`` enumerates what this provider offers so the
      dashboard and ``build_cache.py`` can iterate without hardcoding ids.
    """

    #: Short provider key, e.g. "metrica". Set on each subclass.
    name: str

    @abc.abstractmethod
    def list_matches(self) -> list[str]:
        """Return the match ids this loader can load, as strings.

        For Metrica this is a fixed small list (the sample games). For
        SkillCorner it is derived from the open-data index on GitHub.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_meta(self, match_id: str) -> MatchMeta:
        """Return :class:`MatchMeta` for one match without loading frames.

        Cheap call used to populate dropdowns and to get ``frame_rate``
        before deciding how to downsample.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, match_id: str) -> pd.DataFrame:
        """Load one match as the canonical tracking DataFrame.

        Implementation outline (see subclasses for provider specifics):

        1. Obtain the raw match (kloppy ``load_open_data`` for Metrica;
           download + parse JSON/CSV for SkillCorner).
        2. Transform to meters + fixed orientation. For kloppy:
           ``to_coordinate_system="secondspectrum"`` and
           ``to_orientation="STATIC_HOME_AWAY"``. Neither is the default.
        3. Reshape provider rows/columns into the canonical long format:
           one row per (period, frame, track_id), plus ball rows.
        4. Fill the two added columns: ``is_gk`` from provider position
           data; ``ball_state`` from possession / period / set-piece
           info (default to "alive" only when you can justify it).
        5. ``df = ftm.schema.coerce(df)`` then
           ``return ftm.schema.validate(df)``.

        Raises
        ------
        ftm.schema.SchemaError
            If the assembled frame fails validation — do not swallow it.
        FileNotFoundError / KeyError
            If ``match_id`` is unknown.
        """
        raise NotImplementedError

    # ----- shared helpers subclasses may reuse -------------------------------

    def _download_cached(self, url: str) -> "bytes | str":
        """Fetch ``url`` once, store it under ``data/raw/<provider>/``.

        Return the local file contents (or path). Both loaders pull from
        remote open-data URLs, so a shared on-disk cache keeps CI and
        repeated dashboard builds from re-downloading. Keep raw files out
        of git (``data/`` is gitignored).
        """
        raise NotImplementedError

    def _infer_is_gk(self, *args, **kwargs) -> pd.Series:
        """Derive the ``is_gk`` boolean for each track.

        Prefer the provider's declared position/role. Fallback heuristic
        (document it if you use it): the outfield-excluded player who
        stays closest to their own goal line across the match. A wrong
        ``is_gk`` shifts defensive line height by ~10 m, so this is worth
        getting right rather than guessing per frame.
        """
        raise NotImplementedError
