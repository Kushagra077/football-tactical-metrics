"""The loader contract.

``BaseLoader`` is the interface. Concrete loaders (``metrica.py``,
``skillcorner.py``) subclass it. The metrics layer never sees a loader —
it only ever receives the DataFrame a loader returns — so this file is
where "source-agnostic" is actually enforced.
"""

from __future__ import annotations

import abc
import logging
import os
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: Env var that overrides the on-disk data root (default: ``<repo>/data``).
DATA_DIR_ENV = "FTM_DATA_DIR"

# src/ftm/loaders/base.py -> parents[3] is the repo root in a source checkout.
_SOURCE_ROOT = Path(__file__).resolve().parents[3]

# A leftover temp file older than this is from a killed download, not a live one.
_STALE_PART_S = 3600


def data_root() -> Path:
    """Return the data root.

    ``$FTM_DATA_DIR`` if set; else ``<repo>/data`` when running from a source
    checkout (editable install / ``src`` on the path); else ``./data`` so a
    regular install never writes into ``site-packages``' parent dirs.
    """
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    if (_SOURCE_ROOT / "pyproject.toml").is_file():
        return _SOURCE_ROOT / "data"
    return Path.cwd() / "data"


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

    def _download_cached(self, url: str) -> Path:
        """Fetch ``url`` once into ``data/raw/<provider>/`` and return the local path.

        The cache path mirrors the URL path (``data/raw/<provider>/<url path>``)
        so two remote files with the same basename (e.g. per-match
        ``match_data.json``) never collide. The data root resolves relative
        to the repo, not the CWD, and can be overridden with ``$FTM_DATA_DIR``.
        Writes go to a temp file in the target directory and are renamed
        into place, so an interrupted download never leaves a truncated
        file that later looks cached. ``data/`` is gitignored.
        """
        parsed = urllib.parse.urlparse(url)
        rel = urllib.parse.unquote(parsed.path).lstrip("/")
        parts = [p for p in Path(rel).parts if p not in ("", ".", "..")]
        if not parts:
            raise ValueError(f"URL has no file path component: {url!r}")
        dest = data_root() / "raw" / self.name / Path(*parts)
        if dest.is_file():
            return dest

        dest.parent.mkdir(parents=True, exist_ok=True)
        for stale in dest.parent.glob(f".{dest.name}.*.part"):
            try:
                if time.time() - stale.stat().st_mtime > _STALE_PART_S:
                    stale.unlink()
            except OSError:
                pass
        logger.info("Downloading %s -> %s", url, dest)
        fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
        try:
            written = 0
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as resp:
                expected = resp.headers.get("Content-Length")
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    written += len(chunk)
            # A dropped connection can end the stream "cleanly"; never cache a short file.
            if expected is not None and written != int(expected):
                raise OSError(f"Incomplete download of {url}: got {written} of {expected} bytes")
            os.chmod(tmp_name, 0o644)
            os.replace(tmp_name, dest)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return dest

    def _infer_is_gk(self, df: pd.DataFrame) -> pd.Series:
        """Heuristic ``is_gk`` for providers that declare no roles.

        Prefer the provider's declared position/role; use this only as a
        fallback. Rule: per team and period, the goal a team defends is the
        end its players' mean x sits closest to; the goalkeeper is the one
        track whose mean x (over the frames it appears in) is nearest that
        goal line. Deciding per track across the whole period rather than per
        frame keeps one keeper stable instead of flickering onto whichever
        defender is deepest at a set piece. Tracks present in <10% of the
        team's frames that period are not candidates. A wrong ``is_gk``
        shifts defensive line height by ~10 m.

        Known limitation: exactly one keeper per (period, team), so after a
        mid-period goalkeeper substitution the replacement is not flagged.

        Parameters
        ----------
        df:
            Long frame with at least ``period``, ``track_id``, ``team``,
            ``x_pitch``.

        Returns
        -------
        pandas.Series
            Boolean, aligned to ``df.index``. Always False for the ball.
        """
        is_gk = pd.Series(False, index=df.index, dtype=bool)
        players = df[df["team"].isin(["home", "away"])]
        if players.empty:
            return is_gk

        stats = players.groupby(["period", "team", "track_id"], observed=True)["x_pitch"].agg(
            ["mean", "size"]
        )
        keepers: set[tuple] = set()
        for _, per_track in stats.groupby(level=["period", "team"]):
            # Ignore cameo tracks (e.g. late subs) so a brief spell near
            # the goal line can't outrank the keeper who played throughout.
            regular = per_track[per_track["size"] >= 0.1 * per_track["size"].max()]
            mean_x = regular["mean"]
            # Team centroid below 0 -> defends the -x goal (and vice versa).
            defends_negative = mean_x.mean() < 0
            keepers.add(mean_x.idxmin() if defends_negative else mean_x.idxmax())

        key = pd.MultiIndex.from_frame(players[["period", "team", "track_id"]])
        is_gk.loc[players.index] = key.isin(list(keepers))
        return is_gk
