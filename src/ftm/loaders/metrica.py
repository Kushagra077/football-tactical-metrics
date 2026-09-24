"""Metrica Sample Data loader.

Metrica gives 3 sample matches, 25 Hz, all 22 players + ball, full match.
This is the "complete data" source: it proves the metrics are correct
when nothing is missing.

Game 3 is EPTS/FIFA format (XML metadata + text tracking); Games 1 and 2
use an older two-file CSV layout (one file per team, no roles, no pitch
size).

Licensing: Metrica states no license; the README must credit them. Do not
redistribute their files — they are downloaded at run time into the
gitignored ``data/raw/metrica/`` cache.

Why we don't call ``kloppy.metrica.load_open_data``
---------------------------------------------------
``load_open_data`` streams the remote files through fsspec's
``simplecache`` into ``~/kloppy_cache``. An interrupted download leaves a
truncated file there that is silently reused on every later call, so the
parser hits a half-written last line: the EPTS reader's
``regex.search(line)`` returns ``None`` (``'NoneType' object has no
attribute 'groupdict'``) and the CSV reader indexes past the end of a
short row (``IndexError``). Instead we download each raw file once,
atomically and length-checked, via ``BaseLoader._download_cached``, drop
any line kloppy's own grammar would reject (logged, normally zero), and
hand kloppy local data (``load_tracking_epts`` / ``load_tracking_csv``).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from kloppy import metrica
from kloppy.domain import Ground, PositionType

from ftm.loaders.base import BaseLoader, MatchMeta
from ftm.schema import COORD_TOLERANCE_M, PITCH_LENGTH_M, PITCH_WIDTH_M, coerce, validate

logger = logging.getLogger(__name__)

RAW_BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"

#: match id -> (metadata.xml, tracking.txt), relative to ``RAW_BASE_URL``.
_EPTS_FILES: dict[str, tuple[str, str]] = {
    "3": (
        "Sample_Game_3/Sample_Game_3_metadata.xml",
        "Sample_Game_3/Sample_Game_3_tracking.txt",
    ),
}
#: match id -> (home.csv, away.csv), relative to ``RAW_BASE_URL``.
_CSV_FILES: dict[str, tuple[str, str]] = {
    "1": (
        "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
        "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
    ),
    "2": (
        "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
        "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
    ),
}
_CSV_HEADER_LINES = 3
# Also passed to the deserializer: converting while parsing is ~2x faster
# than a separate kloppy transform pass, which then only has to reorient.
_TARGET_COORDS = "secondspectrum"

_NOTES = (
    "Metrica Sports sample open data; no license stated by Metrica, do not "
    "redistribute the raw files. ball_state derived from ball tracking "
    "(alive = ball tracked and inside the pitch)."
)
 
_CSV_NOTES = (
    " CSV format: no pitch size in the files (kloppy's default 105x68 m is"
    " used) and no roles (is_gk inferred by the goal-line heuristic)."
)


# ----- pure helpers (unit-tested without network) ---------------------------


def derive_ball_state(
    ball_x: pd.Series,
    ball_y: pd.Series,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> pd.Series:
    """Per-frame ball state from ball position alone.

    "alive" iff the ball is tracked (both coordinates non-NaN) and inside
    the pitch rectangle ``x_bounds`` x ``y_bounds`` (lines included);
    otherwise "dead". Metrica stops tracking the ball during stoppages
    (~32% of Game 3 frames), which is what makes this a usable signal.
    """
    alive = (
        ball_x.notna()
        & ball_y.notna()
        & ball_x.between(*x_bounds)
        & ball_y.between(*y_bounds)
    )
    return pd.Series(np.where(alive, "alive", "dead"), index=ball_x.index, dtype=object)


def sanitize_epts_lines(
    lines: Iterable[bytes],
    specs: Sequence[tuple[int, int, re.Pattern[str]]],
) -> tuple[list[bytes], int]:
    """Drop EPTS tracking lines that kloppy's reader would choke on.

    ``specs`` is ``(start_frame, end_frame, compiled_regex)`` per
    ``DataFormatSpecification``; a line is kept iff its leading frame
    number parses and the regex of the spec covering that frame matches
    it (the exact test kloppy applies). Lines for frames outside every
    spec are kept — kloppy stops reading there anyway.

    kloppy only switches format spec after reading a spec's ``end_frame``
    line, so that line cannot be dropped; if it is malformed we raise
    instead of producing a stream kloppy would misparse.

    Returns the kept lines and the number dropped.
    """
    kept: list[bytes] = []
    dropped = 0
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        text = line.decode("ascii", errors="replace")
        head, sep, _ = text.partition(":")
        spec = None
        if sep and head.isdigit():
            frame_id = int(head)
            spec = next((s for s in specs if s[0] <= frame_id <= s[1]), None)
            if spec is None or spec[2].search(text):
                kept.append(line + b"\n")
                continue
            if frame_id == spec[1]:
                raise ValueError(
                    f"EPTS line for frame {frame_id} is malformed and is the last frame "
                    "of its data-format block; kloppy cannot switch formats without it."
                )
        dropped += 1
    return kept, dropped


def sanitize_metrica_csv(home: bytes, away: bytes) -> tuple[bytes, bytes, int]:
    """Keep only frames that are well-formed in BOTH Metrica team CSVs.

    kloppy zips the two files row by row and raises unless frame ids and
    ball coordinates agree, so a frame dropped from one file must be
    dropped from the other. A row is well-formed when it has the header's
    column count and numeric period / frame / time fields.

    Returns the sanitized (home, away) bytes and the number of frames
    dropped (counted once per frame, not per file).
    """

    def parse(data: bytes) -> tuple[list[bytes], dict[int, tuple[bytes, bytes]], set[int]]:
        lines = data.splitlines()
        header, body = lines[:_CSV_HEADER_LINES], lines[_CSV_HEADER_LINES:]
        n_cols = header[-1].count(b",") + 1 if header else 0
        rows: dict[int, tuple[bytes, bytes]] = {}
        seen: set[int] = set()
        for line in body:
            if not line.strip():
                continue
            cols = line.split(b",")
            try:
                frame_id = int(cols[1])
                seen.add(frame_id)
                int(cols[0])
                float(cols[2])
            except (IndexError, ValueError):
                continue
            if len(cols) == n_cols:
                rows[frame_id] = (line, b",".join(cols[-2:]))
        return header, rows, seen

    home_header, home_rows, home_seen = parse(home)
    away_header, away_rows, away_seen = parse(away)
    good = sorted(
        f for f in home_rows.keys() & away_rows.keys() if home_rows[f][1] == away_rows[f][1]
    )
    all_frames = home_seen | away_seen | home_rows.keys() | away_rows.keys()
    dropped = len(all_frames) - len(good)

    def build(header: list[bytes], rows: dict[int, tuple[bytes, bytes]]) -> bytes:
        return b"\n".join([*header, *(rows[f][0] for f in good)]) + b"\n"

    return build(home_header, home_rows), build(away_header, away_rows), dropped


# ----- loader ---------------------------------------------------------------


class MetricaLoader(BaseLoader):
    """Load Metrica sample matches into the canonical frame."""

    name = "metrica"

    #: Sample match ids that load end to end. Source of truth for
    #: ``list_matches``.
    SAMPLE_MATCH_IDS: tuple[str, ...] = ("1", "2", "3")

    def list_matches(self) -> list[str]:
        """Return ``SAMPLE_MATCH_IDS`` as a list."""
        return list(self.SAMPLE_MATCH_IDS)

    def load_meta(self, match_id: str) -> MatchMeta:
        """Build :class:`MatchMeta` without parsing the match's frames.

        Game 3 (EPTS): kloppy is run with ``limit=1`` — periods, frame
        rate, teams and pitch size all come from the XML metadata, so one
        frame is enough. Games 1/2 (CSV): kloppy derives periods from the
        frames themselves, so instead the three leading columns
        (period, frame, time) of the home file are scanned with pandas;
        ``frame_rate`` is read from the time column's step and periods use
        kloppy's convention (``(first_frame - 1) / fps .. last_frame / fps``).
        """
        match_id = self._check_id(match_id)
        metadata = self._read_dataset(match_id, limit=1).metadata
        home_team = next(t for t in metadata.teams if t.ground == Ground.HOME)
        away_team = next(t for t in metadata.teams if t.ground == Ground.AWAY)
        pitch = metadata.pitch_dimensions

        if match_id in _EPTS_FILES:
            frame_rate = float(metadata.frame_rate)
            periods = {
                p.id: (p.start_timestamp.total_seconds(), p.end_timestamp.total_seconds())
                for p in metadata.periods
            }
        else:
            home_path = self._download_cached(RAW_BASE_URL + _CSV_FILES[match_id][0])
            head = pd.read_csv(
                home_path, skiprows=_CSV_HEADER_LINES, header=None, usecols=[0, 1, 2]
            )
            head.columns = ["period", "frame", "time"]
            frame_rate = float(round(1.0 / head["time"].diff().median(), 6))
            bounds = head.groupby("period")["frame"].agg(["min", "max"])
            periods = {
                int(p): (float(row["min"] - 1) / frame_rate, float(row["max"]) / frame_rate)
                for p, row in bounds.iterrows()
            }

        return MatchMeta(
            match_id=match_id,
            provider="metrica",
            frame_rate=frame_rate,
            pitch_length_m=float(pitch.pitch_length),
            pitch_width_m=float(pitch.pitch_width),
            home_team=home_team.name,
            away_team=away_team.name,
            periods=periods,
            notes=_NOTES if match_id in _EPTS_FILES else _NOTES + _CSV_NOTES,
        )

    def load(self, match_id: str) -> pd.DataFrame:
        """Load one full Metrica match as the canonical DataFrame.

        1. ``_read_dataset``: cached local raw files -> sanitized -> kloppy
           ``load_tracking_epts`` / ``load_tracking_csv`` -> transformed to
           ``secondspectrum`` meters + ``STATIC_HOME_AWAY``.
        2. kloppy's wide ``to_df()`` (one row per frame, ``<player>_x`` /
           ``<player>_y`` / ``ball_x`` / ``ball_y``) is melted to long:
           one row per (period, frame_id, track_id), plus ball rows.
        3. ``ball_state``: kloppy populates none for Metrica and the raw
           files carry no ball-status channel, so it is derived per frame
           by :func:`derive_ball_state` — "alive" iff the ball is tracked
           and inside the pitch rectangle taken from kloppy's pitch
           dimensions, else "dead". It is computed before untracked rows
           are dropped, so frames whose ball row disappears stay "dead"
           for every player row.
        4. ``is_gk``: Game 3 metadata declares positions (goalkeeper used
           directly). The CSV games declare none, so the generic
           :meth:`BaseLoader._infer_is_gk` heuristic is used there.
        5. ``jersey_number`` from kloppy's player metadata.
        6. Rows where a track is untracked (NaN) are dropped — the schema
           forbids null coordinates and we never fabricate positions. The
           rare samples further off the pitch than the schema's tolerance
           are dropped too (logged), never clipped.
        7. ``coerce`` then ``validate``.
        """
        match_id = self._check_id(match_id)
        dataset = self._read_dataset(match_id)
        metadata = dataset.metadata
        wide = dataset.to_df()

        team_by_ground = {team.ground: team for team in metadata.teams}
        player_team: dict[str, str] = {}
        player_jersey: dict[str, int | None] = {}
        player_is_gk: dict[str, bool] = {}
        for ground, team_name in ((Ground.HOME, "home"), (Ground.AWAY, "away")):
            for player in team_by_ground[ground].players:
                player_team[player.player_id] = team_name
                player_jersey[player.player_id] = player.jersey_no
                player_is_gk[player.player_id] = (
                    player.starting_position == PositionType.Goalkeeper
                )
        roles_declared = any(player_is_gk.values())

        pitch = metadata.pitch_dimensions
        ball_state = derive_ball_state(
            wide["ball_x"],
            wide["ball_y"],
            x_bounds=(pitch.x_dim.min, pitch.x_dim.max),
            y_bounds=(pitch.y_dim.min, pitch.y_dim.max),
        )
        timestamp = wide["timestamp"].dt.total_seconds()

        def block(track_id: str, team: str, jersey, x, y, is_gk: bool) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "frame_id": wide["frame_id"],
                    "period": wide["period_id"],
                    "timestamp": timestamp,
                    "track_id": track_id,
                    "team": team,
                    "jersey_number": jersey,
                    "x_pitch": x,
                    "y_pitch": y,
                    "is_ball": team == "ball",
                    "is_gk": is_gk,
                    "ball_state": ball_state,
                }
            )

        frames = [block("ball", "ball", pd.NA, wide["ball_x"], wide["ball_y"], False)]
        for player_id, team in player_team.items():
            x_col, y_col = f"{player_id}_x", f"{player_id}_y"
            if x_col not in wide.columns:
                continue
            frames.append(
                block(
                    player_id,
                    team,
                    player_jersey[player_id],
                    wide[x_col],
                    wide[y_col],
                    player_is_gk[player_id],
                )
            )

        long_df = pd.concat(frames, ignore_index=True)
        long_df = long_df.dropna(subset=["x_pitch", "y_pitch"])
        # A handful of samples sit beyond the schema's off-pitch tolerance
        # (e.g. Game 3: one player 5.3 m past the touchline for 3 frames).
        # Treat them like untracked samples rather than clipping positions.
        off_pitch = (long_df["x_pitch"].abs() > PITCH_LENGTH_M / 2 + COORD_TOLERANCE_M) | (
            long_df["y_pitch"].abs() > PITCH_WIDTH_M / 2 + COORD_TOLERANCE_M
        )
        if off_pitch.any():
            logger.info("Dropping %d samples beyond the off-pitch tolerance", int(off_pitch.sum()))
            long_df = long_df[~off_pitch]
        if not roles_declared:
            long_df["is_gk"] = self._infer_is_gk(long_df)

        long_df = coerce(long_df)
        return validate(long_df)

    # ----- internals --------------------------------------------------------

    def _check_id(self, match_id: str) -> str:
        match_id = str(match_id)
        if match_id not in _EPTS_FILES and match_id not in _CSV_FILES:
            raise KeyError(
                f"Unknown Metrica match {match_id!r}; known: {sorted({*_EPTS_FILES, *_CSV_FILES})}"
            )
        return match_id

    def _read_dataset(self, match_id: str, limit: int | None = None):
        """Cached raw files -> (sanitized) kloppy dataset in canonical coordinates.

        The single place the coordinate/orientation transform happens. With
        a ``limit`` the sanitizing pass is skipped: only the first few
        frames are parsed and a full-file scan would defeat the point.
        """
        match_id = self._check_id(match_id)
        if match_id in _EPTS_FILES:
            meta_rel, raw_rel = _EPTS_FILES[match_id]
            meta_path = self._download_cached(RAW_BASE_URL + meta_rel)
            raw_path = self._download_cached(RAW_BASE_URL + raw_rel)
            raw_data: Path | bytes = (
                raw_path if limit else self._sanitized_epts(meta_path, raw_path)
            )
            dataset = metrica.load_tracking_epts(
                meta_data=meta_path, raw_data=raw_data, limit=limit, coordinates=_TARGET_COORDS
            )
        else:
            home_rel, away_rel = _CSV_FILES[match_id]
            home_path = self._download_cached(RAW_BASE_URL + home_rel)
            away_path = self._download_cached(RAW_BASE_URL + away_rel)
            if limit:
                home_data: Path | bytes = home_path
                away_data: Path | bytes = away_path
            else:
                home_data, away_data, dropped = sanitize_metrica_csv(
                    home_path.read_bytes(), away_path.read_bytes()
                )
                self._log_dropped(dropped, f"match {match_id} CSV frames")
            dataset = metrica.load_tracking_csv(
                home_data=home_data, away_data=away_data, limit=limit, coordinates=_TARGET_COORDS
            )
        return dataset.transform(
            to_coordinate_system=_TARGET_COORDS,
            to_orientation="STATIC_HOME_AWAY",
        )

    def _sanitized_epts(self, meta_path: Path, raw_path: Path) -> Path | bytes:
        # kloppy's own metadata parser and regex builder, so "valid" means
        # exactly what the EPTS reader will accept.
        from kloppy.infra.serializers.tracking.metrica_epts.metadata import load_metadata
        from kloppy.infra.serializers.tracking.metrica_epts.reader import build_regex

        with meta_path.open("rb") as fh:
            md = load_metadata(fh)
        specs = [
            (s.start_frame, s.end_frame, re.compile(build_regex(s, md.player_channels, md.sensors)))
            for s in md.data_format_specifications
        ]
        with raw_path.open("rb") as fh:
            kept, dropped = sanitize_epts_lines(fh, specs)
        self._log_dropped(dropped, f"{raw_path.name} lines")
        return raw_path if dropped == 0 else b"".join(kept)

    @staticmethod
    def _log_dropped(n: int, what: str) -> None:
        if n:
            logger.warning("Dropped %d malformed %s", n, what)
        else:
            logger.debug("No malformed %s", what)
