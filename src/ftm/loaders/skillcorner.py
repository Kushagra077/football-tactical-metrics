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

The open-data release only ships ``*_tracking_extrapolated.jsonl``: every
off-camera player gets a model-estimated position flagged
``is_detected: false``. kloppy keeps those rows and drops the flag, so this
loader re-reads the flag from the raw file and keeps only detected rows —
an extrapolated position is a fabricated one, just like a forward-fill.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from kloppy import skillcorner
from kloppy.domain import Ground, PositionType
from kloppy.utils import github_resolve_raw_data_url

from ftm.loaders.base import BaseLoader, MatchMeta
from ftm.schema import coerce, validate

_REPO = "SkillCorner/opendata"
_BRANCH = "master"
_INDEX_PATH = "data/matches.json"
_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "skillcorner"
_CANONICAL_PERIODS = (1, 2)


@dataclass(frozen=True)
class PlayerInfo:
    team: str
    jersey_number: int | None
    is_gk: bool


def wide_to_long(
    wide: pd.DataFrame, players: Mapping[str, PlayerInfo]
) -> pd.DataFrame:
    """Melt kloppy's wide frame (``<pid>_x``/``<pid>_y``, ``ball_x``/``ball_y``)
    into canonical long rows. Rows with missing coordinates are dropped, never
    filled. ``ball_state`` missing -> "dead". Output is not yet coerced."""
    timestamp = wide["timestamp"]
    if pd.api.types.is_timedelta64_dtype(timestamp):
        timestamp = timestamp.dt.total_seconds()
    ball_state = wide["ball_state"].astype(object).where(wide["ball_state"].notna(), "dead")

    def block(track_id, team, jersey, x, y, is_ball, is_gk) -> pd.DataFrame:
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
                "is_ball": is_ball,
                "is_gk": is_gk,
                "ball_state": ball_state,
            }
        )

    frames = [block("ball", "ball", pd.NA, wide["ball_x"], wide["ball_y"], True, False)]
    for player_id, info in players.items():
        x_col, y_col = f"{player_id}_x", f"{player_id}_y"
        if x_col not in wide.columns:
            continue
        jersey = pd.NA if info.jersey_number is None else info.jersey_number
        frames.append(
            block(player_id, info.team, jersey, wide[x_col], wide[y_col], False, info.is_gk)
        )

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.dropna(subset=["x_pitch", "y_pitch"])
    return long_df[long_df["period"].isin(_CANONICAL_PERIODS)].reset_index(drop=True)


def detected_pairs(records: Iterable[Mapping]) -> pd.DataFrame:
    """(frame_id, track_id) pairs the tracker actually saw, from raw SkillCorner
    v3 frame records. The ball's track_id is "ball"."""
    frame_ids: list[int] = []
    track_ids: list[str] = []
    for rec in records:
        if rec.get("period") is None:
            continue
        frame = int(rec["frame"])
        for player in rec.get("player_data") or ():
            if player.get("is_detected", True):
                frame_ids.append(frame)
                track_ids.append(str(player["player_id"]))
        ball = rec.get("ball_data") or {}
        if ball.get("x") is not None and ball.get("is_detected", True):
            frame_ids.append(frame)
            track_ids.append("ball")
    return pd.DataFrame({"frame_id": frame_ids, "track_id": track_ids})


def keep_detected(long_df: pd.DataFrame, detected: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose (frame_id, track_id) is not in ``detected``."""
    key = pd.MultiIndex.from_arrays([long_df["frame_id"], long_df["track_id"].astype(str)])
    seen = pd.MultiIndex.from_arrays([detected["frame_id"], detected["track_id"]])
    return long_df[key.isin(seen)].reset_index(drop=True)


class SkillCornerLoader(BaseLoader):
    """Load SkillCorner open broadcast tracking into the canonical frame."""

    name = "skillcorner"

    #: Base URL of the open-data repo (raw.githubusercontent...). The
    #: match index lists available ids and their metadata/tracking paths.
    OPENDATA_BASE_URL: str = "https://raw.githubusercontent.com/SkillCorner/opendata/master/"

    def __init__(self, raw_dir: str | os.PathLike | None = None) -> None:
        self.raw_dir = Path(raw_dir) if raw_dir is not None else _RAW_DIR
        self._last: tuple[str, object] | None = None

    def list_matches(self) -> list[str]:
        """Read the open-data match index and return available match ids.

        Fetch the index JSON via ``_download_cached``; return the ids as
        strings. Do not hardcode the 10 ids — let the index drive it.
        """
        index = json.loads(self._fetch(_INDEX_PATH).read_text())
        return [str(match["id"]) for match in index]

    def load_meta(self, match_id: str) -> MatchMeta:
        """Build :class:`MatchMeta` from the match's ``match_data.json``.

        ``frame_rate`` is ~10 Hz but READ it from the file. ``pitch_length_m``
        / ``pitch_width_m`` come from the file too (they vary by venue).
        ``provider="skillcorner"``; ``notes`` should record "broadcast
        tracking, players drop out — distance covered NOT reported".
        """
        match_id = str(match_id)
        raw_meta = json.loads(self._fetch(self._match_path(match_id)).read_text())
        metadata = self._read_dataset(match_id).metadata

        periods = {
            period.id: (
                period.start_timestamp.total_seconds(),
                period.end_timestamp.total_seconds(),
            )
            for period in metadata.periods
            if period.id in _CANONICAL_PERIODS
        }
        home_team = next(t for t in metadata.teams if t.ground == Ground.HOME)
        away_team = next(t for t in metadata.teams if t.ground == Ground.AWAY)

        return MatchMeta(
            match_id=match_id,
            provider="skillcorner",
            frame_rate=float(metadata.frame_rate),
            pitch_length_m=float(raw_meta["pitch_length"]),
            pitch_width_m=float(raw_meta["pitch_width"]),
            home_team=home_team.name,
            away_team=away_team.name,
            periods=periods,
            notes=(
                "SkillCorner open data (MIT). Broadcast tracking, players drop out — "
                "distance covered NOT reported. Only is_detected rows kept "
                "(extrapolated off-camera positions dropped)."
            ),
        )

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
        match_id = str(match_id)
        dataset = self._read_dataset(match_id)
        metadata = dataset.metadata

        players: dict[str, PlayerInfo] = {}
        for team in metadata.teams:
            team_name = "home" if team.ground == Ground.HOME else "away"
            for player in team.players:
                players[str(player.player_id)] = PlayerInfo(
                    team=team_name,
                    jersey_number=player.jersey_no,
                    is_gk=player.starting_position == PositionType.Goalkeeper,
                )

        # kloppy's ball_state is possession-derived: ALIVE iff a team holds
        # possession in that frame, else DEAD.
        long_df = wide_to_long(dataset.to_df(), players)
        long_df = keep_detected(long_df, self._detected(match_id))
        return validate(coerce(long_df))

    # ----- internals ---------------------------------------------------------

    @staticmethod
    def _match_path(match_id: str) -> str:
        return f"data/matches/{match_id}/{match_id}_match.json"

    @staticmethod
    def _tracking_path(match_id: str) -> str:
        return f"data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl"

    def _fetch(self, repo_path: str) -> Path:
        """Download ``repo_path`` from the open-data repo once into ``raw_dir``."""
        local = self.raw_dir / repo_path.removeprefix("data/")
        if local.exists():
            return local
        if repo_path.endswith(".jsonl"):
            url = github_resolve_raw_data_url(_REPO, _BRANCH, repo_path)  # Git LFS
        else:
            url = self.OPENDATA_BASE_URL + repo_path
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(local.suffix + ".part")
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        os.replace(tmp, local)
        return local

    def _check_known(self, match_id: str) -> None:
        if match_id not in self.list_matches():
            raise KeyError(f"Unknown SkillCorner match id {match_id!r}")

    def _read_dataset(self, match_id: str):
        if self._last is not None and self._last[0] == match_id:
            return self._last[1]
        self._check_known(match_id)
        dataset = skillcorner.load(
            meta_data=str(self._fetch(self._match_path(match_id))),
            raw_data=str(self._fetch(self._tracking_path(match_id))),
        ).transform(
            to_coordinate_system="secondspectrum",
            to_orientation="STATIC_HOME_AWAY",
        )
        self._last = (match_id, dataset)
        return dataset

    def _detected(self, match_id: str) -> pd.DataFrame:
        with open(self._fetch(self._tracking_path(match_id))) as fh:
            return detected_pairs(json.loads(line) for line in fh if line.strip())
