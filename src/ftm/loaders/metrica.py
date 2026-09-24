"""Metrica Sample Data loader.

Metrica gives 3 sample matches, 25 Hz, all 22 players + ball, full match.
This is the "complete data" source: it proves the metrics are correct
when nothing is missing.

Game 3 is EPTS/FIFA format and the cleanest entry point. Games 1 and 2
use an older CSV layout — support them only after Game 3 works end to end.

Licensing: Metrica states no license; the README must credit them. Do not
redistribute their files — download at build time into gitignored
``data/raw/``.

kloppy usage sketch (the transform arguments are the whole point)
----------------------------------------------------------------
    from kloppy import metrica

    dataset = metrica.load_open_data(match_id=3)
    df = (
        dataset
        .transform(
            to_coordinate_system="secondspectrum",  # meters, origin center
            to_orientation="STATIC_HOME_AWAY",       # ends don't swap at HT
        )
        .to_df()
    )

Then verify before trusting anything downstream::

    assert -55 < df.x_pitch.min() and df.x_pitch.max() < 55   # not [0, 1]
    assert df.frame_rate == dataset.metadata.frame_rate       # read, not assumed
"""

from __future__ import annotations

import pandas as pd
from kloppy import metrica
from kloppy.domain import Ground, PositionType

from ftm.loaders.base import BaseLoader, MatchMeta
from ftm.schema import coerce, validate


class MetricaLoader(BaseLoader):
    """Load Metrica sample matches into the canonical frame."""

    name = "metrica"

    #: kloppy exposes these sample match ids. Keep as the source of truth
    #: for ``list_matches``.
    SAMPLE_MATCH_IDS: tuple[str, ...] = ("1", "2", "3")

    def list_matches(self) -> list[str]:
        """Return ``SAMPLE_MATCH_IDS`` as a list.

        (Later you may filter to just the games whose CSV layout you have
        implemented — start with ["3"].)
        """
        return list(self.SAMPLE_MATCH_IDS)

    def load_meta(self, match_id: str) -> MatchMeta:
        """Build :class:`MatchMeta` from ``dataset.metadata``.

        Must set ``frame_rate`` from ``metadata.frame_rate`` (25.0 here,
        but read it), ``periods`` from ``metadata.periods``, and team
        names from ``metadata.teams``. ``provider="metrica"``.
        """
        metadata = self._read_dataset(match_id).metadata

        periods = {
            period.id: (
                period.start_timestamp.total_seconds(),
                period.end_timestamp.total_seconds(),
            )
            for period in metadata.periods
        }
        home_team = next(t for t in metadata.teams if t.ground == Ground.HOME)
        away_team = next(t for t in metadata.teams if t.ground == Ground.AWAY)
        pitch = metadata.pitch_dimensions

        return MatchMeta(
            match_id=match_id,
            provider="metrica",
            frame_rate=float(metadata.frame_rate),
            pitch_length_m=float(pitch.pitch_length),
            pitch_width_m=float(pitch.pitch_width),
            home_team=home_team.name,
            away_team=away_team.name,
            periods=periods,
            notes=(
                "Metrica Sports sample open data; no license stated by "
                "Metrica, do not redistribute the raw files."
            ),
        )

    def load(self, match_id: str) -> pd.DataFrame:
        """Load one Metrica match as the canonical DataFrame.

        Steps:

        1. ``dataset = metrica.load_open_data(match_id=int(match_id))``.
        2. ``.transform(to_coordinate_system="secondspectrum",
           to_orientation="STATIC_HOME_AWAY")`` then ``.to_df()``.
        3. kloppy's wide df has one row per frame with
           ``<player>_x`` / ``<player>_y`` / ``ball_x`` / ``ball_y``
           columns — melt it to long: (period, frame_id, timestamp,
           track_id, team, x_pitch, y_pitch).
        4. ``is_ball`` = track is the ball. ``team`` category from
           kloppy's home/away plus "ball".
        5. ``is_gk``: Metrica sample metadata marks the keeper; use it.
        6. ``ball_state``: Metrica provides a ball-in-play / status
           column — map to {"alive", "dead"}. If a frame has no status,
           default "dead" for the safe direction (it only drops frames
           from averages).
        7. ``jersey_number`` from metadata where present, else ``pd.NA``.
        8. ``df = ftm.schema.coerce(df)`` ; ``return ftm.schema.validate(df)``.

        Gate for this loader (spec step 1): ``validate`` passes; a printed
        summary shows 22 players + 1 ball, ``x_pitch`` spanning ~+/-52.5 m,
        and the frame rate coming from metadata.
        """
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

        timestamp = wide["timestamp"].dt.total_seconds()

        # kloppy never populates ball_state for Metrica EPTS data; default
        # to "dead" per the loader contract (safe direction — only drops
        # frames from averages).
        ball_state = wide["ball_state"].fillna("dead")

        frames = [
            pd.DataFrame(
                {
                    "frame_id": wide["frame_id"],
                    "period": wide["period_id"],
                    "timestamp": timestamp,
                    "track_id": "ball",
                    "team": "ball",
                    "jersey_number": pd.NA,
                    "x_pitch": wide["ball_x"],
                    "y_pitch": wide["ball_y"],
                    "is_ball": True,
                    "is_gk": False,
                    "ball_state": ball_state,
                }
            )
        ]

        for player_id, team in player_team.items():
            x_col, y_col = f"{player_id}_x", f"{player_id}_y"
            if x_col not in wide.columns:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "frame_id": wide["frame_id"],
                        "period": wide["period_id"],
                        "timestamp": timestamp,
                        "track_id": player_id,
                        "team": team,
                        "jersey_number": player_jersey[player_id],
                        "x_pitch": wide[x_col],
                        "y_pitch": wide[y_col],
                        "is_ball": False,
                        "is_gk": player_is_gk[player_id],
                        "ball_state": ball_state,
                    }
                )
            )

        long_df = pd.concat(frames, ignore_index=True)
        # Rows where the tracker didn't see this track this frame — the
        # schema forbids null coordinates, so drop rather than fabricate.
        long_df = long_df.dropna(subset=["x_pitch", "y_pitch"])

        long_df = coerce(long_df)
        return validate(long_df)

    def _read_dataset(self, match_id: str):
        dataset = metrica.load_open_data(match_id=int(match_id))
        return dataset.transform(
            to_coordinate_system="secondspectrum",
            to_orientation="STATIC_HOME_AWAY",
        )