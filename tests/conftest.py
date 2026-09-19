"""Shared fixtures and synthetic-data builders for the test suite.

Nothing here touches a real provider. These builders are also imported by
``scripts/validate.py`` so the committed validation report exercises the
same known-answer inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ftm.schema import coerce


def make_canonical_frame(
    *,
    n_frames: int = 10,
    frame_rate: float = 25.0,
    home_xy: dict | None = None,
    away_xy: dict | None = None,
    ball_xy=(0.0, 0.0),
    period: int = 1,
    ball_state: str = "alive",
) -> pd.DataFrame:
    """Build a minimal frame that passes ``schema.validate(strict_ranges=?)``.

    ``home_xy`` / ``away_xy`` map ``track_id -> (x, y)`` held constant
    across all ``n_frames`` (or a callable ``frame_idx -> (x, y)`` for
    moving players). Marks one player per team as GK. Fills every
    canonical column with valid dtypes.

    Keep this the single place tests construct canonical data so a schema
    change only needs fixing here.
    """

    df = pd.DataFrame({
    'frame_id': [1, 1],
    'period': [1, 1],
    'timestamp': [0.0, 0.0],
    'track_id': ['player_7', 'ball'],
    'team': ['home', 'ball'],
    'jersey_number': [7, None],
    'x_pitch': [10.0, 0.0],
    'y_pitch': [5.0, 0.0],
    'is_ball': [False, True],
    'is_gk': [False, False],
    'ball_state': ['alive', 'alive'],}
    )

    return coerce(df)


def straight_line_track(
    *, speed_mps: float = 5.0, duration_s: float = 10.0, frame_rate: float = 25.0
) -> pd.DataFrame:
    """One player moving in +x at constant speed. Known distance =
    ``speed_mps * duration_s``."""
    raise NotImplementedError


def noisy_stationary_track(
    *, noise_sd_m: float = 0.05, duration_s: float = 10.0, frame_rate: float = 25.0,
    seed: int = 0,
) -> pd.DataFrame:
    """One player at the origin plus i.i.d. Gaussian position noise. True
    distance travelled ~= 0; naive differencing would report the summed
    noise. This is the test that separates 'measuring movement' from
    'measuring tracking noise'."""
    raise NotImplementedError


def circular_track(
    *, radius_m: float = 9.15, revolutions: float = 1.0, speed_mps: float = 4.0,
    frame_rate: float = 25.0,
) -> pd.DataFrame:
    """One player on a circle. Known distance = ``2*pi*radius_m *
    revolutions``."""
    raise NotImplementedError


def flat_back_four(*, x_line: float = -30.0, team: str = "home") -> pd.DataFrame:
    """A single frame: 4 defenders at ``x = x_line`` (spread in y), plus a
    few players further up and a GK behind. ``line_height`` for ``team``
    must come back == ``x_line``."""
    raise NotImplementedError


@pytest.fixture
def metrics_cfg() -> dict:
    """The parsed ``configs/metrics.yaml`` as a dict, for tests that need
    thresholds. Load the real file so tests fail if a key is renamed."""
    raise NotImplementedError
