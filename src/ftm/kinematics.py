"""Smoothing, velocity, speed, distance.

Build and TEST this before any metric that depends on speed (spec step 2:
"build the measurement path and test it while it's cheap").

Why smoothing is not optional
-----------------------------
Raw position differencing at 25 Hz turns single-frame tracking jitter
into 20+ m/s speed spikes. Every physical metric (distance, HSR, sprints)
would inherit that noise and read ~30% high. The fix is to smooth
*position* with a Savitzky-Golay filter, then difference.

Parameters come from ``configs/metrics.yaml`` — never inline them:
    kinematics:
      savgol_window_frames: <odd int>
      savgol_polyorder: <int, < window>
      max_speed_mps: 12.0        # clip ceiling for impossible speeds

``dt`` comes from ``MatchMeta.frame_rate`` (``dt = 1 / frame_rate``).
Never hardcode 0.04.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class KinematicsConfig:
    """Resolved kinematics parameters (loaded from ``metrics.yaml``).

    savgol_window_frames  Odd window length for the Savitzky-Golay filter,
                          in frames. Larger = smoother = more lag. This is
                          one of the two knobs the sensitivity sweep moves.
    savgol_polyorder      Polynomial order, must be < window.
    max_speed_mps         Speeds above this are clipped and counted.
    frame_rate            Hz, from the match metadata (not the config file).
    """

    savgol_window_frames: int
    savgol_polyorder: int
    max_speed_mps: float
    frame_rate: float

    @property
    def dt(self) -> float:
        """Seconds between frames = ``1 / frame_rate``."""
        raise NotImplementedError


@dataclass(frozen=True)
class ClipReport:
    """How many frames hit the speed ceiling.

    ``n_clipped`` and ``fraction`` go in the README per spec step 2
    ("count how many frames you clipped — that count goes in the README").
    """

    n_clipped: int
    n_total: int
    fraction: float
    max_speed_seen_mps: float


def add_kinematics(
    df: pd.DataFrame,
    cfg: KinematicsConfig,
) -> tuple[pd.DataFrame, ClipReport]:
    """Add per-frame velocity/speed columns to a canonical frame.

    Returns a copy of ``df`` with these columns added, plus a
    :class:`ClipReport`:

        vx, vy      float64   Smoothed velocity components, m/s.
        speed       float64   ``hypot(vx, vy)`` after clipping, m/s.
        step_dist   float64   Distance travelled since the player's
                              previous frame, m (``speed * dt``, or the
                              norm of the smoothed position delta).

    Algorithm, applied INDEPENDENTLY per (track_id, period) group, with
    each group sorted by ``timestamp``:

    1. Skip the ball (``is_ball``) for speed — or process it, but never
       let ball motion enter player physical aggregates.
    2. If the group has fewer frames than ``savgol_window_frames``, fall
       back to a shorter odd window or plain centered differences;
       document the choice.
    3. Savitzky-Golay smooth ``x_pitch`` and ``y_pitch``
       (``scipy.signal.savgol_filter``), window = ``savgol_window_frames``,
       polyorder = ``savgol_polyorder``.
    4. Velocity = centered difference of smoothed position divided by
       ``dt`` (or ``savgol_filter(..., deriv=1, delta=dt)`` directly).
    5. ``speed = hypot(vx, vy)``. Clip to ``max_speed_mps``; accumulate
       clip counts into the ClipReport.
    6. ``step_dist`` from the smoothed deltas; first frame of a group = 0.

    Gaps: for SkillCorner a player can vanish for many frames then
    reappear. Do NOT interpolate across a gap longer than a small
    threshold (say > 1 s) — treat reappearance as a fresh sub-track so a
    teleport doesn't become a 40 m "step".

    Test expectations (``tests/test_kinematics.py``):
    * straight line, 5 m/s for 10 s  -> total ``step_dist`` == 50 m +/- tol
    * stationary + Gaussian noise    -> total ``step_dist`` ~= 0, NOT the
                                        summed noise
    * circular path, known radius    -> total ~= 2*pi*r +/- tol
    """
    raise NotImplementedError


def total_distance(df_with_kin: pd.DataFrame) -> pd.DataFrame:
    """Sum ``step_dist`` per (track_id, period) and per (track_id).

    Returns a tidy frame: one row per track (optionally per period) with
    ``distance_m``. Ball rows excluded.

    Sanity band (spec step 4 gate): for a full Metrica match, per-player
    totals land in 9–12 km. Outside that, the bug is upstream (smoothing
    / dt / orientation) — go fix step 2, do not tune thresholds until the
    number looks nice.
    """
    raise NotImplementedError


def distance_by_speed_zone(
    df_with_kin: pd.DataFrame,
    zones_mps: list[tuple[float, float]],
) -> pd.DataFrame:
    """Distance per player split into speed bands.

    ``zones_mps`` is a list of ``(low, high)`` bounds from
    ``metrics.yaml`` (e.g. walk / jog / run / HSR / sprint). Returns a
    tidy frame: (track_id, zone_label, distance_m). Used by the physical
    metrics module and the per-player dashboard table.
    """
    raise NotImplementedError
