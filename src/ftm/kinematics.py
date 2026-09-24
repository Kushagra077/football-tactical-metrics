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
      max_gap_s: 1.0             # longer absence => fresh sub-track

``dt`` comes from ``MatchMeta.frame_rate`` (``dt = 1 / frame_rate``).
Never hardcode 0.04.

Conventions
-----------
* Ball rows (and rows with missing coordinates) get NaN ``vx``, ``vy``,
  ``speed`` and ``step_dist``: ball motion can never leak into a player
  aggregate, whatever the consumer forgets to filter.
* ``step_dist = speed * (t_i - t_{i-1})`` with the *clipped* speed, so
  clipping a spike also caps its distance contribution. The first frame
  of every (sub-)track has ``step_dist = 0``.
* Speed-zone labels are ``f"{low:g}-{high:g}"`` (e.g. ``"0-2"``,
  ``"5.5-7"``) and zones are half-open ``[low, high)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

KIN_COLUMNS: tuple[str, ...] = ("vx", "vy", "speed", "step_dist")


@dataclass(frozen=True)
class KinematicsConfig:
    """Resolved kinematics parameters (loaded from ``metrics.yaml``).

    savgol_window_frames  Odd window length for the Savitzky-Golay filter,
                          in frames. Larger = smoother = more lag. This is
                          one of the two knobs the sensitivity sweep moves.
    savgol_polyorder      Polynomial order, must be < window.
    max_speed_mps         Speeds above this are clipped and counted.
    frame_rate            Hz, from the match metadata (not the config file).
    max_gap_s             A track absent for longer than this (seconds)
                          restarts as a fresh sub-track on reappearance.
    """

    savgol_window_frames: int
    savgol_polyorder: int
    max_speed_mps: float
    frame_rate: float
    max_gap_s: float

    def __post_init__(self) -> None:
        if self.savgol_window_frames % 2 != 1:
            raise ValueError(
                f"savgol_window_frames must be odd, got {self.savgol_window_frames}"
            )
        if not 1 <= self.savgol_polyorder < self.savgol_window_frames:
            raise ValueError(
                f"savgol_polyorder must be in [1, window), got {self.savgol_polyorder}"
            )
        if self.frame_rate <= 0:
            raise ValueError(f"frame_rate must be positive, got {self.frame_rate}")
        if self.max_speed_mps <= 0 or self.max_gap_s <= 0:
            raise ValueError("max_speed_mps and max_gap_s must be positive")

    @classmethod
    def from_metrics_cfg(cls, metrics_cfg: dict[str, Any], frame_rate: float) -> KinematicsConfig:
        """Build from the parsed ``metrics.yaml`` plus the match frame rate."""
        kin = metrics_cfg["kinematics"]
        return cls(
            savgol_window_frames=int(kin["savgol_window_frames"]),
            savgol_polyorder=int(kin["savgol_polyorder"]),
            max_speed_mps=float(kin["max_speed_mps"]),
            frame_rate=float(frame_rate),
            max_gap_s=float(kin["max_gap_s"]),
        )

    @property
    def dt(self) -> float:
        """Seconds between frames = ``1 / frame_rate``."""
        return 1.0 / self.frame_rate


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


def _uniform_velocity(pos: np.ndarray, cfg: KinematicsConfig) -> np.ndarray:
    """d(pos)/dt for positions sampled every ``cfg.dt`` (shape (n, 2))."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    window = min(cfg.savgol_window_frames, n if n % 2 == 1 else n - 1)
    if window > cfg.savgol_polyorder:
        return savgol_filter(
            pos, window, cfg.savgol_polyorder, deriv=1, delta=cfg.dt, axis=0, mode="interp"
        )
    return np.gradient(pos, cfg.dt, axis=0)


def _velocity(t: np.ndarray, pos: np.ndarray, cfg: KinematicsConfig) -> np.ndarray:
    """Velocity of one sub-track at its own timestamps ``t``.

    Positions are linearly resampled onto a uniform ``dt`` grid first, so
    dropped frames (shorter than ``max_gap_s``) don't get treated as a
    single ``dt`` step; on regularly sampled input this is the identity.
    """
    n_grid = int(round((t[-1] - t[0]) / cfg.dt)) + 1
    grid = t[0] + np.arange(n_grid) * cfg.dt
    regular = n_grid == len(t) and np.allclose(grid, t, rtol=0.0, atol=cfg.dt * 1e-3)
    if regular:
        return _uniform_velocity(pos, cfg)
    resampled = np.column_stack([np.interp(grid, t, pos[:, k]) for k in range(2)])
    vel = _uniform_velocity(resampled, cfg)
    return np.column_stack([np.interp(t, grid, vel[:, k]) for k in range(2)])


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

    Implementation choices (see module docstring for conventions):

    * Velocity is ``savgol_filter(..., deriv=1, delta=dt)`` per sub-track;
      a sub-track is a maximal run of a (track_id, period) group whose
      consecutive timestamps differ by at most ``max_gap_s``. Dropped
      frames inside a sub-track are bridged by linear resampling onto the
      ``dt`` grid before filtering.
    * Sub-tracks shorter than the window use the largest odd window
      ``<= len`` that is still ``> polyorder``; failing that, centered
      differences (``np.gradient``); a single-frame sub-track has zero
      velocity.
    * When speed is clipped, ``vx``/``vy`` are rescaled so that
      ``hypot(vx, vy) == speed`` still holds.

    Test expectations (``tests/test_kinematics.py``):
    * straight line, 5 m/s for 10 s  -> total ``step_dist`` == 50 m +/- tol
    * stationary + Gaussian noise    -> total ``step_dist`` ~= 0, NOT the
                                        summed noise
    * circular path, known radius    -> total ~= 2*pi*r +/- tol
    """
    out = df.copy()
    n_rows = len(out)
    vel = np.full((n_rows, 2), np.nan)
    dt_step = np.full(n_rows, np.nan)

    xy = out[["x_pitch", "y_pitch"]].to_numpy(dtype="float64")
    usable = ~out["is_ball"].to_numpy(dtype=bool) & np.isfinite(xy).all(axis=1)
    sub = out.loc[usable, ["track_id", "period", "timestamp"]].copy()
    sub["_pos"] = np.flatnonzero(usable)
    sub = sub.sort_values(["track_id", "period", "timestamp"], kind="mergesort")

    for _, grp in sub.groupby(["track_id", "period"], sort=False, observed=True):
        pos_idx = grp["_pos"].to_numpy()
        t = grp["timestamp"].to_numpy(dtype="float64")
        gaps = np.diff(t)
        breaks = np.flatnonzero(gaps > cfg.max_gap_s) + 1
        step_t = np.concatenate([[0.0], gaps])
        step_t[breaks] = 0.0
        dt_step[pos_idx] = step_t
        for seg in np.split(np.arange(len(pos_idx)), breaks):
            rows = pos_idx[seg]
            vel[rows] = _velocity(t[seg], xy[rows], cfg)

    raw_speed = np.hypot(vel[:, 0], vel[:, 1])
    n_total = int(usable.sum())
    clipped = raw_speed > cfg.max_speed_mps
    n_clipped = int(clipped.sum())
    speed = np.minimum(raw_speed, cfg.max_speed_mps)
    scale = np.where(clipped, cfg.max_speed_mps / np.where(clipped, raw_speed, 1.0), 1.0)
    vel *= scale[:, None]

    out["vx"] = vel[:, 0]
    out["vy"] = vel[:, 1]
    out["speed"] = speed
    out["step_dist"] = speed * dt_step
    for col in KIN_COLUMNS:
        out[col] = out[col].astype("float64")

    report = ClipReport(
        n_clipped=n_clipped,
        n_total=n_total,
        fraction=n_clipped / n_total if n_total else 0.0,
        max_speed_seen_mps=float(np.nanmax(raw_speed)) if n_total else 0.0,
    )
    return out, report


def _player_rows(df_with_kin: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ("speed", "step_dist") if c not in df_with_kin.columns]
    if missing:
        raise KeyError(f"kinematics columns missing {missing}; run add_kinematics first")
    mask = ~df_with_kin["is_ball"].astype(bool) & df_with_kin["step_dist"].notna()
    return df_with_kin.loc[mask]


def total_distance(df_with_kin: pd.DataFrame, *, by_period: bool = False) -> pd.DataFrame:
    """Sum ``step_dist`` per (track_id, period) and per (track_id).

    Returns a tidy frame: one row per track (optionally per period) with
    ``distance_m``. Ball rows excluded.

    Columns: ``track_id``, ``distance_m``; with ``by_period=True`` also
    ``period`` (one row per (track_id, period)).

    Sanity band (spec step 4 gate): for a full Metrica match, per-player
    totals land in 9–12 km. Outside that, the bug is upstream (smoothing
    / dt / orientation) — go fix step 2, do not tune thresholds until the
    number looks nice.
    """
    keys = ["track_id", "period"] if by_period else ["track_id"]
    players = _player_rows(df_with_kin)
    res = players.groupby(keys, observed=True, sort=True)["step_dist"].sum()
    return res.rename("distance_m").reset_index()


def zone_label(low: float, high: float) -> str:
    """Canonical label for a speed zone, e.g. ``(5.5, 7.0) -> "5.5-7"``."""
    return f"{low:g}-{high:g}"


def distance_by_speed_zone(
    df_with_kin: pd.DataFrame,
    zones_mps: list[tuple[float, float]],
) -> pd.DataFrame:
    """Distance per player split into speed bands.

    ``zones_mps`` is a list of ``(low, high)`` bounds from
    ``metrics.yaml`` (e.g. walk / jog / run / HSR / sprint). Returns a
    tidy frame: (track_id, zone_label, distance_m). Used by the physical
    metrics module and the per-player dashboard table.

    Zones are half-open ``[low, high)``, must be contiguous and ascending,
    and every player frame's speed must fall in one of them (so zone
    distances sum to :func:`total_distance`); otherwise ``ValueError``.
    Every (track, zone) pair is present, zero-filled, in zone order.
    """
    zones = [(float(lo), float(hi)) for lo, hi in zones_mps]
    if not zones:
        raise ValueError("zones_mps is empty")
    contiguous = all(a[1] == b[0] for a, b in zip(zones, zones[1:], strict=False))
    if not contiguous or any(lo >= hi for lo, hi in zones):
        raise ValueError(f"speed zones must be ascending and contiguous, got {zones}")

    players = _player_rows(df_with_kin)
    speed = players["speed"].to_numpy(dtype="float64")
    edges = np.array([lo for lo, _ in zones] + [zones[-1][1]])
    zone_idx = np.searchsorted(edges, speed, side="right") - 1
    outside = (zone_idx < 0) | (zone_idx >= len(zones))
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} frames have speed outside zones {zones} "
            f"(range seen {speed.min():g}..{speed.max():g} m/s)"
        )

    labels = [zone_label(lo, hi) for lo, hi in zones]
    tracks = sorted(players["track_id"].astype(str).unique())
    sums = (
        pd.DataFrame(
            {
                "track_id": players["track_id"].astype(str).to_numpy(),
                "zone_idx": zone_idx,
                "distance_m": players["step_dist"].to_numpy(dtype="float64"),
            }
        )
        .groupby(["track_id", "zone_idx"])["distance_m"]
        .sum()
    )
    full = pd.MultiIndex.from_product([tracks, range(len(zones))], names=["track_id", "zone_idx"])
    res = sums.reindex(full, fill_value=0.0).reset_index()
    res["zone_label"] = [labels[i] for i in res["zone_idx"]]
    return res[["track_id", "zone_label", "distance_m"]]
