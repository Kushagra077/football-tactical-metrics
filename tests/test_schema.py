"""Tests for the canonical schema contract (``ftm.schema``).

These lock the interface that Track A output will later have to satisfy,
so be thorough and specific in the assertions.
"""

from __future__ import annotations


def test_empty_frame_is_valid():
    """``schema.empty_frame()`` passes ``validate(strict_ranges=False)`` and
    has exactly ``schema.COLUMNS`` in order with dtypes matching
    ``schema.DTYPES``."""
    raise NotImplementedError


def test_valid_frame_passes_and_is_reordered():
    """A hand-built good frame (columns shuffled) passes and is returned
    with columns in ``schema.COLUMNS`` order."""
    raise NotImplementedError


def test_missing_column_raises():
    """Dropping any one canonical column makes ``validate`` raise
    ``SchemaError`` naming that column."""
    raise NotImplementedError


def test_wrong_dtype_raises():
    """``x_pitch`` as int, ``team`` as plain object, ``jersey_number`` as
    float64 — each raises ``SchemaError`` naming the column and both
    dtypes."""
    raise NotImplementedError


def test_normalized_coordinates_are_rejected_with_a_clear_message():
    """``x_pitch`` in [0, 1] raises, and the message explicitly mentions
    the missing coordinate transform (the classic loader bug)."""
    raise NotImplementedError


def test_out_of_range_coordinates_raise_when_strict():
    """``x_pitch`` = 200 raises with ``strict_ranges=True`` and passes
    with ``strict_ranges=False``."""
    raise NotImplementedError


def test_duplicate_period_frame_track_raises():
    """Two rows sharing (period, frame_id, track_id) -> ``SchemaError``."""
    raise NotImplementedError


def test_bad_category_values_raise():
    """``team`` == "HOME", ``period`` == 3, ``ball_state`` == "unknown" —
    each raises."""
    raise NotImplementedError


def test_jersey_number_may_be_null_but_others_may_not():
    """All-null ``jersey_number`` is fine; a single null in ``x_pitch`` or
    ``team`` raises."""
    raise NotImplementedError


def test_ball_flag_consistency():
    """``team == "ball"`` with ``is_ball == False`` (or vice versa)
    raises; ``is_gk == True`` on a ball row raises."""
    raise NotImplementedError


def test_frame_id_non_monotonic_within_period_raises():
    """Sorted by timestamp, a decreasing ``frame_id`` inside one period
    raises."""
    raise NotImplementedError


def test_coerce_applies_dtypes_without_fixing_semantics():
    """``schema.coerce`` turns a plain-dtype good frame into one that
    passes ``validate``; but ``coerce`` on a frame with duplicates still
    leaves ``validate`` raising."""
    raise NotImplementedError
