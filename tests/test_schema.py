"""Tests for the canonical schema contract (``ftm.schema``).

These lock the interface that Track A output will later have to satisfy,
so be thorough and specific in the assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ftm import schema
from ftm.schema import COLUMNS, DTYPES, SchemaError, coerce, empty_frame, validate
from tests.conftest import make_canonical_frame


def _good_frame(**kwargs) -> pd.DataFrame:
    params = {
        "n_frames": 3,
        "home_xy": {"h_gk": (-45.0, 0.0), "h1": (-10.0, 5.0)},
        "away_xy": {"a_gk": (45.0, 0.0), "a1": (10.0, -5.0)},
        "ball_xy": (0.0, 0.0),
    }
    params.update(kwargs)
    return make_canonical_frame(**params)


def _dtypes(df: pd.DataFrame) -> dict[str, str]:
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def test_empty_frame_is_valid():
    """``schema.empty_frame()`` passes ``validate(strict_ranges=False)`` and
    has exactly ``schema.COLUMNS`` in order with dtypes matching
    ``schema.DTYPES``."""
    df = empty_frame()
    assert len(df) == 0
    assert list(df.columns) == COLUMNS
    assert _dtypes(df) == DTYPES
    assert set(df["team"].cat.categories) == set(schema.TEAM_CATEGORIES)
    assert set(df["ball_state"].cat.categories) == set(schema.BALL_STATE_CATEGORIES)

    out = validate(df, strict_ranges=False)
    assert list(out.columns) == COLUMNS
    assert _dtypes(out) == DTYPES
    # No rows means no evidence of normalized coordinates either.
    validate(empty_frame(), strict_ranges=True)


def test_valid_frame_passes_and_is_reordered():
    """A hand-built good frame (columns shuffled) passes and is returned
    with columns in ``schema.COLUMNS`` order."""
    good = _good_frame()
    shuffled = good[list(reversed(COLUMNS))]
    assert list(shuffled.columns) != COLUMNS

    out = validate(shuffled)

    assert list(out.columns) == COLUMNS
    assert _dtypes(out) == DTYPES
    pd.testing.assert_frame_equal(out, good)


@pytest.mark.parametrize("column", COLUMNS)
def test_missing_column_raises(column):
    """Dropping any one canonical column makes ``validate`` raise
    ``SchemaError`` naming that column."""
    df = _good_frame().drop(columns=column)
    with pytest.raises(SchemaError, match=column):
        validate(df)


def test_extra_column_raises():
    df = _good_frame().assign(speed=1.0)
    with pytest.raises(SchemaError, match="speed"):
        validate(df)


@pytest.mark.parametrize(
    ("column", "bad_dtype", "shown_dtype"),
    [
        ("x_pitch", "int64", "int64"),
        ("team", object, "object"),
        ("jersey_number", "float64", "float64"),
    ],
)
def test_wrong_dtype_raises(column, bad_dtype, shown_dtype):
    """``x_pitch`` as int, ``team`` as plain object, ``jersey_number`` as
    float64 — each raises ``SchemaError`` naming the column and both
    dtypes."""
    df = _good_frame()
    df[column] = df[column].astype(bad_dtype)
    with pytest.raises(SchemaError) as excinfo:
        validate(df, strict_ranges=False)
    message = str(excinfo.value)
    assert column in message
    assert shown_dtype in message
    assert DTYPES[column] in message


def test_normalized_coordinates_are_rejected_with_a_clear_message():
    """``x_pitch`` in [0, 1] raises, and the message explicitly mentions
    the missing coordinate transform (the classic loader bug)."""
    df = _good_frame(
        home_xy={"h_gk": (0.05, 0.5), "h1": (0.4, 0.3)},
        away_xy={"a_gk": (0.95, 0.5), "a1": (0.6, 0.7)},
        ball_xy=(0.5, 0.5),
    )
    assert df["x_pitch"].between(0, 1).all()
    with pytest.raises(SchemaError, match="coordinate transform"):
        validate(df)


def test_out_of_range_coordinates_raise_when_strict():
    """``x_pitch`` = 200 raises with ``strict_ranges=True`` and passes
    with ``strict_ranges=False``."""
    df = _good_frame()
    df.loc[df["track_id"] == "h1", "x_pitch"] = 200.0
    with pytest.raises(SchemaError, match="x_pitch"):
        validate(df, strict_ranges=True)
    validate(df, strict_ranges=False)

    df = _good_frame()
    df.loc[df["track_id"] == "h1", "y_pitch"] = -200.0
    with pytest.raises(SchemaError, match="y_pitch"):
        validate(df, strict_ranges=True)
    validate(df, strict_ranges=False)


def test_duplicate_period_frame_track_raises():
    """Two rows sharing (period, frame_id, track_id) -> ``SchemaError``."""
    good = _good_frame()
    df = pd.concat([good, good.iloc[[1]]], ignore_index=True)
    assert _dtypes(df) == DTYPES
    with pytest.raises(SchemaError, match="duplicate"):
        validate(df)


def test_same_frame_and_track_in_other_period_is_not_a_duplicate():
    df = pd.concat([_good_frame(period=1), _good_frame(period=2)], ignore_index=True)
    validate(df)


def test_bad_category_values_raise():
    """``team`` == "HOME", ``period`` == 3, ``ball_state`` == "unknown" —
    each raises."""
    df = _good_frame()
    df["team"] = df["team"].cat.rename_categories({"home": "HOME"})
    with pytest.raises(SchemaError, match="team"):
        validate(df)

    with pytest.raises(SchemaError, match="period"):
        validate(_good_frame(period=3))

    df = _good_frame()
    df["ball_state"] = df["ball_state"].cat.rename_categories({"alive": "unknown"})
    with pytest.raises(SchemaError, match="ball_state"):
        validate(df)


def test_jersey_number_may_be_null_but_others_may_not():
    """All-null ``jersey_number`` is fine; a single null in ``x_pitch`` or
    ``team`` raises."""
    df = _good_frame()
    assert df["jersey_number"].isna().all()
    validate(df)

    for column in ("x_pitch", "team"):
        df = _good_frame()
        df.loc[0, column] = np.nan
        assert df[column].isna().sum() == 1
        assert _dtypes(df) == DTYPES
        with pytest.raises(SchemaError, match=column):
            validate(df)


def test_ball_flag_consistency():
    """``team == "ball"`` with ``is_ball == False`` (or vice versa)
    raises; ``is_gk == True`` on a ball row raises."""
    df = _good_frame()
    df.loc[df["team"] == "ball", "is_ball"] = False
    with pytest.raises(SchemaError, match="is_ball"):
        validate(df)

    df = _good_frame()
    df.loc[df["track_id"] == "h1", "is_ball"] = True
    with pytest.raises(SchemaError, match="is_ball"):
        validate(df)

    df = _good_frame()
    df.loc[df["team"] == "ball", "is_gk"] = True
    with pytest.raises(SchemaError, match="is_gk"):
        validate(df)


def test_frame_id_non_monotonic_within_period_raises():
    """Sorted by timestamp, a decreasing ``frame_id`` inside one period
    raises."""
    df = _good_frame()
    frame_ids = df["frame_id"].to_numpy().copy()
    frame_ids[df["frame_id"] == 1] = 2
    frame_ids[df["frame_id"] == 2] = 1
    df["frame_id"] = frame_ids.astype("int64")
    assert not df.duplicated(["period", "frame_id", "track_id"]).any()
    with pytest.raises(SchemaError, match="monotonic"):
        validate(df)


def test_coerce_applies_dtypes_without_fixing_semantics():
    """``schema.coerce`` turns a plain-dtype good frame into one that
    passes ``validate``; but ``coerce`` on a frame with duplicates still
    leaves ``validate`` raising."""
    good = _good_frame()
    plain = pd.DataFrame(good.to_dict("list"))
    assert _dtypes(plain) != DTYPES
    with pytest.raises(SchemaError):
        validate(plain)

    coerced = coerce(plain)
    assert _dtypes(coerced) == DTYPES
    pd.testing.assert_frame_equal(validate(coerced), good)
    assert _dtypes(plain) != DTYPES  # coerce does not mutate its input

    duplicated = pd.concat([plain, plain.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicate"):
        validate(coerce(duplicated))
