"""THE CONTRACT: the canonical tracking DataFrame and its validator.

This is the single most important module in the repo. Read the design
rule in ``ftm/__init__.py`` before touching anything here.

Every loader must emit a DataFrame with exactly these columns, these
dtypes, and coordinates in these ranges. Every metric assumes it. When
Track A output arrives later, it will be handed to ``validate()`` and
either pass (and then run through every metric untouched) or fail loudly.

Canonical columns
-----------------
frame_id       int64     Monotonic within a period. NOT globally unique
                         across periods — pair it with ``period``.
period         int8      1 or 2.
timestamp      float64   Seconds since the start of the period.
track_id       string    Stable player identifier. The ball also gets one.
team           category  One of {"home", "away", "ball"}.
jersey_number  Int16     Nullable (pandas nullable int). Broadcast
                         providers frequently lack it.
x_pitch        float64   Meters. Range roughly -52.5 .. +52.5
                         (105 m pitch, origin at the center circle).
y_pitch        float64   Meters. Range roughly -34 .. +34 (68 m pitch).
is_ball        bool      True only for the ball row(s).
is_gk         bool      True for goalkeepers. ADDED vs. the program plan:
                         nearly every shape metric needs the GK excluded,
                         and without this column your defensive line
                         height is just the keeper.
ball_state     category  {"alive", "dead"}. ADDED vs. the program plan:
                         without it, average compactness is dominated by
                         throw-in and free-kick queues.

Orientation & coordinate conventions (must hold AFTER the loader runs)
---------------------------------------------------------------------
* Origin at pitch center; +x toward one fixed goal for the whole match
  (kloppy ``STATIC_HOME_AWAY``). Teams do NOT visually switch ends in the
  data even though they do on the pitch at half time.
* Units are meters (kloppy ``secondspectrum`` coordinate system), never
  normalized [0, 1].
* "Own goal" direction per team is a convention the shape metrics encode;
  document it in ``metrics/shape.py`` and keep it consistent.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

# ---------------------------------------------------------------------------
# Column spec. Keep this list and the dtype map as the single source of
# truth; tests import them directly.
# ---------------------------------------------------------------------------

COLUMNS: Final[list[str]] = [
    "frame_id",
    "period",
    "timestamp",
    "track_id",
    "team",
    "jersey_number",
    "x_pitch",
    "y_pitch",
    "is_ball",
    "is_gk",
    "ball_state",
]

# Pandas dtype string for each column. Used by ``validate()`` and by
# loaders that want to coerce a frame into shape before validating.
DTYPES: Final[dict[str, str]] = {
    "frame_id": "int64",
    "period": "int8",
    "timestamp": "float64",
    "track_id": "string",
    "team": "category",
    "jersey_number": "Int16",
    "x_pitch": "float64",
    "y_pitch": "float64",
    "is_ball": "bool",
    "is_gk": "bool",
    "ball_state": "category",
}

# Pitch dimensions in meters. The coordinate ranges are half these plus a
# small tolerance for players stepping off the pitch.
PITCH_LENGTH_M: Final[float] = 105.0
PITCH_WIDTH_M: Final[float] = 68.0
COORD_TOLERANCE_M: Final[float] = 5.0  # players/ball can briefly exceed the lines

TEAM_CATEGORIES: Final[tuple[str, ...]] = ("home", "away", "ball")
BALL_STATE_CATEGORIES: Final[tuple[str, ...]] = ("alive", "dead")


class SchemaError(ValueError):
    """Raised by ``validate()`` when a frame violates the contract.

    Deliberately a subclass of ``ValueError`` so callers can catch it
    specifically or fall back to catching ``ValueError``. ``validate()``
    RAISES this — it must never merely warn — so a malformed frame can
    never reach the metrics layer.
    """

def validate(df: pd.DataFrame, *, strict_ranges: bool = True) -> pd.DataFrame:
    """Assert that ``df`` is a valid canonical tracking frame.

    This is the gate every loader's output and every future Track A
    DataFrame must pass. On success it returns the same frame (so it can
    be used inline: ``df = validate(loader.load(3))``). On any violation
    it raises :class:`SchemaError` describing the first problem found.

    Checks to implement, roughly in this order:

    1. Columns. Exactly ``COLUMNS`` present, no extras, no missing. Order
       does not matter for correctness but reindex to ``COLUMNS`` before
       returning so downstream code can rely on it.
    2. Dtypes. Each column matches ``DTYPES``. Report the column name and
       both dtypes in the message. Categories must also have the right
       set of categories (``TEAM_CATEGORIES`` etc.), not just be
       ``category`` dtype.
    3. Null rules. ``jersey_number`` MAY be null. Nothing else may:
       ``frame_id``, ``period``, ``timestamp``, ``track_id``, ``team``,
       ``x_pitch``, ``y_pitch``, ``is_ball``, ``is_gk``, ``ball_state``
       must be fully populated.
    4. Value domains. ``period`` in {1, 2}. ``team`` in
       ``TEAM_CATEGORIES``. ``ball_state`` in ``BALL_STATE_CATEGORIES``.
    5. Coordinate ranges (only when ``strict_ranges``). ``x_pitch`` within
       +/-(PITCH_LENGTH_M / 2 + COORD_TOLERANCE_M); ``y_pitch`` within
       +/-(PITCH_WIDTH_M / 2 + COORD_TOLERANCE_M). If ``x_pitch`` is
       actually in [0, 1] the loader forgot the coordinate transform —
       call that out explicitly in the message, it is the classic bug.
    6. Uniqueness. No ``(period, frame_id, track_id)`` triple repeats.
       (The plan says ``(frame_id, track_id)``; include ``period``
       because ``frame_id`` only resets per period.)
    7. Ball/flag consistency. Rows with ``team == "ball"`` have
       ``is_ball == True`` and vice versa; ball rows have
       ``is_gk == False``; ``is_gk`` is only ever True for
       ``team in {"home", "away"}``.
    8. Monotonic frames. Within each period, ``frame_id`` is
       non-decreasing when the frame is sorted by ``timestamp``.

    Parameters
    ----------
    df:
        Candidate canonical frame.
    strict_ranges:
        When False, skip check 5. Useful for unit tests that build tiny
        hand-placed formations outside normal coordinates.

    Returns
    -------
    pandas.DataFrame
        The same rows, columns reordered to ``COLUMNS``.

    Raises
    ------
    SchemaError
        On the first violated rule.
    """
    df_columns = set(df.columns)
    original_columns = set(COLUMNS)

    missing = original_columns - df_columns
    extra = df_columns - original_columns

    if missing:
        raise SchemaError(f"Missing canonical columns: {sorted(missing)}")
    if extra:
        raise SchemaError(f"Unexpected extra columns: {sorted(extra, key=str)}")

    df = df[COLUMNS]

    for column in COLUMNS:
        actual_dtype = str(df[column].dtype)
        expected_dtype = DTYPES[column]

        if actual_dtype != expected_dtype:
            raise SchemaError(
                f"Column {column} has dtype {actual_dtype}, expected {expected_dtype}"
            )

    if set(df["team"].cat.categories) != set(TEAM_CATEGORIES):
        raise SchemaError(
            f"team categories are {set(df['team'].cat.categories)}, "
            f"expected {set(TEAM_CATEGORIES)}"
        )

    if set(df['ball_state'].cat.categories) != set(BALL_STATE_CATEGORIES):
        raise SchemaError(
            f"ball_state categories are {set(df['ball_state'].cat.categories)}, "
            f"expected {set(BALL_STATE_CATEGORIES)}"
        )

    for column in df.columns:
        if column == "jersey_number":
            continue
        if df[column].isnull().any():
            raise SchemaError(
                f"The column {column} contains NULL values but must be fully populated."
            )

    if not df["period"].isin([1,2]).all():
        raise SchemaError("The column period must contain value from [1,2]")

    if strict_ranges:
        x_bound = (PITCH_LENGTH_M / 2) + COORD_TOLERANCE_M
        y_bound = (PITCH_WIDTH_M / 2) + COORD_TOLERANCE_M

        if (df["x_pitch"] < -x_bound).any() or (df['x_pitch'] > x_bound).any():
            raise SchemaError(f"x_pitch must be within range of -{x_bound} and {x_bound}")

        if (df["y_pitch"] < -y_bound).any() or (df['y_pitch'] > y_bound).any():
            raise SchemaError(f"y_pitch must be within range of -{y_bound} and {y_bound}")

        if len(df) and df["x_pitch"].between(0, 1).all():
            raise SchemaError(
                "x_pitch appears normalized to [0,1] — did you forget the coordinate transform?"
            )

    duplicates = df.duplicated(subset=["period", "frame_id", "track_id"])
    if duplicates.any():
        n = duplicates.sum()
        raise SchemaError(
            f"Found {n} duplicate (period, frame_id, track_id) rows."
        )

    team_is_ball = df['team'] == 'ball'
    if not(team_is_ball == df['is_ball']).all():
        raise SchemaError("team==ball and is_ball flag are inconsistent for some rows.")

    bad_gk = df["is_gk"] & (df["team"] == "ball")
    if bad_gk.any():
        raise SchemaError("Found ball rows incorrectly flagged is_gk as True.")

    df_sorted = df.sort_values(["period", "timestamp"])
    df_sorted = df_sorted.drop_duplicates(["period", "frame_id", "timestamp"])
    if (0 > df_sorted.groupby("period")["frame_id"].diff()).any():
        raise SchemaError("Frames must be monotonic within each period.")

    return df


def empty_frame() -> pd.DataFrame:
    """Return a zero-row DataFrame with the canonical columns and dtypes.

    Handy for loaders (start empty, ``concat`` provider rows in) and for
    tests. The result must itself pass ``validate(..., strict_ranges=False)``.
    """
    columns = {col: pd.Series(dtype=DTYPES[col]) for col in COLUMNS}
    df = pd.DataFrame(columns)

    team_type = pd.CategoricalDtype(categories=TEAM_CATEGORIES)
    df["team"] = df["team"].astype(team_type)

    ball_type = pd.CategoricalDtype(categories=BALL_STATE_CATEGORIES)
    df["ball_state"] = df["ball_state"].astype(ball_type)

    return df


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort cast of a nearly-canonical frame to the exact dtypes.

    Loaders typically build a plain-dtype DataFrame and then call this to
    apply ``DTYPES`` (set categoricals with the full category list, cast
    ``jersey_number`` to ``Int16``, etc.) before handing off to
    ``validate()``. This does NOT fix semantic problems (bad ranges,
    duplicates) — only dtype presentation.
    """
    df = df.copy()
    for column,types in DTYPES.items():
        if column == "team":
            team_type = pd.CategoricalDtype(categories=TEAM_CATEGORIES)
            df[column] = df[column].astype(team_type)
        elif column == "ball_state":
            ball_type = pd.CategoricalDtype(categories=BALL_STATE_CATEGORIES)
            df[column] = df[column].astype(ball_type)
        else:
            df[column] = df[column].astype(types)

    return df
