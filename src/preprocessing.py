"""
preprocessing.py
----------------
Stage 2 of the pipeline: clean and prepare the ingested data.

Operations (in order):
  1. Standardize Wind_Direction values
  2. Engineer weather boolean features from Weather_Condition
  3. Extract datetime features from Start_Time
  4. Drop Weather_Timestamp
  5. Handle missing values
  6. Save to data/interim/preprocessed.csv

Input:  data/interim/ingested.csv
Output: data/interim/preprocessed.csv
"""

import argparse
import logging

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section 1: Wind Direction — normalization mapping
#
# The raw dataset has 24 distinct values due to:
#   (a) full-word spellings: 'North', 'South', 'East', 'West'
#   (b) secondary compass points: 'NNE', 'NNW', 'ENE', 'ESE', 'SSE', 'SSW', 'WNW', 'WSW'
#   (c) mixed case: 'Calm' vs 'CALM', 'Variable' vs 'VAR'
#
# Strategy: collapse to 8 directions (N, NE, E, SE, S, SW, W, NW) + CALM + VAR.
# Each cardinal direction absorbs its two adjacent secondary compass points:
#   N absorbs NNE, NNW  |  E absorbs ENE, ESE
#   S absorbs SSE, SSW  |  W absorbs WSW, WNW
# The 4 ordinal directions (NE, SE, SW, NW) are kept as-is.
# ---------------------------------------------------------------------------
WIND_DIRECTION_MAP = {
    # Full-word spellings → abbreviation
    "North":    "N",
    "South":    "S",
    "East":     "E",
    "West":     "W",
    "Calm":     "CALM",
    "Variable": "VAR",
    # Secondary compass points → nearest cardinal direction
    "NNE":      "N",
    "NNW":      "N",
    "ENE":      "E",
    "ESE":      "E",
    "SSE":      "S",
    "SSW":      "S",
    "WNW":      "W",
    "WSW":      "W",
}

# All valid values after cleanup — used to detect unexpected entries
VALID_WIND_DIRECTIONS = {"N", "NE", "E", "SE", "S", "SW", "W", "NW", "CALM", "VAR"}

# ---------------------------------------------------------------------------
# Section 2: Weather Condition — boolean feature patterns
#
# 144 unique raw values are collapsed into 7 binary features using substring
# matching. Patterns are case-insensitive regex strings.
#
# Note: features are NOT mutually exclusive — a row with 'Heavy Rain / Windy'
# will have both Rain=True AND Heavy_Rain=True. This is intentional; the model
# treats each as an independent signal.
#
# Keyword choices follow Road Weather Management Program research on weather
# conditions most correlated with accident severity.
# ---------------------------------------------------------------------------
WEATHER_PATTERNS = {
    "Clear":      r"Clear",
    "Cloud":      r"Cloud|Overcast",
    "Rain":       r"Rain|storm",
    "Heavy_Rain": r"Heavy Rain|Rain Shower|Heavy T-Storm|Heavy Thunderstorms",
    "Snow":       r"Snow|Sleet|Ice",
    "Heavy_Snow": r"Heavy Snow|Heavy Sleet|Heavy Ice Pellets|Snow Showers|Squalls",
    "Fog":        r"Fog",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_data(input_path: str) -> pd.DataFrame:
    """Load the ingested CSV from disk."""
    logger.info("Loading ingested data from: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    logger.info("Loaded dataset — shape: %s", df.shape)
    return df


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """Save the preprocessed DataFrame to CSV."""
    df.to_csv(output_path, index=False)
    logger.info("Saved preprocessed data to: %s", output_path)


# ---------------------------------------------------------------------------
# Section 1: Wind Direction cleanup
# ---------------------------------------------------------------------------

def clean_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Wind_Direction to consistent abbreviations."""
    if "Wind_Direction" not in df.columns:
        logger.warning("Wind_Direction column not found — skipping cleanup.")
        return df

    before_unique = set(df["Wind_Direction"].dropna().unique())
    df["Wind_Direction"] = df["Wind_Direction"].replace(WIND_DIRECTION_MAP)
    after_unique = set(df["Wind_Direction"].dropna().unique())

    # Detect any values not in the expected final set — signals schema drift in new data delivery
    unexpected = after_unique - VALID_WIND_DIRECTIONS
    if unexpected:
        counts = df["Wind_Direction"].value_counts()
        logger.warning(
            "Unexpected Wind_Direction values found (possible data drift): %s",
            {v: counts[v] for v in unexpected if v in counts}
        )

    logger.info(
        "Wind_Direction: standardized %d raw values → %d unique values",
        len(before_unique),
        len(after_unique),
    )
    return df


# ---------------------------------------------------------------------------
# Section 2: Weather boolean features
# ---------------------------------------------------------------------------

def engineer_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 7 binary columns from Weather_Condition using substring matching.
    The original Weather_Condition column is retained here for imputation use
    in Section 5 and dropped later in feature_engineering.py.
    """
    if "Weather_Condition" not in df.columns:
        logger.warning("Weather_Condition column not found — skipping weather feature engineering.")
        return df

    total_rows = len(df)
    for feature, pattern in WEATHER_PATTERNS.items():
        df[feature] = df["Weather_Condition"].str.contains(
            pattern, case=False, na=False
        )
        pct = df[feature].sum() / total_rows * 100
        logger.info("  %-12s → %7.2f%% of rows", feature, pct)

    logger.info(
        "Weather features created: %s",
        list(WEATHER_PATTERNS.keys()),
    )
    return df


# ---------------------------------------------------------------------------
# Section 3: Datetime feature extraction + drop Weather_Timestamp
#
# Raw datetime strings are unusable by ML models. We decompose Start_Time
# into numeric components that carry genuine predictive signal (seasonality,
# rush-hour patterns, weekday vs weekend behaviour).
#
# Month is extracted here intentionally early — it is required as a groupby
# key in the imputation step (Section 5).
#
# Weather_Timestamp is dropped: it records when weather was observed at the
# nearest station, which lags Start_Time by ~90 seconds on average. It adds
# no information that Start_Time doesn't already provide.
#
# Note on cyclical encoding: Hour, Month, and Weekday are cyclical by nature
# (Hour 23 is adjacent to Hour 0). Sine/cosine encoding would be the correct
# treatment for linear models and neural networks. For the tree-based models
# in this project (RF, GradientBoosting, XGBoost), raw integers are sufficient
# as trees can approximate cycles via threshold splits at both ends of the range.
# ---------------------------------------------------------------------------

def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from Start_Time and drop Weather_Timestamp."""
    if "Start_Time" not in df.columns:
        logger.warning("Start_Time column not found — skipping datetime extraction.")
        return df

    # format='ISO8601' handles mixed precision in the same column
    # (some rows have nanoseconds appended: "2016-02-08 05:46:00.000000000")
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], format="ISO8601")

    df["Year"]    = df["Start_Time"].dt.year
    df["Month"]   = df["Start_Time"].dt.month
    df["Weekday"] = df["Start_Time"].dt.weekday   # 0=Monday, 6=Sunday
    df["Day"]     = df["Start_Time"].dt.day
    df["Hour"]    = df["Start_Time"].dt.hour
    df["Minute"]  = df["Start_Time"].dt.minute

    logger.info(
        "Datetime features extracted: Year, Month, Weekday, Day, Hour, Minute"
    )
    logger.info(
        "Year range: %d – %d", df["Year"].min(), df["Year"].max()
    )

    if "Weather_Timestamp" in df.columns:
        df = df.drop(columns=["Weather_Timestamp"])
        logger.info("Dropped Weather_Timestamp (redundant — avg ~90s lag from Start_Time)")

    return df


# ---------------------------------------------------------------------------
# Section 4: Missing value handling
#
# Strategy (in order):
#
#   4a. Drop rows where low-missing categorical columns are null.
#       These columns have < 0.30% nulls — losing those rows is negligible.
#       Airport_Code must be dropped first: it is the groupby key for
#       imputation below. A null Airport_Code means we cannot impute anything
#       for that row.
#
#   4b. Drop rows where Wind_Chill(F) or Precipitation(in) are null.
#       Both have ~25-29% missingness — too high to impute reliably.
#       Wind_Chill is a derived value (from Temperature + Wind_Speed) that
#       weather stations often omit in warm conditions. Precipitation is
#       frequently unreported at stations. Dropping these rows follows the
#       original analysis decision.
#
#   4c. Impute continuous weather columns with median grouped by
#       Airport_Code + Month. Uses transform() for efficiency and correct
#       index alignment — improvement over the notebook's apply().
#       Temperature(F) is added to the imputation list (notebook omitted it
#       despite the same ~2% missingness pattern).
#
#   4d. Impute Wind_Direction with mode grouped by Airport_Code + Month.
#
#   4e. Final dropna — catches any rows where an entire Airport+Month group
#       had no valid values to impute from.
# ---------------------------------------------------------------------------

# 4a: columns where nulls are rare enough to simply drop the rows
DROP_NULL_COLS = [
    "Airport_Code",           # groupby key — must be non-null before imputation
    "City", "Zipcode", "Timezone",
    "Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight",
]

# 4b: columns dropped explicitly due to high/structural missingness
DROP_HIGH_NULL_COLS = ["Wind_Chill(F)", "Precipitation(in)"]

# 4c: continuous weather columns imputed with group median
IMPUTE_NUMERIC_COLS = [
    "Temperature(F)",  # ~2.1% null — added vs notebook for consistency
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
]


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the three-step missing value strategy:
    drop negligible nulls → impute numeric → impute categorical.
    """
    start_rows = len(df)
    logger.info("Missing value handling — starting rows: %s", f"{start_rows:,}")

    # ------------------------------------------------------------------
    # 4a: Drop rows with nulls in low-missing columns
    # ------------------------------------------------------------------
    existing_drop = [c for c in DROP_NULL_COLS if c in df.columns]
    df = df.dropna(subset=existing_drop)
    logger.info(
        "  After dropping null rows (%s): %s rows (lost %s)",
        existing_drop,
        f"{len(df):,}",
        f"{start_rows - len(df):,}",
    )

    # ------------------------------------------------------------------
    # 4b: Drop rows with null Wind_Chill(F) and Precipitation(in)
    # ------------------------------------------------------------------
    existing_high = [c for c in DROP_HIGH_NULL_COLS if c in df.columns]
    before = len(df)
    df = df.dropna(subset=existing_high)
    logger.info(
        "  After dropping null Wind_Chill / Precipitation: %s rows (lost %s)",
        f"{len(df):,}",
        f"{before - len(df):,}",
    )

    # ------------------------------------------------------------------
    # 4c: Impute continuous weather with median by Airport_Code + Month
    # ------------------------------------------------------------------
    existing_numeric = [c for c in IMPUTE_NUMERIC_COLS if c in df.columns]
    for col in existing_numeric:
        null_before = df[col].isnull().sum()
        df[col] = df.groupby(["Airport_Code", "Month"])[col].transform(
            lambda x: x.fillna(x.median()) if x.notna().any() else x
        )
        null_after = df[col].isnull().sum()
        logger.info(
            "  Imputed %-20s  nulls: %s → %s",
            col, f"{null_before:,}", f"{null_after:,}",
        )

    # Drop rows where imputation failed (entire Airport+Month group was null)
    before = len(df)
    df = df.dropna(subset=existing_numeric)
    if before > len(df):
        logger.warning(
            "  Dropped %s rows where entire Airport+Month group had no valid values to impute from.",
            f"{before - len(df):,}",
        )

    # ------------------------------------------------------------------
    # 4d: Impute Wind_Direction with mode by Airport_Code + Month
    # ------------------------------------------------------------------
    if "Wind_Direction" in df.columns and df["Wind_Direction"].isnull().any():
        null_before = df["Wind_Direction"].isnull().sum()
        df["Wind_Direction"] = df.groupby(["Airport_Code", "Month"])["Wind_Direction"].transform(
            lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x
        )
        logger.info(
            "  Imputed Wind_Direction nulls: %s → %s",
            f"{null_before:,}", f"{df['Wind_Direction'].isnull().sum():,}",
        )

    # ------------------------------------------------------------------
    # 4e: Final dropna — remove any rows still containing nulls
    # ------------------------------------------------------------------
    before = len(df)
    df = df.dropna()
    logger.info(
        "  Final dropna: removed %s remaining rows with any nulls",
        f"{before - len(df):,}",
    )

    logger.info(
        "Missing value handling complete — final rows: %s (total lost: %s)",
        f"{len(df):,}",
        f"{start_rows - len(df):,}",
    )
    return df


# ---------------------------------------------------------------------------
# Main — orchestrates all sections in the correct order
# ---------------------------------------------------------------------------

def main(input_path: str, output_path: str) -> None:
    """Run the full preprocessing pipeline."""
    df = load_data(input_path)

    # Section 1 — must happen before imputation (Wind_Direction is a groupby target)
    df = clean_wind_direction(df)

    # Section 2 — creates weather boolean features from Weather_Condition
    df = engineer_weather_features(df)

    # Section 3 — extracts Month (needed as groupby key in Section 4)
    df = extract_datetime_features(df)

    # Section 4 — drop, impute, and finalize missing values
    df = handle_missing_values(df)

    logger.info("Final dataset shape: %s", df.shape)
    save_data(df, output_path)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ingested accident data.")
    parser.add_argument("--input",  required=True, help="Path to ingested CSV")
    parser.add_argument("--output", required=True, help="Path to save preprocessed CSV")
    args = parser.parse_args()

    main(args.input, args.output)
