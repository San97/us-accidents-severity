"""
feature_engineering.py
----------------------
Stage 3 of the pipeline: transform cleaned data into model-ready features.

Operations (in order):
  1. Filter and balance Severity classes
  2. Drop high-cardinality and identifier columns
  3. Encode boolean and Day/Night columns to integers
  4. One-hot encode Timezone and Weekday
  5. Save to data/processed/features.csv

Input:  data/interim/preprocessed.csv
Output: data/processed/features.csv
"""

import argparse
import logging

import pandas as pd
import yaml

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
# Section 2: Column drops
#
# These columns are dropped after balancing because they are either:
#   (a) Identifiers with thousands of unique values — no generalisable signal
#       (Street, City, County, State, Zipcode, Airport_Code).
#   (b) Already encoded into boolean features — original column no longer needed
#       (Weather_Condition → Clear/Cloud/Rain/etc. in preprocessing.py).
#   (c) Already decomposed into numeric features — raw string unusable by models
#       (Start_Time → Year/Month/Day/Hour/Minute/Weekday in preprocessing.py).
#   (d) Already standardized into 10 values but still categorical string —
#       Wind_Direction: adding 10 dummies would be low-signal noise given the
#       weather boolean features already capture the relevant conditions.
# ---------------------------------------------------------------------------
DROP_COLS = [
    "Start_Time",
    "Street",
    "City",
    "County",
    "State",
    "Zipcode",
    "Airport_Code",
    "Wind_Direction",
    "Weather_Condition",
]

# ---------------------------------------------------------------------------
# Section 3: Encoding maps
#
# Day/Night columns: Sunrise_Sunset, Civil_Twilight, Nautical_Twilight,
# Astronomical_Twilight. Each has exactly two values: 'Day' and 'Night'.
# We encode Day=1, Night=0.
#
# Boolean columns (POI infrastructure features): Amenity, Bump, Crossing, etc.
# These come through as Python bool dtype after pd.read_csv. Encoding to int
# is required for all sklearn and XGBoost estimators.
#
# We do NOT use a global df.replace([True, False], [1, 0]) as the original
# notebook does — that touches every column blindly and can silently corrupt
# columns if their contents change in a future data delivery.
# ---------------------------------------------------------------------------
DAY_NIGHT_COLS = [
    "Sunrise_Sunset",
    "Civil_Twilight",
    "Nautical_Twilight",
    "Astronomical_Twilight",
]

# ---------------------------------------------------------------------------
# Section 4: One-hot encoding targets
#
# Timezone: 4 US timezone strings (e.g. 'US/Eastern') → 4 binary columns.
# Weekday:  integers 0–6 (Mon–Sun) — treated as categorical here to match
#           the original notebook design. One-hot adds 7 columns and lets the
#           model learn arbitrary per-day effects without assuming a linear
#           relationship across days of the week.
# ---------------------------------------------------------------------------
OHE_COLS = ["Timezone", "Weekday"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_data(input_path: str) -> pd.DataFrame:
    """Load the preprocessed CSV from disk."""
    logger.info("Loading preprocessed data from: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    logger.info("Loaded dataset — shape: %s", df.shape)
    return df


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """Save the feature-engineered DataFrame to CSV."""
    df.to_csv(output_path, index=False)
    logger.info("Saved features to: %s", output_path)


# ---------------------------------------------------------------------------
# Section 1: Class filtering and balancing
# ---------------------------------------------------------------------------

def filter_and_balance(df: pd.DataFrame, drop_severity: int, random_seed: int) -> pd.DataFrame:
    """
    Drop the excluded severity class, then undersample remaining classes
    to the size of the smallest class.

    Improvements over the original notebook:
      - random_state passed to .sample() → reproducible across runs
      - explicit shuffle + reset_index after concat → clean output CSV
      - class distribution logged before and after for observability
    """
    # Log class distribution before any changes
    counts_before = df["Severity"].value_counts().sort_index()
    logger.info("Severity distribution before balancing:")
    for sev, count in counts_before.items():
        logger.info("  Severity %d: %s rows", sev, f"{count:,}")

    # Drop the excluded severity class
    df = df[df["Severity"] != drop_severity].copy()
    logger.info("Dropped Severity=%d. Remaining rows: %s", drop_severity, f"{len(df):,}")

    # Find the target size: size of the smallest remaining class
    counts = df["Severity"].value_counts()
    n_per_class = int(counts.min())
    logger.info(
        "Undersampling each class to %s rows (min class: Severity %d)",
        f"{n_per_class:,}",
        int(counts.idxmin()),
    )

    # Sample n_per_class rows from each class with a fixed seed
    classes = sorted(df["Severity"].unique())
    sampled = pd.concat(
        [
            df[df["Severity"] == sev].sample(n=n_per_class, random_state=random_seed)
            for sev in classes
        ],
        axis=0,
    )

    # Shuffle rows and reset index so the saved CSV is in random order
    # (pd.concat produces all Severity=2 first, then 3, then 4)
    sampled = sampled.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    logger.info(
        "Balanced dataset — %d classes × %s rows = %s total rows",
        len(classes),
        f"{n_per_class:,}",
        f"{len(sampled):,}",
    )
    return sampled


# ---------------------------------------------------------------------------
# Section 2: Drop high-cardinality and identifier columns
# ---------------------------------------------------------------------------

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier and already-encoded columns that carry no model signal."""
    existing = [c for c in DROP_COLS if c in df.columns]
    missing = set(DROP_COLS) - set(existing)
    if missing:
        logger.warning("Expected columns not found (already dropped upstream?): %s", missing)

    df = df.drop(columns=existing)
    logger.info("Dropped %d columns: %s", len(existing), existing)
    logger.info("Shape after column drop: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Section 3: Encode booleans and Day/Night columns
# ---------------------------------------------------------------------------

def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert bool columns → int (0/1) and Day/Night strings → int (1/0).

    Uses explicit column targeting rather than a global df.replace() to
    avoid silently corrupting unrelated columns with coincidental True/False
    or Day/Night values in future data deliveries.
    """
    # 3a: Boolean dtype columns → int
    bool_cols = [col for col in df.columns if df[col].dtype == bool]
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        logger.info("Encoded %d bool columns to int: %s", len(bool_cols), bool_cols)

    # 3b: Day/Night string columns → 1/0
    existing_dn = [c for c in DAY_NIGHT_COLS if c in df.columns]
    if existing_dn:
        df[existing_dn] = df[existing_dn].apply(lambda col: col.map({"Day": 1, "Night": 0}))
        logger.info("Encoded Day/Night columns to 1/0: %s", existing_dn)

    return df


# ---------------------------------------------------------------------------
# Section 4: One-hot encode Timezone and Weekday
# ---------------------------------------------------------------------------

def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode Timezone and Weekday.

    dtype=int ensures the resulting columns are 0/1 integers, not bool.
    pandas 2.x changed get_dummies to return bool dtype by default — this
    explicit dtype=int guards against that regression.
    """
    existing_ohe = [c for c in OHE_COLS if c in df.columns]
    if not existing_ohe:
        logger.warning("No OHE columns found: %s", OHE_COLS)
        return df

    before_cols = df.shape[1]
    df = pd.get_dummies(df, columns=existing_ohe, dtype=int)
    added = df.shape[1] - before_cols + len(existing_ohe)  # net new columns
    logger.info(
        "One-hot encoded %s → added %d new columns. Shape: %s",
        existing_ohe,
        added,
        df.shape,
    )
    return df


# ---------------------------------------------------------------------------
# Main — orchestrates all sections in the correct order
# ---------------------------------------------------------------------------

def main(input_path: str, output_path: str, params_path: str) -> None:
    """Run the full feature engineering pipeline."""
    # Load params
    with open(params_path) as f:
        params = yaml.safe_load(f)

    random_seed  = params["feature_engineering"]["random_seed"]
    drop_severity = params["feature_engineering"]["drop_severity"]
    logger.info("Params — random_seed: %d, drop_severity: %d", random_seed, drop_severity)

    df = load_data(input_path)

    # Section 1 — filter + balance classes
    df = filter_and_balance(df, drop_severity=drop_severity, random_seed=random_seed)

    # Section 2 — drop identifier and already-encoded columns
    df = drop_columns(df)

    # Section 3 — encode booleans and Day/Night to integers
    df = encode_binary_columns(df)

    # Section 4 — one-hot encode Timezone and Weekday
    df = one_hot_encode(df)

    logger.info("Final feature matrix shape: %s", df.shape)
    save_data(df, output_path)
    logger.info("Feature engineering complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engineer features from preprocessed data.")
    parser.add_argument("--input",  required=True, help="Path to preprocessed CSV")
    parser.add_argument("--output", required=True, help="Path to save features CSV")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    main(args.input, args.output, args.params)
