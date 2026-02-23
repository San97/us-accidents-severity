"""
data_ingestion.py
-----------------
Stage 1 of the pipeline: load the raw CSV and drop columns that are
either post-accident leakage or carry zero predictive signal.

Output: data/interim/ingested.csv
"""

import argparse
import logging
import sys

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
# Columns to drop — decisions made during EDA (see reference/Final_Code.ipynb)
# ---------------------------------------------------------------------------

# Post-accident information: collected after the event, therefore data leakage.
# A model predicting severity cannot use information that only exists after the
# fact.
LEAKAGE_COLS = [
    "ID",               # Row identifier, not a feature
    "Description",      # Free-text, post-accident narrative
    "Distance(mi)",     # Length of road impact — only known after accident clears
    "End_Time",         # When the accident cleared — post-accident
    "End_Lat",          # End coordinates of traffic impact — post-accident
    "End_Lng",
]

# Single unique value across the entire dataset — zero variance means zero
# predictive signal. Confirmed during EDA: Country='US' only, Turning_Loop=False only.
SINGLE_VALUE_COLS = [
    "Country",
    "Turning_Loop",
]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_data(input_path: str) -> pd.DataFrame:
    """Load raw CSV from disk and return as a DataFrame."""
    logger.info("Loading raw data from: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    logger.info("Loaded dataset — shape: %s", df.shape)
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leakage and single-value columns."""
    cols_to_drop = LEAKAGE_COLS + SINGLE_VALUE_COLS

    # Only drop columns that actually exist — guards against schema changes
    existing = [c for c in cols_to_drop if c in df.columns]
    missing = set(cols_to_drop) - set(existing)

    if missing:
        logger.warning("Expected columns not found in dataset: %s", missing)

    df = df.drop(columns=existing)
    logger.info("Dropped %d columns: %s", len(existing), existing)
    logger.info("Dataset shape after dropping: %s", df.shape)
    return df


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """Save the processed DataFrame to CSV."""
    df.to_csv(output_path, index=False)
    logger.info("Saved ingested data to: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_path: str, output_path: str) -> None:
    df = load_data(input_path)
    df = drop_columns(df)
    save_data(df, output_path)
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest raw accident data.")
    parser.add_argument("--input", required=True, help="Path to raw CSV file")
    parser.add_argument("--output", required=True, help="Path to save ingested CSV")
    args = parser.parse_args()

    main(args.input, args.output)
