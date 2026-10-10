"""Load, validate, clean, engineer — the Session 5 wire, as functions.

Notebook cells become named, testable units. Each function takes a
DataFrame and returns one, so the pipeline reads top to bottom.
"""

from __future__ import annotations

import re

import pandas as pd

from .exceptions import DataError
from .logging_setup import get_logger

log = get_logger(__name__)

# Columns we expect in the class CSV. Checked right after loading.
REQUIRED_COLUMNS = {
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
}

# Rare name titles grouped together so Age imputation has enough rows.
TITLE_MAP = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Rare",
    "Countess": "Rare", "Capt": "Rare", "Col": "Rare", "Don": "Rare",
    "Dr": "Rare", "Major": "Rare", "Rev": "Rare", "Sir": "Rare",
    "Jonkheer": "Rare", "Dona": "Rare",
}

# Age brackets used for both training and the API. Keep in one place.
AGE_BINS = [0, 12, 18, 35, 60, 200]
AGE_LABELS = ["Child", "Teen", "Adult", "Senior", "Elder"]


def load_raw(path) -> pd.DataFrame:
    # Read the CSV and fail fast with a clear error if columns are missing.
    path = str(path)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise DataError(f"Raw CSV not found: {path}") from exc
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataError(f"Raw CSV is missing columns: {sorted(missing)}")
    log.info("Loaded raw data: %s rows x %s cols", *df.shape)
    return df


def extract_title(name: str) -> str:
    # "Braund, Mr. Owen Harris" -> "Mr". Unknown format -> "Rare".
    match = re.search(r",\s([^.]+)\.", str(name))
    return match.group(1).strip() if match else "Rare"


def remove_outliers_iqr(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    # Keep only rows inside the 1.5 * IQR fence (standard outlier rule).
    q1, q3 = frame[col].quantile(0.25), frame[col].quantile(0.75)
    iqr = q3 - q1
    return frame[(frame[col] >= q1 - 1.5 * iqr) & (frame[col] <= q3 + 1.5 * iqr)]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Inspect-driven fixes: Embarked mode, drop Cabin, Title->Age median, Fare IQR."""
    df = df.copy()  # never mutate the caller's DataFrame
    # Most passengers embarked at S, so fill the 2 missing values with the mode.
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    # Cabin is mostly empty, so drop it instead of guessing.
    df = df.drop(columns=["Cabin"])
    # Age depends on title (Master = child, Mr = adult), so fill per-title median.
    df["Title"] = df["Name"].map(extract_title).replace(TITLE_MAP)
    df["Age"] = df.groupby("Title")["Age"].transform("median")
    if df["Age"].isna().any():
        raise DataError("Age still has missing values after Title-median imputation")
    before = len(df)
    # Drop extreme fares so one luxury ticket does not skew the model.
    df = remove_outliers_iqr(df, "Fare")
    log.info("Clean: %d -> %d rows (Fare IQR fence)", before, len(df))
    return df


def age_bracket(age: float) -> str:
    """Single-value version of the AgeBracket rule — reused by the API."""
    for lo, hi, label in zip(AGE_BINS[:-1], AGE_BINS[1:], AGE_LABELS):
        if lo < age <= hi or (lo == 0 and age == 0):
            return label
    return "Adult"  # safety net for out-of-range ages


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """family_size / IsAlone / AgeBracket — raw counts are weak alone."""
    df = df.copy()
    # Count the passenger themselves (+1) so singles get family_size == 1.
    df["family_size"] = df["SibSp"] + df["Parch"] + 1
    # Binary flag: travelling alone or not.
    df["IsAlone"] = (df["family_size"] == 1).astype(int)
    # Bucketed age, easier for the model than raw years.
    df["AgeBracket"] = df["Age"].map(age_bracket)
    return df
