"""Deterministic encoding — the contract training and serving share.

The notebook used ``pd.get_dummies`` inline, which silently changes columns
whenever the data changes. Here every categorical has FIXED categories, so
a single passenger at serve time produces exactly the columns training saw.
"""

from __future__ import annotations

import pandas as pd

# Fixed vocabularies. drop_first=True drops the FIRST entry of each list.
CATEGORIES: dict[str, list[str]] = {
    "Sex": ["female", "male"],            # keeps Sex_male
    "Embarked": ["C", "Q", "S"],          # keeps Embarked_Q, Embarked_S
    "AgeBracket": ["Child", "Teen", "Adult", "Senior", "Elder"],
}

# Text/ID columns the model never sees (already used for cleaning/engineering).
DROP_COLUMNS = ["PassengerId", "Name", "Ticket", "Title", "Age", "SibSp", "Parch"]
# Only these two are scaled; the rest are 0/1 dummies or small ints.
NUMERIC_FEATURES = ["Fare", "family_size"]


def encode(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers/text, one-hot fixed categoricals, return numeric matrix (no target)."""
    # Drop IDs/text first; `if c in frame` keeps this working for one-row API input.
    work = frame.drop(columns=[c for c in DROP_COLUMNS if c in frame.columns]).copy()
    # Lock each column to its fixed vocabulary so train and serve match.
    for col, cats in CATEGORIES.items():
        work[col] = pd.Categorical(work[col], categories=cats)
    # drop_first avoids duplicate columns (e.g. female vs male); float for sklearn.
    return pd.get_dummies(work, columns=list(CATEGORIES), drop_first=True).astype(float)


def feature_names() -> list[str]:
    """Column order of a fully-populated encode — single source of truth."""
    # Fake one row with every category present, then read back the column order.
    probe = pd.DataFrame(
        [{c: cats[-1] for c, cats in CATEGORIES.items()}
         | {"Pclass": 1, "Fare": 0.0, "family_size": 1, "IsAlone": 0}]
    )
    return list(encode(probe).columns)


def align_single_row(row: dict) -> pd.DataFrame:
    """Encode one passenger dict and align to training columns (missing -> 0)."""
    encoded = encode(pd.DataFrame([row]))
    # Missing dummy columns (e.g. no Embarked_Q) become 0.0.
    return encoded.reindex(columns=feature_names(), fill_value=0.0)
