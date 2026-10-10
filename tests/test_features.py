"""Feature wire on the real class CSV: clean -> engineer -> encode."""

from titanic_prod.config import REPO_ROOT
from titanic_prod.data import clean, engineer, load_raw
from titanic_prod.features import align_single_row, encode, feature_names


def _matrix():
    df = engineer(clean(load_raw(REPO_ROOT / "data" / "raw" / "Titanic-Dataset.csv")))
    return df, encode(df.drop(columns=["Survived"]))


def test_clean_removes_outliers_and_fills_age():
    df = clean(load_raw(REPO_ROOT / "data" / "raw" / "Titanic-Dataset.csv"))
    assert df["Age"].isna().sum() == 0
    assert 700 < len(df) < 891  # Fare IQR fence applied
    assert "Cabin" not in df.columns


def test_encode_is_deterministic_11_features():
    _, X = _matrix()
    assert X.shape[1] == 11  # Pclass/Fare/family_size/IsAlone + 1 + 2 + 4 dummies
    assert X.isna().sum().sum() == 0
    assert list(X.columns) == feature_names()


def test_single_row_aligns_to_training_columns():
    row = {
        "Pclass": 1, "Sex": "female", "Embarked": "C", "AgeBracket": "Adult",
        "Fare": 80.0, "family_size": 1, "IsAlone": 1,
    }
    aligned = align_single_row(row)
    assert list(aligned.columns) == feature_names()
    assert float(aligned["Sex_male"].iloc[0]) == 0.0
    assert float(aligned["IsAlone"].iloc[0]) == 1.0
