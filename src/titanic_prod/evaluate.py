"""Evaluate a trained bundle: accuracy + report + confusion matrix, printed and saved."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import AppConfig, load_config
from .display import show_confusion, show_metrics
from .exceptions import ServingError
from .features import NUMERIC_FEATURES, encode
from .logging_setup import get_logger
from .metrics import compute_metrics

log = get_logger(__name__)


def run(config_path: str | Path, artifacts_dir=None) -> dict:
    # Rebuild the SAME test split as train.py (same seed + same processed CSV).
    cfg: AppConfig = load_config(config_path)
    artifacts = Path(artifacts_dir) if artifacts_dir else cfg.artifacts_dir
    bundle = [artifacts / "model.pkl", artifacts / "scaler.pkl", artifacts / "features.json"]
    if not all(p.is_file() for p in bundle):
        raise ServingError(f"Model bundle incomplete in {artifacts} — run train first.")

    # Load the saved bundle (never retrain here).
    model = joblib.load(bundle[0])
    scaler = joblib.load(bundle[1])
    cols = json.loads(bundle[2].read_text(encoding="utf-8"))

    df = pd.read_csv(cfg.processed_csv)
    X = encode(df.drop(columns=[cfg.target]))
    X = X.reindex(columns=cols, fill_value=0.0)  # same column order as training
    y = df[cfg.target]
    strat = y if cfg.split.stratify else None
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=cfg.split.test_size, stratify=strat, random_state=cfg.seed
    )
    # Scale test with the TRAIN-fitted scaler (no refit = no leakage).
    Xt = X_test.copy()
    Xt[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES])
    pred = model.predict(Xt)
    proba = model.predict_proba(Xt)[:, 1] if hasattr(model, "predict_proba") else None

    # Same formulas as train.py so numbers are comparable; saved for CI/DVC.
    metrics = compute_metrics(y_test, pred, proba)
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    show_metrics("Evaluation", metrics)  # rich scorecard in the terminal
    show_confusion(metrics["confusion_matrix"])  # rich 2x2 table
    return metrics
