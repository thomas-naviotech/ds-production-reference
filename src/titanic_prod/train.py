"""Split-first training: fit everything on train, log everything to MLflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import __version__ as PIPELINE_VERSION
from .config import AppConfig, load_config
from .data import clean, engineer, load_raw
from .exceptions import TrainingError
from .features import NUMERIC_FEATURES, encode, feature_names
from .logging_setup import get_logger
from .metrics import compute_metrics

log = get_logger(__name__)

# Optional XGBoost: used only if installed (`uv pip install xgboost`).
try:
    from xgboost import XGBClassifier  # type: ignore

    _HAS_XGB = True
except ImportError:
    XGBClassifier = None  # type: ignore
    _HAS_XGB = False


def _make_svm(C: float = 1.0, cv: int = 5, random_state: int = 42):
    # Calibrated wrapper gives predict_proba (needed for ROC AUC + the API).
    # ensemble=False = one single model, simpler and faster for class.
    # (sklearn deprecated SVC(probability=True); this is its replacement.)
    return CalibratedClassifierCV(
        estimator=SVC(C=C, random_state=random_state), cv=cv, ensemble=False
    )


# Allowed model names in configs/*.yaml. Add a new entry to plug in a model.
MODEL_REGISTRY: dict = {
    "logreg": LogisticRegression,  # linear baseline
    "rf": RandomForestClassifier,  # bagging forest
    "svm": _make_svm,  # calibrated SVM (see factory above)
    "knn": KNeighborsClassifier,  # distance-based, loves the scaler below
    "nb": GaussianNB,  # Naive Bayes, no knobs
    "gb": GradientBoostingClassifier,  # boosting baseline
}
if _HAS_XGB:
    MODEL_REGISTRY["xgb"] = XGBClassifier  # gradient-boosted trees (optional dep)

# Sensible student defaults per model; YAML params override these.
DEFAULT_PARAMS: dict[str, dict] = {
    "logreg": {"max_iter": 1000},
    "rf": {"n_estimators": 300},
    "svm": {"C": 1.0, "cv": 5},
    "knn": {"n_neighbors": 5},
    "nb": {},
    "gb": {"n_estimators": 100},
    "xgb": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
            "subsample": 0.9, "eval_metric": "logloss"},
}

# Models that accept a `random_state` / `seed` for repeatable runs.
SEEDED_MODELS = {"logreg", "rf", "svm", "gb", "xgb"}


def _build_model(name: str, seed: int, extra: dict | None = None):
    # Merge defaults <- YAML so configs only list what they change.
    if name not in MODEL_REGISTRY:
        raise TrainingError(
            f"Unknown model '{name}' — choose from {sorted(MODEL_REGISTRY)}"
        )
    params = {**DEFAULT_PARAMS.get(name, {}), **(extra or {})}
    if name in SEEDED_MODELS and "random_state" not in params:
        # XGBoost uses `random_state` too (alias of `seed`).
        params["random_state"] = seed
    try:
        return MODEL_REGISTRY[name](**params), params
    except Exception as exc:
        raise TrainingError(f"Bad params for model '{name}': {exc}") from exc


def prepare_split(cfg: AppConfig):
    # Shared data prep: raw -> cleaned -> scaled train/test. One place so
    # train.py, compare.py, and tune.py all race on the identical split.
    # NOTE: scaler is fit on TRAIN only (no leakage), test is only transformed.
    df = engineer(clean(load_raw(cfg.raw_csv)))
    df.to_csv(cfg.processed_csv, index=False)  # cached table reused by evaluate.py
    X = encode(df.drop(columns=[cfg.target]))
    y = df[cfg.target]
    cols = feature_names()
    X = X[cols]
    # stratify keeps the survived ratio equal; seed makes the split repeatable.
    strat = y if cfg.split.stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.split.test_size, stratify=strat, random_state=cfg.seed
    )
    log.info("Split: %d train / %d test (stratified=%s)",
             len(X_train), len(X_test), cfg.split.stratify)
    scaler = StandardScaler().fit(X_train[NUMERIC_FEATURES])
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERIC_FEATURES] = scaler.transform(X_train[NUMERIC_FEATURES])
    X_test[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES])
    return X_train, X_test, y_train, y_test, cols, scaler


def _save_bundle(artifacts: Path, model, scaler, cols: list, metrics: dict) -> None:
    # The serving bundle: everything the API needs, plus the scorecard.
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts / "model.pkl")
    joblib.dump(scaler, artifacts / "scaler.pkl")
    (artifacts / "features.json").write_text(json.dumps(cols), encoding="utf-8")
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _mlflow_log(cfg: AppConfig, model_name: str, used_params: dict, metrics: dict,
                artifacts: Path, tracking_uri: str, run_name: str,
                config_path: str | Path | None = None) -> str:
    # One MLflow run = params + metrics + files + version stamps, so any
    # result can be traced back to code version + data version + config.
    data_hash = hashlib.sha256(Path(cfg.processed_csv).read_bytes()).hexdigest()[:12]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)
    with mlflow.start_run(run_name=run_name) as active:
        mlflow.log_params({"model": model_name, "seed": cfg.seed, **used_params})
        # Pipeline versioning: which code + which data produced this run.
        mlflow.log_param("pipeline_version", PIPELINE_VERSION)
        mlflow.log_param("data_hash", data_hash)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "n_test"):
            if metrics.get(key) is not None:
                mlflow.log_metric(key, float(metrics[key]))
        mlflow.log_artifact(str(artifacts / "model.pkl"))
        mlflow.log_artifact(str(artifacts / "features.json"))
        mlflow.log_artifact(str(artifacts / "metrics.json"))
        if config_path is not None and Path(config_path).is_file():
            mlflow.log_artifact(str(config_path))  # exact config snapshot
        return active.info.run_id


def _train_and_log(cfg: AppConfig, model_name: str, model_params: dict | None,
                   artifacts: Path, tracking_uri: str,
                   config_path: str | Path | None = None) -> dict:
    # Shared core used by run() and compare.py so every model gets a fair race.
    X_train, X_test, y_train, y_test, cols, scaler = prepare_split(cfg)

    # Fit the chosen model.
    model, used_params = _build_model(model_name, cfg.seed, model_params)
    try:
        model.fit(X_train, y_train)
    except Exception as exc:
        raise TrainingError(f"Model fit failed ({model_name}): {exc}") from exc
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = compute_metrics(y_test, pred, proba)

    _save_bundle(artifacts, model, scaler, cols, metrics)
    run_id = _mlflow_log(cfg, model_name, used_params, metrics, artifacts,
                         tracking_uri, f"{model_name}-titanic", config_path)
    log.info("%s accuracy=%.2f f1=%.2f roc_auc=%s run_id=%s",
             model_name, metrics["accuracy"], metrics["f1"],
             metrics["roc_auc"], run_id)
    return {"run_id": run_id, "artifacts": str(artifacts), **metrics}


def run(config_path: str | Path, artifacts_dir=None, mlflow_uri=None) -> dict:
    cfg: AppConfig = load_config(config_path)
    cfg.ensure_dirs()
    artifacts = Path(artifacts_dir) if artifacts_dir else cfg.artifacts_dir
    tracking_uri = mlflow_uri or cfg.mlflow_tracking_uri

    if cfg.model.name not in MODEL_REGISTRY:
        raise TrainingError(
            f"Unknown model '{cfg.model.name}' — choose from {sorted(MODEL_REGISTRY)}"
        )
    # YAML params win over DEFAULT_PARAMS (see _build_model).
    return _train_and_log(cfg, cfg.model.name, cfg.model.params,
                          artifacts, tracking_uri, config_path)
