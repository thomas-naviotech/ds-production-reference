"""scripts/tune.py — 2-stage tuning for the F1 winner (or any model).

Usage:
    uv run python scripts/compare.py                  # first: rank all models
    uv run python scripts/tune.py --model winner      # tune the F1 winner
    uv run python scripts/tune.py --model svm         # ...or name any model
    uv run python scripts/tune.py --model rf --n-iter 15

Stage 1 (broad): RandomizedSearchCV over wide value lists.
Stage 2 (narrow): GridSearchCV zoomed around the stage-1 best.
Same split as compare.py (same seed/config), scored by F1.
Best bundle -> artifacts/tune/<name>/ + one MLflow run `<name>-tune`.
"""

import argparse
import json
from pathlib import Path

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from titanic_prod.config import load_config
from titanic_prod.display import show_metrics, show_params
from titanic_prod.logging_setup import get_logger
from titanic_prod.metrics import compute_metrics
from titanic_prod.train import (
    MODEL_REGISTRY,
    _build_model,
    _mlflow_log,
    _save_bundle,
    prepare_split,
)

log = get_logger("tune")

# Broad search lists per model. "zoom" tells stage 2 how to narrow around
# the stage-1 best: log = x/3..3x (C-like), count/int = neighbors by size,
# frac = +-0.1, fixed = keep as-is. min/max are safety rails.
SEARCH_SPACES: dict[str, dict[str, dict]] = {
    "logreg": {
        "C": {"broad": [0.01, 0.1, 1.0, 10.0, 100.0], "zoom": "log",
              "min": 1e-4, "max": 1e3},
    },
    "rf": {
        "n_estimators": {"broad": [100, 200, 300, 500], "zoom": "count",
                         "min": 50, "max": 1000},
        "max_depth": {"broad": [6, 10, 16, 24], "zoom": "int", "min": 2, "max": 32},
        "min_samples_split": {"broad": [2, 5, 10], "zoom": "int", "min": 2, "max": 20},
    },
    "svm": {
        # CalibratedClassifierCV wraps SVC, so grid keys use `estimator__`.
        "estimator__C": {"broad": [0.01, 0.1, 1.0, 10.0, 100.0], "zoom": "log",
                         "min": 1e-4, "max": 1e3},
    },
    "knn": {
        "n_neighbors": {"broad": [3, 5, 7, 9, 15], "zoom": "int", "min": 1, "max": 31},
        "weights": {"broad": ["uniform", "distance"], "zoom": "fixed"},
        "p": {"broad": [1, 2], "zoom": "fixed"},
    },
    "nb": {
        "var_smoothing": {"broad": [1e-9, 1e-8, 1e-7, 1e-6], "zoom": "log",
                          "min": 1e-12, "max": 1e-3},
    },
    "gb": {
        "n_estimators": {"broad": [50, 100, 200], "zoom": "count",
                         "min": 20, "max": 500},
        "learning_rate": {"broad": [0.02, 0.05, 0.1, 0.2], "zoom": "log",
                          "min": 1e-3, "max": 0.5},
        "max_depth": {"broad": [2, 3, 4, 5], "zoom": "int", "min": 1, "max": 8},
    },
    "xgb": {
        "n_estimators": {"broad": [100, 200, 400], "zoom": "count",
                         "min": 50, "max": 1000},
        "max_depth": {"broad": [3, 4, 5, 6], "zoom": "int", "min": 1, "max": 12},
        "learning_rate": {"broad": [0.03, 0.05, 0.1, 0.2], "zoom": "log",
                          "min": 1e-3, "max": 0.5},
        "subsample": {"broad": [0.7, 0.85, 1.0], "zoom": "frac"},
    },
}


def _zoom_values(spec: dict, best):
    # Build the stage-2 narrow list around the stage-1 best value.
    kind = spec.get("zoom", "fixed")
    if kind == "fixed" or best is None or isinstance(best, (str, bool)):
        return [best]
    if kind == "log":  # multiply/divide by 3 (right scale for C, lr, smoothing)
        cands = [round(float(best) / 3, 6), best, round(float(best) * 3, 6)]
        cands = [c for c in cands if c > 0]
    elif kind == "count":  # halve/double (right scale for tree counts)
        cands = [int(best // 2), int(best), int(best * 2)]
        cands = [c for c in cands if c >= 1]
    elif kind == "int":  # step sideways (depths, neighbors)
        step = max(1, round(best * 0.4)) if best < 50 else max(1, best // 4)
        cands = [int(best - step), int(best), int(best + step)]
        cands = [c for c in cands if c >= 1]
    elif kind == "frac":  # +-0.1 inside (0, 1] (subsample ratios)
        cands = [round(best - 0.1, 2), round(best, 2), round(best + 0.1, 2)]
        cands = [c for c in cands if 0 < c <= 1.0]
    else:
        return [best]
    if spec.get("min") is not None:
        cands = [c for c in cands if c >= spec["min"]]
    if spec.get("max") is not None:
        cands = [c for c in cands if c <= spec["max"]]
    return sorted(set(cands)) or [best]


def _pick_winner(compare_dir: Path) -> tuple[str, dict]:
    # Highest F1 wins; 2dp ties broken by ROC AUC, then accuracy, then name.
    # Deterministic: first alphabetically wins a full tie.
    best_key, best_name, best_m = None, "", {}
    for mfile in sorted(compare_dir.glob("*/metrics.json")):
        m = json.loads(mfile.read_text(encoding="utf-8"))
        key = (m.get("f1") if m.get("f1") is not None else -1,
               m.get("roc_auc") if m.get("roc_auc") is not None else -1,
               m.get("accuracy") if m.get("accuracy") is not None else -1)
        if best_key is None or key > best_key:
            best_key, best_name, best_m = key, mfile.parent.name, m
    if not best_name:
        raise SystemExit(f"No compare results in {compare_dir} — run compare.py first.")
    return best_name, best_m


def main() -> None:
    parser = argparse.ArgumentParser(description="2-stage tuning for one model.")
    parser.add_argument("--config", default="configs/logreg.yaml",
                        help="Base config for seed/split/paths (model block ignored).")
    parser.add_argument("--model", default="winner",
                        help="'winner' (F1 best from compare.py) or a model name: "
                             f"{sorted(MODEL_REGISTRY)}")
    parser.add_argument("--n-iter", type=int, default=8,
                        help="Stage-1 random combinations to try.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.ensure_dirs()
    name = args.model
    if name == "winner":
        name, prev = _pick_winner(Path(cfg.artifacts_dir) / "compare")
        print(f"Winner from compare.py: {name} "
              f"(f1={prev.get('f1'):.2f} roc_auc={prev.get('roc_auc')})")
    if name not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model '{name}' — choices: {sorted(MODEL_REGISTRY)}")
    if name not in SEARCH_SPACES:
        raise SystemExit(f"No search space for '{name}' yet — add it to SEARCH_SPACES.")

    # Same split compare.py used, so tuning gains are real, not split luck.
    X_train, X_test, y_train, y_test, cols, scaler = prepare_split(cfg)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.seed)
    space = SEARCH_SPACES[name]
    broad = {k: v["broad"] for k, v in space.items()}

    # Stage 1 — broad random sweep.
    base, _ = _build_model(name, cfg.seed, {})
    rs = RandomizedSearchCV(base, broad, n_iter=args.n_iter, cv=cv, scoring="f1",
                            random_state=cfg.seed, n_jobs=-1)
    rs.fit(X_train, y_train)
    print(f"\nStage 1 (broad, {args.n_iter} random combos) CV f1={rs.best_score_:.2f}")
    show_params("Stage 1 best params", dict(rs.best_params_))

    # Stage 2 — narrow grid zoomed around the stage-1 best.
    narrow = {k: _zoom_values(space[k], v) for k, v in rs.best_params_.items()
              if k in space}
    base2, _ = _build_model(name, cfg.seed, {})
    gs = GridSearchCV(base2, narrow, cv=cv, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f"\nStage 2 (narrow grid) CV f1={gs.best_score_:.2f}")
    show_params("Stage 2 best params", dict(gs.best_params_))

    # Final score on the held-out test split + bundle + MLflow run.
    best = gs.best_estimator_
    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1] if hasattr(best, "predict_proba") else None
    metrics = compute_metrics(y_test, pred, proba)
    artifacts = Path(cfg.artifacts_dir) / "tune" / name
    _save_bundle(artifacts, best, scaler, cols, metrics)
    used = {**rs.best_params_, **gs.best_params_,
            "stage1_cv_f1": round(float(rs.best_score_), 2),
            "stage2_cv_f1": round(float(gs.best_score_), 2)}
    run_id = _mlflow_log(cfg, name, used, metrics, artifacts,
                         cfg.mlflow_tracking_uri, f"{name}-tune", args.config)
    show_metrics(f"Tuned {name} (held-out test)", metrics)  # rich scorecard
    print(f"run_id={run_id}")
    print(f"Bundle: {artifacts} (serve it via create_app('{artifacts.as_posix()}'))")


if __name__ == "__main__":
    main()
