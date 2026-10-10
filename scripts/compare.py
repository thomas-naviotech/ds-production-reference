"""scripts/compare.py — train EVERY model on the same split, rank by F1.

Usage:
    uv run python scripts/compare.py                        # all sklearn models
    uv run python scripts/compare.py --models logreg rf svm # subset
    uv run python scripts/compare.py --config configs/rf.yaml  # reuse its split/seed

Each model gets its own MLflow run + artifacts/compare/<name>/ folder,
so the main artifacts/ bundle (used by serve.py) is untouched.
Open `mlflow ui` to compare runs visually.
"""

import argparse
from pathlib import Path

from titanic_prod.config import load_config
from titanic_prod.display import leaderboard
from titanic_prod.logging_setup import get_logger
from titanic_prod.train import DEFAULT_PARAMS, MODEL_REGISTRY, _train_and_log

log = get_logger("compare")

# Classroom order: simple first, boosting last. xgb auto-skips if not installed.
ALL_MODELS = [m for m in ("logreg", "rf", "svm", "knn", "nb", "gb", "xgb")
              if m in MODEL_REGISTRY]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all models on one split.")
    parser.add_argument("--config", default="configs/logreg.yaml",
                        help="Base config for seed/split/paths (model block ignored).")
    parser.add_argument("--models", nargs="*", default=ALL_MODELS,
                        help=f"Subset to run. Choices: {sorted(MODEL_REGISTRY)}")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.ensure_dirs()
    base_artifacts = Path(cfg.artifacts_dir)
    rows: list[dict] = []
    for name in args.models:
        if name not in MODEL_REGISTRY:
            log.info("Skipping unknown model '%s' (choices: %s)", name, sorted(MODEL_REGISTRY))
            continue
        # One subfolder per model so bundles never overwrite each other.
        artifacts = base_artifacts / "compare" / name
        res = _train_and_log(cfg, name, dict(DEFAULT_PARAMS.get(name, {})),
                             artifacts, cfg.mlflow_tracking_uri, args.config)
        rows.append({"model": name, **{k: res.get(k) for k in
                                       ("accuracy", "precision", "recall", "f1", "roc_auc")}})
        print(f"{name:8s} acc={res['accuracy']:.2f} f1={res['f1']:.2f} "
              f"roc_auc={res['roc_auc']} run_id={res['run_id']}")

    # Rank by F1, then ROC AUC, then accuracy (2dp ties happen — chain breaks them).
    rows.sort(key=lambda r: (r["f1"], r["roc_auc"], r["accuracy"], r["model"]),
              reverse=True)
    leaderboard(rows)  # rich table + F1 bar chart in the terminal
    if rows:
        print(f"\nTune the winner: uv run python scripts/tune.py --model {rows[0]['model']}")
        print("Or auto-pick it:   uv run python scripts/tune.py --model winner")
    if "xgb" not in MODEL_REGISTRY:
        print("\nNote: xgboost not installed - skipped 'xgb'. "
              "Run `uv pip install xgboost` to include it.")


if __name__ == "__main__":
    main()
