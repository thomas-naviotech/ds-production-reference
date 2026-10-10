"""scripts/train.py — train a model bundle from a pluggable config."""

import argparse

from titanic_prod.display import show_metrics
from titanic_prod.train import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and log a model bundle.")
    parser.add_argument("--config", default="configs/logreg.yaml")
    args = parser.parse_args()
    result = run(args.config)
    # Rich scorecard in the terminal; same numbers MLflow just logged.
    show_metrics(f"Training result ({args.config})", result)
    print(f"run_id={result['run_id']}  artifacts={result['artifacts']}")


if __name__ == "__main__":
    main()
