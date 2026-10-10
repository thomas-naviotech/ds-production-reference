"""scripts/evaluate.py — score the trained bundle (DVC `evaluate` stage)."""

import argparse

from titanic_prod.evaluate import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained bundle.")
    parser.add_argument("--config", default="configs/logreg.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
