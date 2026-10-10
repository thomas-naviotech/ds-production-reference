"""scripts/prepare.py — raw CSV -> processed feature table (DVC `prepare` stage)."""

import argparse

from titanic_prod.config import load_config
from titanic_prod.data import clean, engineer, load_raw
from titanic_prod.logging_setup import get_logger

log = get_logger("prepare")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the processed Titanic table.")
    parser.add_argument("--config", default="configs/logreg.yaml")
    args = parser.parse_args()

    # Same three steps as train.py's prepare stage: load -> clean -> engineer.
    cfg = load_config(args.config)
    cfg.ensure_dirs()  # make data/processed/ if missing
    df = engineer(clean(load_raw(cfg.raw_csv)))
    df.to_csv(cfg.processed_csv, index=False)  # cached input for train/evaluate
    log.info("Wrote %s (%d rows)", cfg.processed_csv, len(df))


if __name__ == "__main__":
    main()
