"""Typed config loader — the single place settings enter the program.

Every path in a config file is relative to the repo root, so scripts work
no matter which directory you launch them from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .exceptions import ConfigError

# Repo root so config paths work from any launch directory.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# frozen = read-only after creation; nobody can tweak settings mid-run.
@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    stratify: bool = True  # keep survived/died ratio equal in train and test


@dataclass(frozen=True)
class ModelConfig:
    name: str = "logreg"  # must match a key in train.py MODEL_REGISTRY
    params: dict = field(default_factory=dict)  # passed straight to sklearn


@dataclass(frozen=True)
class AppConfig:
    seed: int
    target: str
    raw_csv: Path
    processed_csv: Path
    artifacts_dir: Path
    mlflow_tracking_uri: str
    mlflow_experiment: str
    split: SplitConfig
    model: ModelConfig

    def ensure_dirs(self) -> None:
        # Create output folders if missing so train never crashes on first run.
        self.processed_csv.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


def _resolve(value: str) -> Path:
    # Relative paths in YAML are relative to the repo root, not cwd.
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _resolve_uri(value: str) -> str:
    """Local sqlite URIs resolve against the repo root; anything else passes through."""
    prefix = "sqlite:///"
    if value.startswith(prefix):
        return prefix + str(_resolve(value[len(prefix):]))
    return value  # e.g. a remote tracking server URL stays untouched


def load_config(path: str | Path) -> AppConfig:
    # Single entry point for settings; swap the YAML file to swap runs.
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config is not valid YAML ({path}): {exc}") from exc

    try:
        paths = raw["paths"]
        split = raw.get("split", {})
        model = raw["model"]
        return AppConfig(
            seed=int(raw.get("seed", 42)),
            target=str(raw.get("target", "Survived")),
            raw_csv=_resolve(paths["raw_csv"]),
            processed_csv=_resolve(paths["processed_csv"]),
            artifacts_dir=_resolve(paths["artifacts_dir"]),
            mlflow_tracking_uri=_resolve_uri(paths.get("mlflow_uri", "sqlite:///mlruns/mlflow.db")),
            mlflow_experiment=str(paths.get("mlflow_experiment", "titanic")),
            split=SplitConfig(
                test_size=float(split.get("test_size", 0.2)),
                stratify=bool(split.get("stratify", True)),
            ),
            model=ModelConfig(name=str(model["name"]), params=dict(model.get("params", {}))),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"Config {path} is missing required keys: {exc}") from exc
