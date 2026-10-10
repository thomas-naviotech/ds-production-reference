"""Config loader: both run configs parse, pluggable model block intact."""

import pytest

from titanic_prod.config import REPO_ROOT, load_config
from titanic_prod.exceptions import ConfigError


def test_logreg_config_loads():
    cfg = load_config(REPO_ROOT / "configs" / "logreg.yaml")
    assert cfg.model.name == "logreg"
    assert cfg.model.params["max_iter"] == 1000
    assert cfg.target == "Survived"
    assert cfg.split.test_size == 0.2


def test_rf_config_loads():
    cfg = load_config(REPO_ROOT / "configs" / "rf.yaml")
    assert cfg.model.name == "rf"
    assert cfg.model.params["n_estimators"] == 300


def test_missing_config_raises():
    with pytest.raises(ConfigError):
        load_config(REPO_ROOT / "configs" / "nope.yaml")


def test_paths_resolve_to_repo_root():
    cfg = load_config(REPO_ROOT / "configs" / "logreg.yaml")
    assert cfg.raw_csv.parent.parent.parent == REPO_ROOT
    assert cfg.raw_csv.is_file()  # bundled class CSV present
    assert cfg.mlflow_tracking_uri.startswith("sqlite:///")
    assert "mlruns/mlflow.db" in cfg.mlflow_tracking_uri.replace("\\", "/")
