"""Model registry + shared metrics: every classroom model builds and scores."""

from titanic_prod.metrics import compute_metrics
from titanic_prod.train import DEFAULT_PARAMS, MODEL_REGISTRY, _build_model


def test_registry_has_classroom_models():
    # The compare.py leaderboard needs at least these sklearn-only models.
    for name in ("logreg", "rf", "svm", "knn", "nb", "gb"):
        assert name in MODEL_REGISTRY, f"missing model: {name}"


def test_each_model_builds_with_defaults():
    # Every registered model must construct from its defaults + seed 42.
    for name in MODEL_REGISTRY:
        model, params = _build_model(name, seed=42, extra=None)
        assert model is not None
        for key, val in DEFAULT_PARAMS.get(name, {}).items():
            assert params[key] == val


def test_compute_metrics_toy_example():
    # Tiny hand-checkable case: 3/4 correct, so accuracy == 0.75.
    m = compute_metrics([0, 1, 1, 0], [0, 1, 0, 0], [0.1, 0.9, 0.4, 0.2])
    assert m["accuracy"] == 0.75
    assert 0.0 <= m["f1"] <= 1.0
    assert m["roc_auc"] is not None and 0.0 <= m["roc_auc"] <= 1.0
    assert m["n_test"] == 4
