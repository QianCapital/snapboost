import pickle

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import GridSearchCV

from snapboost import (
    SnapBoost,
    SnapBoostClassifier,
    SnapBoostKernelRidgeClassifier,
    SnapBoostKernelRidgeRegressor,
    SnapBoostRegressor,
    SnapBoost_KernelRidge,
    __version__,
)


@pytest.mark.parametrize("estimator_class", [SnapBoostClassifier, SnapBoostRegressor])
def test_clone_and_set_params_rebuild_learner_pool(estimator_class):
    estimator = estimator_class(num_iterations=2, random_state=7)
    cloned = clone(estimator)

    result = cloned.set_params(p_tree=0.5, min_max_depth=1, max_max_depth=2)

    assert result is cloned
    assert cloned.probabilities_ == pytest.approx([0.25, 0.25, 0.5])


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_invalid_tree_depth_is_rejected(value):
    with pytest.raises(ValueError, match="min_max_depth"):
        SnapBoostRegressor(min_max_depth=value)


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_invalid_max_tree_depth_is_rejected(value):
    with pytest.raises(ValueError, match="max_max_depth"):
        SnapBoostRegressor(max_max_depth=value)


def test_classifier_fits_and_predicts_probabilities():
    X, y = make_classification(n_samples=60, n_features=5, random_state=4)
    model = SnapBoostClassifier(num_iterations=3, random_state=4).fit(X, y)

    probabilities = model.predict_proba(X)
    assert probabilities.shape == (60, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_classifier_preserves_string_labels():
    X, y = make_classification(n_samples=60, n_features=5, random_state=4)
    labels = np.where(y == 0, "negative", "positive")
    model = SnapBoostClassifier(num_iterations=3, random_state=4).fit(X, labels)

    assert np.array_equal(model.classes_, ["negative", "positive"])
    assert set(model.predict(X)).issubset({"negative", "positive"})


def test_regressor_works_with_grid_search():
    X, y = make_regression(n_samples=60, n_features=4, random_state=5)
    search = GridSearchCV(
        SnapBoostRegressor(num_iterations=2, random_state=5),
        {"p_tree": [0.5, 1.0]},
        cv=2,
    )

    search.fit(X, y)
    assert search.best_estimator_.ensemble_


@pytest.mark.parametrize(
    "parameter, value, message",
    [
        ("min_max_depth", True, "integer"),
        ("max_max_depth", True, "integer"),
        ("n_components", 1.5, "integer"),
        ("n_components", True, "integer"),
        ("p_tree", float("nan"), "between"),
        ("alpha", float("nan"), "finite"),
        ("gamma", float("inf"), "finite"),
        ("min_samples_leaf", 1.5, "integer"),
    ],
)
def test_invalid_snapboost_parameters_are_rejected(parameter, value, message):
    with pytest.raises(ValueError, match=message):
        SnapBoostRegressor(**{parameter: value})


def test_snapboost_set_params_is_transactional():
    model = SnapBoostRegressor(min_max_depth=2, max_max_depth=4)
    original_pool = model.base_learners_

    with pytest.raises(ValueError, match="min_max_depth"):
        model.set_params(min_max_depth=5)

    assert model.min_max_depth == 2
    assert model.max_max_depth == 4
    assert model.base_learners_ is original_pool


def test_unknown_set_param_does_not_partially_mutate_fitted_model():
    X, y = make_regression(n_samples=30, n_features=3, random_state=5)
    model = SnapBoostRegressor(num_iterations=2, random_state=5).fit(X, y)
    original_pool = model.base_learners_
    original_ensemble = model.ensemble_
    expected = model.predict(X)

    with pytest.raises(ValueError, match="does_not_exist"):
        model.set_params(p_tree=0.25, does_not_exist=True)

    assert model.p_tree == 0.9
    assert model.base_learners_ is original_pool
    assert model.ensemble_ is original_ensemble
    assert model.predict(X) == pytest.approx(expected)


@pytest.mark.parametrize("value", [-1, np.int64(-1)])
def test_negative_random_state_is_rejected_early(value):
    with pytest.raises(ValueError, match="non-negative"):
        SnapBoostRegressor(random_state=value)


def test_large_random_state_is_safely_derived_for_base_learners():
    X, y = make_regression(n_samples=30, n_features=3, random_state=5)
    model = SnapBoostRegressor(
        num_iterations=2,
        random_state=2**64,
    ).fit(X, y)

    assert np.all(np.isfinite(model.predict(X)))


@pytest.mark.parametrize("p_tree", [0.0, 1.0])
def test_learner_pool_probability_boundaries_fit(p_tree):
    X, y = make_regression(n_samples=30, n_features=3, random_state=5)
    model = SnapBoostRegressor(
        num_iterations=2,
        p_tree=p_tree,
        random_state=5,
    ).fit(X, y)

    assert sum(model.probabilities_) == pytest.approx(1.0)
    assert np.all(np.isfinite(model.predict(X)))


def test_fitted_estimator_round_trips_through_pickle():
    X, y = make_regression(n_samples=30, n_features=3, random_state=5)
    model = SnapBoostRegressor(num_iterations=2, random_state=5).fit(X, y)

    restored = pickle.loads(pickle.dumps(model))

    assert restored.predict(X) == pytest.approx(model.predict(X))


@pytest.mark.parametrize("mode", ["classification", "regression"])
def test_legacy_snapboost_fits(mode):
    if mode == "classification":
        X, y = make_classification(n_samples=30, n_features=4, random_state=9)
    else:
        X, y = make_regression(n_samples=30, n_features=3, random_state=9)

    model = SnapBoost(
        num_iterations=2,
        mode=mode,
        random_state=9,
        verbose=False,
    ).fit(X, y)

    if mode == "classification":
        assert model.predict_proba(X).shape == (30, 2)
    else:
        assert np.all(np.isfinite(model.predict(X)))


def test_legacy_kernel_ridge_set_params_is_transactional():
    model = SnapBoost_KernelRidge(verbose=False)
    original_pool = model.base_learners_

    with pytest.raises(ValueError, match="finite"):
        model.set_params(gamma=float("nan"))

    assert model.gamma == 1.0
    assert model.base_learners_ is original_pool


def test_legacy_kernel_ridge_unknown_param_is_transactional():
    model = SnapBoost_KernelRidge(verbose=False)

    with pytest.raises(ValueError, match="does_not_exist"):
        model.set_params(p_tree=0.25, does_not_exist=True)

    assert model.p_tree == 0.9


@pytest.mark.parametrize(
    "estimator_class, checker",
    [
        (SnapBoostKernelRidgeClassifier, is_classifier),
        (SnapBoostKernelRidgeRegressor, is_regressor),
    ],
)
def test_task_specific_kernel_ridge_estimators_fit(estimator_class, checker):
    if checker is is_classifier:
        X, y = make_classification(n_samples=30, n_features=5, random_state=5)
    else:
        X, y = make_regression(n_samples=30, n_features=3, random_state=5)
    model = estimator_class(
        num_iterations=2,
        p_tree=0.0,
        random_state=5,
    ).fit(X, y)

    assert checker(model)
    assert np.all(np.isfinite(model.predict(X)))


def test_package_exposes_version():
    assert __version__ == "0.1.7"
