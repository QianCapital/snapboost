import inspect
import pickle

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from snapboost import (
    RandomFourierRidgeRegressor,
    SnapBoost,
    SnapBoost_KernelRidge,
    SnapBoostClassifier,
    SnapBoostKernelRidgeClassifier,
    SnapBoostKernelRidgeRegressor,
    SnapBoostRegressor,
    WeightedLinearRegressor,
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
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    with pytest.raises(ValueError, match="min_max_depth"):
        SnapBoostRegressor(min_max_depth=value).fit(X, y)


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_invalid_max_tree_depth_is_rejected(value):
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    with pytest.raises(ValueError, match="max_max_depth"):
        SnapBoostRegressor(max_max_depth=value).fit(X, y)


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


def test_fit_forwards_sample_weight_and_eval_set():
    X, y = make_regression(n_samples=60, n_features=4, random_state=5)
    sample_weight = np.linspace(1.0, 2.0, X.shape[0])
    model = SnapBoostRegressor(
        num_iterations=3,
        random_state=5,
    ).fit(
        X,
        y,
        sample_weight=sample_weight,
        eval_set=(X, y),
    )

    assert model.n_iter_ == 3
    assert len(model.history_["validation_loss"]) == 3


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
        ("max_features", np.array([0.5]), "max_features"),
        ("monotonic_cst", (0, 2), "monotonic_cst"),
        ("p_linear", -0.1, "p_linear"),
    ],
)
def test_invalid_snapboost_parameters_are_rejected(parameter, value, message):
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    with pytest.raises(ValueError, match=message):
        SnapBoostRegressor(**{parameter: value}).fit(X, y)


def test_snapboost_set_params_stores_values_before_fit_validation():
    model = SnapBoostRegressor(min_max_depth=2, max_max_depth=4)
    result = model.set_params(min_max_depth=5)

    assert result is model
    assert model.min_max_depth == 5
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    with pytest.raises(ValueError, match="min_max_depth"):
        model.fit(X, y)


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
def test_negative_random_state_is_rejected_at_fit(value):
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    with pytest.raises(ValueError, match="non-negative"):
        SnapBoostRegressor(random_state=value).fit(X, y)


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

    with pytest.warns(FutureWarning, match="deprecated"):
        model = SnapBoost(
            num_iterations=2,
            mode=mode,
            random_state=9,
            verbose=False,
        )
    model.fit(X, y)

    if mode == "classification":
        assert model.predict_proba(X).shape == (30, 2)
    else:
        assert np.all(np.isfinite(model.predict(X)))


def test_legacy_kernel_ridge_set_params_stores_values_before_fit_validation():
    with pytest.warns(FutureWarning, match="deprecated"):
        model = SnapBoost_KernelRidge(verbose=False)
    model.set_params(gamma=float("nan"))
    assert np.isnan(model.gamma)
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    with pytest.raises(ValueError, match="finite"):
        model.fit(X, y)


def test_legacy_kernel_ridge_unknown_param_is_transactional():
    with pytest.warns(FutureWarning, match="deprecated"):
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
    assert __version__ == "1.0.0"


def test_advanced_regressor_supports_greedy_selection_and_line_search():
    X, y = make_regression(n_samples=80, n_features=4, random_state=7)
    model = SnapBoostRegressor(
        num_iterations=3,
        selection_strategy="greedy",
        line_search=True,
        subsample=0.8,
        max_features=0.75,
        random_state=7,
    ).fit(X, y)

    assert model.n_iter_ == 3
    assert len(model.learner_weights_) == 3
    assert len(model.history_["training_loss"]) == 3
    assert np.all(np.isfinite(model.predict(X)))


def test_validation_and_sample_weights_flow_through_snapboost():
    X, y = make_classification(n_samples=100, n_features=5, random_state=8)
    weights = np.where(y == 1, 2.0, 1.0)
    model = SnapBoostClassifier(
        num_iterations=10,
        early_stopping_rounds=2,
        min_delta=1e100,
        random_state=8,
    ).fit(
        X[:80], y[:80], sample_weight=weights[:80], eval_set=(X[80:], y[80:])
    )

    assert model.n_iter_ == 1
    assert model.history_["validation_loss"]


def test_rff_rounds_receive_distinct_reproducible_bases():
    X, y = make_regression(n_samples=40, n_features=3, random_state=9)
    model = SnapBoostRegressor(
        num_iterations=2, p_tree=0.0, random_state=9
    ).fit(X, y)

    seeds = [learner.random_state for learner in model.ensemble_]
    assert seeds[0] != seeds[1]
    assert all(learner.pipeline_.named_steps.get("scale") is not None
               for learner in model.ensemble_)


def test_ridge_only_greedy_pool_contains_no_zero_probability_trees():
    X, y = make_regression(n_samples=40, n_features=3, random_state=9)
    model = SnapBoostRegressor(
        num_iterations=2,
        p_tree=0.0,
        selection_strategy="greedy",
        random_state=9,
    ).fit(X, y)

    assert model.probabilities_ == [1.0]
    assert all(isinstance(learner, RandomFourierRidgeRegressor)
               for learner in model.ensemble_)


def test_multiscale_kernel_pool_is_opt_in_and_normalized():
    X, y = make_regression(n_samples=30, n_features=3, random_state=9)
    model = SnapBoostRegressor(
        num_iterations=1,
        p_tree=0.5,
        min_max_depth=2,
        max_max_depth=2,
        kernel_gammas=(0.1, 1.0),
        kernel_types=("rbf", "laplacian"),
        random_state=9,
    ).fit(X, y)

    assert model.probabilities_ == pytest.approx([0.5, 0.125, 0.125, 0.125, 0.125])
    kernels = [learner.kernel for learner in model.base_learners_[1:]]
    gammas = [learner.gamma for learner in model.base_learners_[1:]]
    assert kernels == ["rbf", "rbf", "laplacian", "laplacian"]
    assert gammas == [0.1, 1.0, 0.1, 1.0]


def test_monotonic_constraints_are_opt_in_and_version_guarded():
    X, y = make_regression(n_samples=30, n_features=3, random_state=9)
    model = SnapBoostRegressor(
        num_iterations=1, monotonic_cst=(1, 0, -1), random_state=9
    )
    if "monotonic_cst" in inspect.signature(DecisionTreeRegressor).parameters:
        model.fit(X, y)
        tree = next(
            learner for learner in model.base_learners_
            if isinstance(learner, DecisionTreeRegressor)
        )
        assert tuple(tree.monotonic_cst) == (1, 0, -1)
    else:
        with pytest.raises(RuntimeError, match="scikit-learn"):
            model.fit(X, y)


def test_optional_linear_family_receives_explicit_probability():
    X, y = make_regression(n_samples=30, n_features=3, random_state=9)
    model = SnapBoostRegressor(
        num_iterations=1,
        p_tree=0.5,
        p_linear=0.25,
        min_max_depth=2,
        max_max_depth=2,
        random_state=9,
    ).fit(X, y)

    assert model.probabilities_ == pytest.approx([0.5, 0.25, 0.25])
    assert isinstance(model.base_learners_[-1], WeightedLinearRegressor)


def test_tree_and_linear_probabilities_cannot_exceed_one():
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    with pytest.raises(ValueError, match="p_tree.*p_linear"):
        SnapBoostRegressor(p_tree=0.8, p_linear=0.3).fit(X, y)


def test_snapboost_forwards_objective_metrics_callbacks_and_parallelism():
    X, y = make_regression(n_samples=50, n_features=3, random_state=9)
    callback_iterations = []

    def callback(state):
        callback_iterations.append(state["iteration"])
        return state["iteration"] == 1

    model = SnapBoostRegressor(
        num_iterations=10,
        objective="pseudo_huber",
        objective_parameter=2.0,
        selection_strategy="greedy",
        random_state=9,
    ).fit(
        X,
        y,
        eval_metric=lambda truth, raw: np.mean(np.abs(truth - raw)),
        callbacks=[callback],
        candidate_n_jobs=2,
    )

    assert model.n_iter_ == 2
    assert callback_iterations == [0, 1]
    assert len(model.history_["training_metric"]) == 2


@pytest.mark.parametrize(
    "p_tree, p_linear",
    [(0.7, 0.3), (0.9, 0.1), (0.8, 0.2), (0.6, 0.4), (0.5, 0.5), (0.1, 0.9)],
)
def test_exhausted_probability_budget_excludes_kernel_learners(p_tree, p_linear):
    """Rounding in ``1 - p_tree - p_linear`` must not smuggle in an RFF learner."""
    model = SnapBoostRegressor(p_tree=p_tree, p_linear=p_linear)
    model._build_base_learners()

    assert not any(
        isinstance(learner, RandomFourierRidgeRegressor)
        for learner in model.base_learners_
    )
    assert all(probability > 1e-12 for probability in model.probabilities_)
    assert sum(model.probabilities_) == pytest.approx(1.0)


def test_greedy_selection_ignores_excluded_kernel_family():
    X, y = make_regression(n_samples=60, n_features=4, random_state=3)
    model = SnapBoostRegressor(
        p_tree=0.7,
        p_linear=0.3,
        selection_strategy="greedy",
        num_iterations=5,
        random_state=3,
    ).fit(X, y)

    assert not any(
        isinstance(learner, RandomFourierRidgeRegressor)
        for learner in model.ensemble_
    )


def test_dataframe_column_order_is_validated():
    pd = pytest.importorskip("pandas")
    X, y = make_regression(n_samples=60, n_features=4, random_state=5)
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    model = SnapBoostRegressor(num_iterations=5, random_state=5).fit(frame, y)

    assert list(model.feature_names_in_) == ["a", "b", "c", "d"]
    with pytest.raises(ValueError, match="feature names"):
        model.predict(frame[["d", "c", "b", "a"]])


def test_task_estimators_reject_mode_in_set_params():
    with pytest.raises(ValueError, match="mode"):
        SnapBoostClassifier().set_params(mode="regression")
    with pytest.raises(ValueError, match="mode"):
        SnapBoostRegressor().set_params(mode="classification")
    with pytest.raises(ValueError, match="mode"):
        SnapBoostKernelRidgeClassifier().set_params(mode="regression")
    with pytest.raises(ValueError, match="mode"):
        SnapBoostKernelRidgeRegressor().set_params(mode="classification")


def test_exact_kernel_rejects_non_rbf_family():
    X, y = make_regression(n_samples=30, n_features=3, random_state=4)
    model = SnapBoostKernelRidgeRegressor(p_tree=0.0, num_iterations=1)
    model.kernel_types = ("laplacian",)
    with pytest.raises(ValueError, match="RBF"):
        model.fit(X, y)


def test_max_features_string_and_boolean_are_validated_at_fit():
    X, y = make_regression(n_samples=30, n_features=3, random_state=4)
    SnapBoostRegressor(max_features="sqrt", num_iterations=1, random_state=4).fit(X, y)
    with pytest.raises(ValueError, match="max_features"):
        SnapBoostRegressor(max_features=True, num_iterations=1).fit(X, y)


def test_legacy_snapboost_set_params_rebuilds_valid_pool():
    with pytest.warns(FutureWarning, match="deprecated"):
        model = SnapBoost(num_iterations=1, verbose=False, random_state=0)
    model.set_params(p_tree=0.5, min_max_depth=1, max_max_depth=1)
    assert model.probabilities_ == pytest.approx([0.5, 0.5])


def test_all_zero_sample_weights_mention_zero_in_the_error():
    X, y = make_regression(n_samples=30, n_features=3, random_state=4)
    with pytest.raises(ValueError, match=r"(.*weight.*zero.*)|(.*zero.*weight.*)"):
        SnapBoostRegressor(num_iterations=1, random_state=4).fit(
            X, y, sample_weight=np.zeros(len(y))
        )

