import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import GridSearchCV

from snapboost import SnapBoostClassifier, SnapBoostRegressor


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


def test_regressor_works_with_grid_search():
    X, y = make_regression(n_samples=60, n_features=4, random_state=5)
    search = GridSearchCV(
        SnapBoostRegressor(num_iterations=2, random_state=5),
        {"p_tree": [0.5, 1.0]},
        cv=2,
    )

    search.fit(X, y)
    assert search.best_estimator_.ensemble_
