import inspect

import pytest
from sklearn.utils.estimator_checks import check_estimator

from snapboost import SnapBoostClassifier, SnapBoostRegressor


def _expected_failed_checks():
    return {
        "check_sample_weight_equivalence_on_dense_data": (
            "Newton boosting with Hessian-weighted learners is not equivalent "
            "to repeating rows according to integer sample weights."
        ),
    }


@pytest.mark.skipif(
    "expected_failed_checks" not in inspect.signature(check_estimator).parameters,
    reason="sklearn 1.6+ expected_failed_checks is required",
)
@pytest.mark.parametrize(
    "estimator",
    [
        SnapBoostClassifier(
            num_iterations=8, p_tree=1.0, random_state=0, verbose=False
        ),
        SnapBoostRegressor(
            num_iterations=8, p_tree=1.0, random_state=0, verbose=False
        ),
    ],
)
def test_snapboost_passes_sklearn_check_estimator(estimator):
    check_estimator(estimator, expected_failed_checks=_expected_failed_checks())
