import numpy as np
import pytest

from snapboost.kernel_learner import WeightedKernelRidgeRegressor


def test_weighted_kernel_ridge_is_weight_scale_invariant():
    X = np.arange(30.0).reshape(15, 2)
    y = np.arange(15.0)
    weights = np.linspace(1.0, 2.0, 15)
    first = WeightedKernelRidgeRegressor().fit(X, y, sample_weight=weights)
    second = WeightedKernelRidgeRegressor().fit(X, y, sample_weight=weights * 100)

    assert first.predict(X) == pytest.approx(second.predict(X))
    assert first.pipeline_.named_steps.get("scale") is not None


def test_weighted_kernel_ridge_rejects_invalid_weights():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0)
    with pytest.raises(ValueError, match="positive"):
        WeightedKernelRidgeRegressor().fit(X, y, sample_weight=np.zeros(10))
    with pytest.raises(ValueError, match="boolean"):
        WeightedKernelRidgeRegressor(scale_features="yes").fit(X, y)
    with pytest.raises(ValueError, match="finite"):
        WeightedKernelRidgeRegressor(alpha=-1.0).fit(X, y)
    with pytest.raises(ValueError, match="finite"):
        WeightedKernelRidgeRegressor(gamma=float("nan")).fit(X, y)
    with pytest.raises(ValueError, match="one value"):
        WeightedKernelRidgeRegressor().fit(X, y, sample_weight=np.ones((10, 1)))
    fitted = WeightedKernelRidgeRegressor().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        fitted.predict(np.ones((2, 3)))
