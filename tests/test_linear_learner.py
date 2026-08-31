import numpy as np
import pytest

from snapboost import WeightedLinearRegressor


def test_weighted_linear_regressor_is_weight_scale_invariant():
    X = np.arange(30.0).reshape(15, 2)
    y = np.arange(15.0)
    weights = np.linspace(1.0, 2.0, 15)
    first = WeightedLinearRegressor().fit(X, y, sample_weight=weights)
    second = WeightedLinearRegressor().fit(X, y, sample_weight=weights * 100)

    assert first.predict(X) == pytest.approx(second.predict(X))


def test_weighted_linear_regressor_rejects_invalid_weights():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0)
    with pytest.raises(ValueError, match="positive"):
        WeightedLinearRegressor().fit(X, y, sample_weight=np.zeros(10))
    with pytest.raises(ValueError, match="boolean"):
        WeightedLinearRegressor(scale_features="yes").fit(X, y)
    with pytest.raises(ValueError, match="finite"):
        WeightedLinearRegressor(alpha=-1.0).fit(X, y)
    with pytest.raises(ValueError, match="one value"):
        WeightedLinearRegressor().fit(X, y, sample_weight=np.ones((10, 1)))
    fitted = WeightedLinearRegressor().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        fitted.predict(np.ones((2, 3)))

