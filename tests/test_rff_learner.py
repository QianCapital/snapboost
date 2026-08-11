import numpy as np
import pytest
from sklearn.exceptions import DataConversionWarning

from snapboost import RandomFourierRidgeRegressor


@pytest.mark.parametrize(
    "parameter, value, message",
    [
        ("alpha", float("nan"), "finite"),
        ("gamma", float("inf"), "finite"),
        ("n_components", 1.5, "integer"),
        ("n_components", True, "integer"),
        ("random_state", True, "integer"),
    ],
)
def test_invalid_parameters_are_rejected(parameter, value, message):
    model = RandomFourierRidgeRegressor(**{parameter: value})
    with pytest.raises(ValueError, match=message):
        model.fit(np.ones((3, 2)), np.ones(3))


def test_weighted_fit_and_prediction_are_finite():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0)
    model = RandomFourierRidgeRegressor(random_state=2).fit(
        X, y, sample_weight=np.ones(10)
    )

    assert np.all(np.isfinite(model.predict(X)))
    assert model.n_features_in_ == 2


def test_weight_scaling_does_not_change_regularization_strength():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0)
    weights = np.linspace(1.0, 2.0, 10)
    model = RandomFourierRidgeRegressor(random_state=2).fit(
        X, y, sample_weight=weights
    )
    scaled = RandomFourierRidgeRegressor(random_state=2).fit(
        X, y, sample_weight=weights * 100.0
    )

    assert model.predict(X) == pytest.approx(scaled.predict(X))


@pytest.mark.parametrize(
    "sample_weight, message",
    [
        (np.zeros(10), "positive total"),
        (np.array([1.0] * 9 + [-1.0]), "non-negative"),
        (np.array([1.0] * 9 + [np.nan]), "finite"),
        (np.ones((10, 1)), "one-dimensional"),
        (np.ones(9), "one value per sample"),
    ],
)
def test_invalid_sample_weights_are_rejected(sample_weight, message):
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0)

    with pytest.raises(ValueError, match=message):
        RandomFourierRidgeRegressor(random_state=2).fit(
            X, y, sample_weight=sample_weight
        )


def test_failed_refit_preserves_previous_pipeline():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0)
    model = RandomFourierRidgeRegressor(random_state=2).fit(X, y)
    expected = model.predict(X)
    original_pipeline = model.pipeline_

    with pytest.raises(ValueError):
        model.fit(X, y[:-1])

    assert model.pipeline_ is original_pipeline
    assert model.predict(X) == pytest.approx(expected)


def test_column_target_uses_standard_sklearn_warning():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0).reshape(-1, 1)

    with pytest.warns(DataConversionWarning):
        model = RandomFourierRidgeRegressor(random_state=2).fit(X, y)

    assert model.predict(X).shape == (10,)


def test_prediction_rejects_wrong_feature_count():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0)
    model = RandomFourierRidgeRegressor(random_state=2).fit(X, y)

    with pytest.raises(ValueError, match="features"):
        model.predict(np.ones((3, 4)))
