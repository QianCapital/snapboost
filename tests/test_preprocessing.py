import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from snapboost import SnapBoostClassifier, make_tabular_preprocessor


def test_tabular_preprocessor_handles_missing_and_unseen_categories():
    X_train = np.array([
        [1.0, "a"],
        [np.nan, "b"],
        [3.0, None],
        [4.0, "a"],
    ], dtype=object)
    y_train = np.array([0, 1, 1, 0])
    X_test = np.array([[np.nan, "never-seen"]], dtype=object)
    model = Pipeline([
        (
            "prepare",
            make_tabular_preprocessor(
                categorical_features=(1,), add_missing_indicators=True
            ),
        ),
        (
            "model",
            SnapBoostClassifier(num_iterations=2, random_state=3),
        ),
    ]).fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)
    assert probabilities.shape == (1, 2)
    assert np.all(np.isfinite(probabilities))


def test_tabular_preprocessor_can_scale_numeric_columns():
    X = np.array([[1.0, 10.0], [np.nan, 20.0], [3.0, 30.0]])
    transformer = make_tabular_preprocessor(scale_numeric=True)
    transformed = transformer.fit_transform(X)
    assert transformed.shape[0] == 3
    assert np.all(np.isfinite(transformed))


def test_tabular_preprocessor_treats_a_string_as_one_column_name():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({
        "num": [1.0, np.nan, 3.0],
        "cat": ["a", "b", "a"],
    })
    transformer = make_tabular_preprocessor(categorical_features="cat")
    transformed = transformer.fit_transform(frame)
    assert transformed.shape[0] == 3
    assert np.all(np.isfinite(transformed))


def test_tabular_preprocessor_accepts_a_single_column_index():
    X = np.array([[1.0, "a"], [2.0, "b"], [3.0, "a"]], dtype=object)
    transformer = make_tabular_preprocessor(categorical_features=1)
    transformed = transformer.fit_transform(X)
    assert transformed.shape[0] == 3
    assert np.all(np.isfinite(transformed))


def test_tabular_preprocessor_rejects_invalid_column_selectors():
    with pytest.raises(ValueError, match="categorical_features"):
        make_tabular_preprocessor(categorical_features=True)
    with pytest.raises(ValueError, match="categorical_features"):
        make_tabular_preprocessor(categorical_features=object())

