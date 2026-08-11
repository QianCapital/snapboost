"""Weighted linear base learners for optional SnapBoost pools."""

from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class WeightedLinearRegressor(BaseEstimator, RegressorMixin):
    """Standardized weighted ridge regression for Newton working targets."""

    def __init__(self, alpha=1.0, scale_features=True):
        self.alpha = alpha
        self.scale_features = scale_features

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, dtype=float, y_numeric=True)
        if (
            isinstance(self.alpha, (bool, np.bool_))
            or not isinstance(self.alpha, Real)
            or not np.isfinite(self.alpha)
            or self.alpha <= 0
        ):
            raise ValueError("alpha must be a finite number > 0.")
        if not isinstance(self.scale_features, (bool, np.bool_)):
            raise ValueError("scale_features must be a boolean.")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.ndim != 1 or sample_weight.shape[0] != X.shape[0]:
                raise ValueError("sample_weight must have one value per sample.")
            if not np.all(np.isfinite(sample_weight)) or np.any(sample_weight < 0):
                raise ValueError("sample_weight must be finite and non-negative.")
            positive = sample_weight > 0
            if not np.any(positive):
                raise ValueError("sample_weight must have a positive total weight.")
            sample_weight = sample_weight[positive]
            sample_weight *= positive.sum() / sample_weight.sum()
            X, y = X[positive], y[positive]
        steps = []
        if self.scale_features:
            steps.append(("scale", StandardScaler()))
        steps.append(("ridge", Ridge(alpha=self.alpha)))
        pipeline = Pipeline(steps)
        fit_params = {}
        if sample_weight is not None:
            fit_params["ridge__sample_weight"] = sample_weight
            if self.scale_features:
                fit_params["scale__sample_weight"] = sample_weight
        pipeline.fit(X, y, **fit_params)
        self.pipeline_ = pipeline
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["pipeline_", "n_features_in_"])
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained with "
                f"{self.n_features_in_}."
            )
        return self.pipeline_.predict(X)
