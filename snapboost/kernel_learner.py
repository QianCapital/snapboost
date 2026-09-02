"""Weighted exact-kernel base learners for the specialized SnapBoost estimators."""

from __future__ import annotations

from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .rff_learner import is_allowed_gamma, resolve_kernel_gamma


class WeightedKernelRidgeRegressor(BaseEstimator, RegressorMixin):
    """Standardized weighted RBF kernel ridge for Newton working targets."""

    def __init__(self, alpha=1.0, gamma=1.0, scale_features=True):
        self.alpha = alpha
        self.gamma = gamma
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
        if not is_allowed_gamma(self.gamma):
            raise ValueError("gamma must be a finite number > 0 or 'scale'.")
        if not isinstance(self.scale_features, (bool, np.bool_)):
            raise ValueError("scale_features must be a boolean.")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.ndim == 0:
                sample_weight = np.full(X.shape[0], sample_weight.item())
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
        gamma_source = X
        if self.scale_features:
            scaler = StandardScaler()
            if sample_weight is None:
                scaler.fit(X)
            else:
                scaler.fit(X, sample_weight=sample_weight)
            gamma_source = scaler.transform(X)
        gamma = resolve_kernel_gamma(self.gamma, gamma_source)
        steps = []
        if self.scale_features:
            steps.append(("scale", StandardScaler()))
        steps.append((
            "ridge",
            KernelRidge(alpha=self.alpha, kernel="rbf", gamma=gamma),
        ))
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
