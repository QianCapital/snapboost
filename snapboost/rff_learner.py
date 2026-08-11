from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class LaplacianSampler(BaseEstimator, TransformerMixin):
    """Random Fourier features for the Laplacian (L1 exponential) kernel."""

    def __init__(self, gamma=1.0, n_components=100, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.random_weights_ = rng.standard_cauchy(
            size=(X.shape[1], self.n_components)
        ) * self.gamma
        self.random_offset_ = rng.uniform(0.0, 2.0 * np.pi, self.n_components)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, ["random_weights_", "random_offset_"])
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but sampler was trained with "
                f"{self.n_features_in_}."
            )
        projection = X @ self.random_weights_ + self.random_offset_
        return np.sqrt(2.0 / self.n_components) * np.cos(projection)


class RandomFourierRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Ridge regression on random Fourier features approximating an RBF kernel.

    Matches the linear + RFF base learner used in the original SnapBoost paper,
    scaling linearly in the number of samples instead of exact KernelRidge.
    Features are standardized by default because RBF distances are sensitive to
    input scale. Set ``scale_features=False`` for pre-scaled inputs.
    """
    def __init__(
        self, alpha=1.0, gamma=1.0, n_components=100, random_state=None,
        scale_features=True, kernel="rbf",
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        self.scale_features = scale_features
        self.kernel = kernel

    def _make_pipeline(self):
        steps = []
        if self.scale_features:
            steps.append(("scale", StandardScaler()))
        if self.kernel == "rbf":
            sampler = RBFSampler(
                gamma=self.gamma,
                n_components=self.n_components,
                random_state=self.random_state,
            )
        else:
            sampler = LaplacianSampler(
                gamma=self.gamma,
                n_components=self.n_components,
                random_state=self.random_state,
            )
        steps.extend([
            (
                "rff",
                sampler,
            ),
            ("ridge", Ridge(alpha=self.alpha)),
        ])
        return Pipeline(steps)

    def _validate_params(self):
        if not isinstance(self.scale_features, (bool, np.bool_)):
            raise ValueError("scale_features must be a boolean.")
        if self.kernel not in ("rbf", "laplacian"):
            raise ValueError("kernel must be 'rbf' or 'laplacian'.")
        if (
            isinstance(self.alpha, (bool, np.bool_))
            or not isinstance(self.alpha, Real)
            or not np.isfinite(self.alpha)
            or self.alpha <= 0
        ):
            raise ValueError(
                f"alpha must be a finite number > 0, got {self.alpha}."
            )
        if (
            isinstance(self.gamma, (bool, np.bool_))
            or not isinstance(self.gamma, Real)
            or not np.isfinite(self.gamma)
            or self.gamma <= 0
        ):
            raise ValueError(
                f"gamma must be a finite number > 0, got {self.gamma}."
            )
        if (
            isinstance(self.n_components, (bool, np.bool_))
            or not isinstance(self.n_components, Integral)
            or self.n_components < 1
        ):
            raise ValueError(
                "n_components must be an integer >= 1, "
                f"got {self.n_components}."
            )
        if self.random_state is not None and (
            isinstance(self.random_state, (bool, np.bool_))
            or not isinstance(self.random_state, Integral)
            or not 0 <= self.random_state <= np.iinfo(np.uint32).max
        ):
            raise ValueError(
                "random_state must be an integer between 0 and 2**32 - 1 "
                f"or None, got {self.random_state}."
            )

    def fit(self, X, y, sample_weight=None):
        self._validate_params()
        X, y = check_X_y(X, y, dtype=float, y_numeric=True)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.ndim == 0:
                sample_weight = np.full(X.shape[0], sample_weight.item())
            if sample_weight.ndim != 1 or sample_weight.shape[0] != X.shape[0]:
                raise ValueError(
                    "sample_weight must be a scalar or a one-dimensional array "
                    "with one value per sample."
                )
            if not np.all(np.isfinite(sample_weight)):
                raise ValueError("sample_weight must contain only finite values.")
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight must be non-negative.")
            weight_sum = sample_weight.sum()
            if weight_sum <= 0:
                raise ValueError("sample_weight must have a positive total weight.")
            # Ridge regularization depends on the absolute weight scale. Normalize
            # over observations that contribute to the fit so uniformly scaling
            # weights has no effect and zero-weight rows are equivalent to removal.
            positive_weight_mask = sample_weight > 0
            positive_weight_count = np.count_nonzero(positive_weight_mask)
            sample_weight = sample_weight * (positive_weight_count / weight_sum)
            X = X[positive_weight_mask]
            y = y[positive_weight_mask]
            sample_weight = sample_weight[positive_weight_mask]
        pipeline = self._make_pipeline()
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
                f"{self.n_features_in_} features."
            )
        return self.pipeline_.predict(X)
