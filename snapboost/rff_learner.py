from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


class RandomFourierRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Ridge regression on random Fourier features approximating an RBF kernel.

    Matches the linear + RFF base learner used in the original SnapBoost paper,
    scaling linearly in the number of samples instead of exact KernelRidge.
    """
    def __init__(self, alpha=1.0, gamma=1.0, n_components=100, random_state=None):
        self.alpha = alpha
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def _make_pipeline(self):
        return Pipeline([
            (
                "rff",
                RBFSampler(
                    gamma=self.gamma,
                    n_components=self.n_components,
                    random_state=self.random_state,
                ),
            ),
            ("ridge", Ridge(alpha=self.alpha)),
        ])

    def fit(self, X, y, sample_weight=None):
        self.pipeline_ = self._make_pipeline()
        fit_params = {}
        if sample_weight is not None:
            fit_params["ridge__sample_weight"] = sample_weight
        self.pipeline_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        check_is_fitted(self, "pipeline_")
        return self.pipeline_.predict(X)
