from numbers import Integral

from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from hnbm import HNBM, HNBMClassifier, HNBMRegressor
from .rff_learner import RandomFourierRidgeRegressor


class _SnapBoostMixin:
    """Shared SnapBoost learner pool configuration."""

    _REBUILD_PARAMS = frozenset({
        "p_tree", "alpha", "gamma", "n_components",
        "min_max_depth", "max_max_depth", "min_samples_leaf", "random_state",
    })

    def _init_snapboost_params(
        self,
        p_tree=0.9,
        min_max_depth=2,
        max_max_depth=4,
        min_samples_leaf=10,
        alpha=1.0,
        gamma=1.0,
        n_components=100,
    ):
        self.p_tree = p_tree
        self.min_max_depth = min_max_depth
        self.max_max_depth = max_max_depth
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.gamma = gamma
        self.n_components = n_components

    def _validate_snapboost_params(self):
        if not 0.0 <= self.p_tree <= 1.0:
            raise ValueError(f"p_tree must be between 0 and 1, got {self.p_tree}.")
        if not isinstance(self.min_max_depth, Integral) or self.min_max_depth < 1:
            raise ValueError(
                f"min_max_depth must be an integer >= 1, got {self.min_max_depth}."
            )
        if not isinstance(self.max_max_depth, Integral) or self.max_max_depth < 1:
            raise ValueError(
                f"max_max_depth must be an integer >= 1, got {self.max_max_depth}."
            )
        if self.min_max_depth > self.max_max_depth:
            raise ValueError("min_max_depth must be <= max_max_depth.")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}.")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {self.gamma}.")
        if self.n_components < 1:
            raise ValueError(
                f"n_components must be >= 1, got {self.n_components}."
            )
        if self.min_samples_leaf < 1:
            raise ValueError(
                f"min_samples_leaf must be >= 1, got {self.min_samples_leaf}."
            )

    def _build_base_learners(self):
        self.base_learners_ = []
        self.probabilities_ = []

        tree_seed = 42 if self.random_state is None else self.random_state
        depth_range = range(self.min_max_depth, 1 + self.max_max_depth)
        for d in depth_range:
            self.base_learners_.append(
                DecisionTreeRegressor(
                    max_depth=d,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=tree_seed + d,
                )
            )
            self.probabilities_.append(self.p_tree / len(depth_range))

        if self.p_tree < 1.0:
            rff_seed = None if self.random_state is None else self.random_state
            self.base_learners_.append(
                RandomFourierRidgeRegressor(
                    alpha=self.alpha,
                    gamma=self.gamma,
                    n_components=self.n_components,
                    random_state=rff_seed,
                )
            )
            self.probabilities_.append(1.0 - self.p_tree)

    def _set_snapboost_params(self, **params):
        rebuild = bool(self._REBUILD_PARAMS.intersection(params))
        result = super().set_params(**params)
        if rebuild:
            self._validate_snapboost_params()
            self._build_base_learners()
        return result


class SnapBoost(_SnapBoostMixin, HNBM):
    """
    A particular realization of a HNBM that uses decision trees and RFF ridge regressors.

    Prefer :class:`SnapBoostClassifier` or :class:`SnapBoostRegressor` for
    task-specific models without a ``mode`` parameter.

    Args:
        num_iterations (int): number of boosting iterations
        learning_rate (float): learning rate
        p_tree (float): probability of selecting a tree at each iteration
        min_max_depth (int): minimum maximum depth of a tree in the ensemble
        max_max_depth (int): maximum maximum depth of a tree in the ensemble
        min_samples_leaf (int): minimum samples per leaf for decision trees
        alpha (float): L2-regularization penalty in the ridge regression
        gamma (float): RBF-kernel parameter for random Fourier features
        n_components (int): number of random Fourier features
        mode (string): classification or regression
        random_state (int): random seed for tree fitting and learner selection
        verbose (bool): whether to show a progress bar during training
    """

    def __init__(
        self,
        num_iterations=100,
        learning_rate=0.1,
        p_tree=0.9,
        min_max_depth=2,
        max_max_depth=4,
        min_samples_leaf=10,
        alpha=1.0,
        gamma=1.0,
        n_components=100,
        mode="classification",
        random_state=None,
        verbose=False,
    ):
        self._init_snapboost_params(
            p_tree=p_tree,
            min_max_depth=min_max_depth,
            max_max_depth=max_max_depth,
            min_samples_leaf=min_samples_leaf,
            alpha=alpha,
            gamma=gamma,
            n_components=n_components,
        )
        super().__init__(num_iterations, learning_rate, mode, random_state, verbose)
        self._validate_snapboost_params()
        self._build_base_learners()

    def set_params(self, **params):
        return self._set_snapboost_params(**params)


class SnapBoostClassifier(_SnapBoostMixin, HNBMClassifier):
    """
    SnapBoost for binary classification.

    A heterogeneous Newton boosting machine that uses decision trees and
    random Fourier feature ridge regressors.
    """

    def __init__(
        self,
        num_iterations=100,
        learning_rate=0.1,
        p_tree=0.9,
        min_max_depth=2,
        max_max_depth=4,
        min_samples_leaf=10,
        alpha=1.0,
        gamma=1.0,
        n_components=100,
        random_state=None,
        verbose=False,
    ):
        self._init_snapboost_params(
            p_tree=p_tree,
            min_max_depth=min_max_depth,
            max_max_depth=max_max_depth,
            min_samples_leaf=min_samples_leaf,
            alpha=alpha,
            gamma=gamma,
            n_components=n_components,
        )
        super().__init__(
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=verbose,
        )
        self._validate_snapboost_params()
        self._build_base_learners()

    def set_params(self, **params):
        if "mode" in params:
            raise ValueError("mode cannot be set on SnapBoostClassifier.")
        return self._set_snapboost_params(**params)


class SnapBoostRegressor(_SnapBoostMixin, HNBMRegressor):
    """
    SnapBoost for regression.

    A heterogeneous Newton boosting machine that uses decision trees and
    random Fourier feature ridge regressors.
    """

    def __init__(
        self,
        num_iterations=100,
        learning_rate=0.1,
        p_tree=0.9,
        min_max_depth=2,
        max_max_depth=4,
        min_samples_leaf=10,
        alpha=1.0,
        gamma=1.0,
        n_components=100,
        random_state=None,
        verbose=False,
    ):
        self._init_snapboost_params(
            p_tree=p_tree,
            min_max_depth=min_max_depth,
            max_max_depth=max_max_depth,
            min_samples_leaf=min_samples_leaf,
            alpha=alpha,
            gamma=gamma,
            n_components=n_components,
        )
        super().__init__(
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=verbose,
        )
        self._validate_snapboost_params()
        self._build_base_learners()

    def set_params(self, **params):
        if "mode" in params:
            raise ValueError("mode cannot be set on SnapBoostRegressor.")
        return self._set_snapboost_params(**params)


class SnapBoost_KernelRidge(HNBM):
    """
    A particular realization of a HNBM that uses decision trees and kernel ridge regressors.
    Args:
        num_iterations (int): number of boosting iterations
        learning_rate (float): learning rate
        p_tree (float): probability of selecting a tree at each iteration
        min_max_depth (int): minimum maximum depth of a tree in the ensemble
        max_max_depth (int): maximum maximum depth of a tree in the ensemble
        min_samples_leaf (int): minimum samples per leaf for decision trees
        alpha (float): L2-regularization penalty in the ridge regression
        gamma (float): RBF-kernel parameter
        mode (string): classification or regression
        random_state (int): random seed for tree fitting and learner selection
        verbose (bool): whether to show a progress bar during training
    """
    _REBUILD_PARAMS = frozenset({
        "p_tree", "alpha", "gamma", "min_max_depth", "max_max_depth",
        "min_samples_leaf", "random_state",
    })

    def __init__(self, num_iterations=100, learning_rate=0.1, p_tree=0.9,
                 min_max_depth=2, max_max_depth=4, min_samples_leaf=10,
                 alpha=1.0, gamma=1.0,
                 mode="classification", random_state=None, verbose=True):

        self.p_tree = p_tree
        self.min_max_depth = min_max_depth
        self.max_max_depth = max_max_depth
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.gamma = gamma

        super().__init__(num_iterations, learning_rate, mode, random_state, verbose)
        self._validate_snapboost_params()
        self._build_base_learners()

    def _validate_snapboost_params(self):
        if not 0.0 <= self.p_tree <= 1.0:
            raise ValueError(f"p_tree must be between 0 and 1, got {self.p_tree}.")
        if not isinstance(self.min_max_depth, Integral) or self.min_max_depth < 1:
            raise ValueError(
                f"min_max_depth must be an integer >= 1, got {self.min_max_depth}."
            )
        if not isinstance(self.max_max_depth, Integral) or self.max_max_depth < 1:
            raise ValueError(
                f"max_max_depth must be an integer >= 1, got {self.max_max_depth}."
            )
        if self.min_max_depth > self.max_max_depth:
            raise ValueError("min_max_depth must be <= max_max_depth.")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}.")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {self.gamma}.")
        if self.min_samples_leaf < 1:
            raise ValueError(
                f"min_samples_leaf must be >= 1, got {self.min_samples_leaf}."
            )

    def _build_base_learners(self):
        self.base_learners_ = []
        self.probabilities_ = []

        tree_seed = 42 if self.random_state is None else self.random_state
        depth_range = range(self.min_max_depth, 1 + self.max_max_depth)
        for d in depth_range:
            self.base_learners_.append(
                DecisionTreeRegressor(
                    max_depth=d,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=tree_seed + d,
                )
            )
            self.probabilities_.append(self.p_tree / len(depth_range))

        if self.p_tree < 1.0:
            self.base_learners_.append(
                KernelRidge(alpha=self.alpha, kernel="rbf", gamma=self.gamma)
            )
            self.probabilities_.append(1.0 - self.p_tree)

    def set_params(self, **params):
        rebuild = bool(self._REBUILD_PARAMS.intersection(params))
        result = super().set_params(**params)
        if rebuild:
            self._validate_snapboost_params()
            self._build_base_learners()
        return result
