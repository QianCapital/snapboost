from numbers import Integral, Real

import numpy as np

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

    def _validate_snapboost_params(self, overrides=None):
        overrides = {} if overrides is None else overrides
        p_tree = overrides.get("p_tree", self.p_tree)
        min_max_depth = overrides.get("min_max_depth", self.min_max_depth)
        max_max_depth = overrides.get("max_max_depth", self.max_max_depth)
        min_samples_leaf = overrides.get(
            "min_samples_leaf", self.min_samples_leaf
        )
        alpha = overrides.get("alpha", self.alpha)
        gamma = overrides.get("gamma", self.gamma)
        n_components = overrides.get(
            "n_components", getattr(self, "n_components", None)
        )
        random_state = overrides.get("random_state", self.random_state)

        if (
            isinstance(p_tree, (bool, np.bool_))
            or not isinstance(p_tree, Real)
            or not np.isfinite(p_tree)
            or not 0.0 <= p_tree <= 1.0
        ):
            raise ValueError(f"p_tree must be between 0 and 1, got {p_tree}.")
        if (
            isinstance(min_max_depth, (bool, np.bool_))
            or not isinstance(min_max_depth, Integral)
            or min_max_depth < 1
        ):
            raise ValueError(
                f"min_max_depth must be an integer >= 1, got {min_max_depth}."
            )
        if (
            isinstance(max_max_depth, (bool, np.bool_))
            or not isinstance(max_max_depth, Integral)
            or max_max_depth < 1
        ):
            raise ValueError(
                f"max_max_depth must be an integer >= 1, got {max_max_depth}."
            )
        if min_max_depth > max_max_depth:
            raise ValueError("min_max_depth must be <= max_max_depth.")
        if (
            isinstance(alpha, (bool, np.bool_))
            or not isinstance(alpha, Real)
            or not np.isfinite(alpha)
            or alpha <= 0
        ):
            raise ValueError(f"alpha must be a finite number > 0, got {alpha}.")
        if (
            isinstance(gamma, (bool, np.bool_))
            or not isinstance(gamma, Real)
            or not np.isfinite(gamma)
            or gamma <= 0
        ):
            raise ValueError(f"gamma must be a finite number > 0, got {gamma}.")
        if n_components is not None:
            if (
                isinstance(n_components, (bool, np.bool_))
                or not isinstance(n_components, Integral)
                or n_components < 1
            ):
                raise ValueError(
                    f"n_components must be an integer >= 1, got {n_components}."
                )
        if (
            isinstance(min_samples_leaf, (bool, np.bool_))
            or not isinstance(min_samples_leaf, Integral)
            or min_samples_leaf < 1
        ):
            raise ValueError(
                "min_samples_leaf must be an integer >= 1, "
                f"got {min_samples_leaf}."
            )
        if random_state is not None and (
            isinstance(random_state, (bool, np.bool_))
            or not isinstance(random_state, Integral)
            or random_state < 0
        ):
            raise ValueError(
                "random_state must be a non-negative integer or None, "
                f"got {random_state}."
            )

    def _learner_seeds(self, count):
        if self.random_state is None:
            return [None] * count
        return [
            int(seed)
            for seed in np.random.SeedSequence(self.random_state).generate_state(count)
        ]

    def _make_non_tree_learner(self, random_state):
        return RandomFourierRidgeRegressor(
            alpha=self.alpha,
            gamma=self.gamma,
            n_components=self.n_components,
            random_state=random_state,
        )

    def _build_base_learners(self):
        self.base_learners_ = []
        self.probabilities_ = []

        depth_range = range(self.min_max_depth, 1 + self.max_max_depth)
        seeds = self._learner_seeds(len(depth_range) + 1)
        for d, tree_seed in zip(depth_range, seeds):
            self.base_learners_.append(
                DecisionTreeRegressor(
                    max_depth=d,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=tree_seed,
                )
            )
            self.probabilities_.append(self.p_tree / len(depth_range))

        if self.p_tree < 1.0:
            self.base_learners_.append(self._make_non_tree_learner(seeds[-1]))
            self.probabilities_.append(1.0 - self.p_tree)

    def _validate_set_param_keys(self, params):
        valid_params = self.get_params(deep=True)
        invalid = [key for key in params if key not in valid_params]
        if invalid:
            raise ValueError(
                f"Invalid parameter {invalid[0]} for estimator {self}. "
                "Check the list of available parameters with "
                "`estimator.get_params().keys()`."
            )

    def _set_snapboost_params(self, **params):
        self._validate_set_param_keys(params)
        rebuild = bool(self._REBUILD_PARAMS.intersection(params))
        self._validate_snapboost_params(params)
        result = super().set_params(**params)
        if rebuild:
            self._build_base_learners()
        return result

    def fit(self, X, y):
        self._validate_snapboost_params()
        self._build_base_learners()
        return super().fit(X, y)


class SnapBoost(_SnapBoostMixin, HNBM):
    """
    HNBM realization using decision trees and RFF ridge regressors.

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

    def set_params(self, **params):
        if "mode" in params:
            raise ValueError("mode cannot be set on SnapBoostRegressor.")
        return self._set_snapboost_params(**params)


class _KernelRidgePoolMixin(_SnapBoostMixin):
    """SnapBoost learner pool using exact RBF kernel ridge regression."""

    _REBUILD_PARAMS = _SnapBoostMixin._REBUILD_PARAMS - {"n_components"}

    def _init_kernel_ridge_params(
        self,
        *,
        p_tree,
        min_max_depth,
        max_max_depth,
        min_samples_leaf,
        alpha,
        gamma,
    ):
        self._init_snapboost_params(
            p_tree=p_tree,
            min_max_depth=min_max_depth,
            max_max_depth=max_max_depth,
            min_samples_leaf=min_samples_leaf,
            alpha=alpha,
            gamma=gamma,
            n_components=None,
        )

    def _make_non_tree_learner(self, random_state):
        del random_state
        return KernelRidge(alpha=self.alpha, kernel="rbf", gamma=self.gamma)


class SnapBoost_KernelRidge(_KernelRidgePoolMixin, HNBM):
    """
    HNBM realization using decision trees and exact kernel ridge regressors.
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
    def __init__(self, num_iterations=100, learning_rate=0.1, p_tree=0.9,
                 min_max_depth=2, max_max_depth=4, min_samples_leaf=10,
                 alpha=1.0, gamma=1.0,
                 mode="classification", random_state=None, verbose=True):

        self._init_kernel_ridge_params(
            p_tree=p_tree,
            min_max_depth=min_max_depth,
            max_max_depth=max_max_depth,
            min_samples_leaf=min_samples_leaf,
            alpha=alpha,
            gamma=gamma,
        )

        super().__init__(num_iterations, learning_rate, mode, random_state, verbose)
        self._validate_snapboost_params()

    def set_params(self, **params):
        return self._set_snapboost_params(**params)


class SnapBoostKernelRidgeClassifier(_KernelRidgePoolMixin, HNBMClassifier):
    """Binary SnapBoost classifier using exact RBF kernel ridge learners."""

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
        random_state=None,
        verbose=False,
    ):
        self._init_kernel_ridge_params(
            p_tree=p_tree,
            min_max_depth=min_max_depth,
            max_max_depth=max_max_depth,
            min_samples_leaf=min_samples_leaf,
            alpha=alpha,
            gamma=gamma,
        )
        super().__init__(
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=verbose,
        )
        self._validate_snapboost_params()

    def set_params(self, **params):
        if "mode" in params:
            raise ValueError("mode cannot be set on SnapBoostKernelRidgeClassifier.")
        return self._set_snapboost_params(**params)


class SnapBoostKernelRidgeRegressor(_KernelRidgePoolMixin, HNBMRegressor):
    """SnapBoost regressor using exact RBF kernel ridge learners."""

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
        random_state=None,
        verbose=False,
    ):
        self._init_kernel_ridge_params(
            p_tree=p_tree,
            min_max_depth=min_max_depth,
            max_max_depth=max_max_depth,
            min_samples_leaf=min_samples_leaf,
            alpha=alpha,
            gamma=gamma,
        )
        super().__init__(
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=verbose,
        )
        self._validate_snapboost_params()

    def set_params(self, **params):
        if "mode" in params:
            raise ValueError("mode cannot be set on SnapBoostKernelRidgeRegressor.")
        return self._set_snapboost_params(**params)
