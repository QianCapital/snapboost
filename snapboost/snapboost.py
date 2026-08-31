import inspect
import warnings
from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np
from hnbm import HNBM, HNBMClassifier, HNBMRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor

from .linear_learner import WeightedLinearRegressor
from .rff_learner import RandomFourierRidgeRegressor

# ``1 - p_tree - p_linear`` can leave a rounding residue near 1e-17 even when the
# two probabilities already sum to one. Anything below this counts as an empty
# family, so greedy selection never evaluates a learner the caller excluded.
_MIN_FAMILY_PROBABILITY = 1e-12


class _SnapBoostMixin:
    """Shared SnapBoost learner pool configuration."""

    _REBUILD_PARAMS = frozenset({
        "p_tree", "p_linear", "alpha", "gamma", "n_components",
        "min_max_depth", "max_max_depth", "min_samples_leaf", "random_state",
        "max_features", "scale_features",
        "kernel_gammas", "kernel_types",
        "monotonic_cst",
    })

    def _init_snapboost_params(
        self,
        p_tree=0.9,
        p_linear=0.0,
        min_max_depth=2,
        max_max_depth=4,
        min_samples_leaf=10,
        alpha=1.0,
        gamma=1.0,
        n_components=100,
        max_features=None,
        scale_features=True,
        kernel_gammas=None,
        kernel_types=("rbf",),
        monotonic_cst=None,
    ):
        self.p_tree = p_tree
        self.p_linear = p_linear
        self.min_max_depth = min_max_depth
        self.max_max_depth = max_max_depth
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.gamma = gamma
        self.n_components = n_components
        self.max_features = max_features
        self.scale_features = scale_features
        self.kernel_gammas = kernel_gammas
        self.kernel_types = kernel_types
        self.monotonic_cst = monotonic_cst

    def _validate_snapboost_params(self, overrides=None):
        overrides = {} if overrides is None else overrides
        p_tree = overrides.get("p_tree", self.p_tree)
        p_linear = overrides.get("p_linear", self.p_linear)
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
        max_features = overrides.get("max_features", self.max_features)
        scale_features = overrides.get("scale_features", self.scale_features)
        kernel_gammas = overrides.get("kernel_gammas", self.kernel_gammas)
        kernel_types = overrides.get("kernel_types", self.kernel_types)
        monotonic_cst = overrides.get("monotonic_cst", self.monotonic_cst)

        if (
            isinstance(p_tree, (bool, np.bool_))
            or not isinstance(p_tree, Real)
            or not np.isfinite(p_tree)
            or not 0.0 <= p_tree <= 1.0
        ):
            raise ValueError(f"p_tree must be between 0 and 1, got {p_tree}.")
        if (
            isinstance(p_linear, (bool, np.bool_))
            or not isinstance(p_linear, Real)
            or not np.isfinite(p_linear)
            or not 0.0 <= p_linear <= 1.0
        ):
            raise ValueError(f"p_linear must be between 0 and 1, got {p_linear}.")
        if p_tree + p_linear > 1.0:
            raise ValueError("p_tree + p_linear must be <= 1.")
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
        if n_components is not None and (
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
        if max_features is None:
            valid_max_features = True
        elif isinstance(max_features, str):
            valid_max_features = max_features in ("sqrt", "log2")
        elif isinstance(max_features, (bool, np.bool_)):
            valid_max_features = False
        elif isinstance(max_features, Integral):
            valid_max_features = max_features >= 1
        elif isinstance(max_features, Real):
            valid_max_features = (
                np.isfinite(max_features) and 0 < max_features <= 1
            )
        else:
            valid_max_features = False
        if not valid_max_features:
            raise ValueError(
                "max_features must be None, 'sqrt', 'log2', an integer >= 1, "
                "or a float in (0, 1]."
            )
        if not isinstance(scale_features, (bool, np.bool_)):
            raise ValueError("scale_features must be a boolean.")
        if kernel_gammas is not None and (
            isinstance(kernel_gammas, (str, bytes))
            or not isinstance(kernel_gammas, (Sequence, np.ndarray))
            or len(kernel_gammas) == 0
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value <= 0
                for value in kernel_gammas
            )
        ):
            raise ValueError(
                "kernel_gammas must be a nonempty sequence of positive numbers "
                "or None."
            )
        if (
            isinstance(kernel_types, (str, bytes))
            or not isinstance(kernel_types, (Sequence, np.ndarray))
            or len(kernel_types) == 0
            or any(value not in ("rbf", "laplacian") for value in kernel_types)
        ):
            raise ValueError(
                "kernel_types must be a nonempty sequence containing 'rbf' "
                "and/or 'laplacian'."
            )
        if monotonic_cst is not None and (
            isinstance(monotonic_cst, (str, bytes))
            or not isinstance(monotonic_cst, (Sequence, np.ndarray))
            or any(value not in (-1, 0, 1) for value in monotonic_cst)
        ):
            raise ValueError(
                "monotonic_cst must contain only -1, 0, and 1, or be None."
            )

    def _learner_seeds(self, count):
        if self.random_state is None:
            return [None] * count
        return [
            int(seed)
            for seed in np.random.SeedSequence(self.random_state).generate_state(count)
        ]

    def _make_non_tree_learner(self, random_state, gamma, kernel):
        return RandomFourierRidgeRegressor(
            alpha=self.alpha,
            gamma=gamma,
            n_components=self.n_components,
            random_state=random_state,
            scale_features=self.scale_features,
            kernel=kernel,
        )

    def _make_linear_learner(self):
        return WeightedLinearRegressor(
            alpha=self.alpha,
            scale_features=self.scale_features,
        )

    def _build_base_learners(self):
        self.base_learners_ = []
        self.probabilities_ = []

        depth_range = range(self.min_max_depth, 1 + self.max_max_depth)
        gammas = (
            (self.gamma,) if self.kernel_gammas is None else tuple(self.kernel_gammas)
        )
        kernel_specs = [
            (kernel, gamma) for kernel in self.kernel_types for gamma in gammas
        ]
        linear_count = int(self.p_linear > 0.0)
        seeds = self._learner_seeds(
            len(depth_range) + len(kernel_specs) + linear_count
        )
        if self.p_tree > 0.0:
            tree_options = {}
            if self.monotonic_cst is not None:
                if "monotonic_cst" not in inspect.signature(
                    DecisionTreeRegressor
                ).parameters:
                    raise RuntimeError(
                        "monotonic_cst requires a scikit-learn version whose "
                        "DecisionTreeRegressor supports monotonic constraints."
                    )
                tree_options["monotonic_cst"] = self.monotonic_cst
            for d, tree_seed in zip(depth_range, seeds):
                self.base_learners_.append(
                    DecisionTreeRegressor(
                        max_depth=d,
                        min_samples_leaf=self.min_samples_leaf,
                        max_features=self.max_features,
                        random_state=tree_seed,
                        **tree_options,
                    )
                )
                self.probabilities_.append(self.p_tree / len(depth_range))

        kernel_probability = 1.0 - self.p_tree - self.p_linear
        if kernel_probability > _MIN_FAMILY_PROBABILITY:
            non_tree_probability = kernel_probability / len(kernel_specs)
            kernel_seeds = seeds[len(depth_range):]
            for (kernel, gamma), seed in zip(kernel_specs, kernel_seeds):
                self.base_learners_.append(
                    self._make_non_tree_learner(seed, gamma, kernel)
                )
                self.probabilities_.append(non_tree_probability)
        if self.p_linear > 0.0:
            self.base_learners_.append(self._make_linear_learner())
            self.probabilities_.append(self.p_linear)

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
        result = super().set_params(**params)
        try:
            self._validate_snapboost_params()
        except (TypeError, ValueError):
            return result
        rebuild = bool(self._REBUILD_PARAMS.intersection(params))
        if rebuild or not self.base_learners_:
            self._build_base_learners()
        return result

    def fit(
        self, X, y, sample_weight=None, eval_set=None,
        eval_metric=None, callbacks=None, candidate_n_jobs=1,
    ):
        self._validate_snapboost_params()
        self._build_base_learners()
        return super().fit(
            X,
            y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            eval_metric=eval_metric,
            callbacks=callbacks,
            candidate_n_jobs=candidate_n_jobs,
        )


class SnapBoost(_SnapBoostMixin, HNBM):
    """
    HNBM realization using decision trees and RFF ridge regressors.

    Deprecated. Prefer :class:`SnapBoostClassifier` or
    :class:`SnapBoostRegressor` for task-specific models without a ``mode``
    parameter.
    """

    def __init__(
        self,
        num_iterations=100,
        learning_rate=0.1,
        p_tree=0.9,
        p_linear=0.0,
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
            p_linear=p_linear,
            min_max_depth=min_max_depth,
            max_max_depth=max_max_depth,
            min_samples_leaf=min_samples_leaf,
            alpha=alpha,
            gamma=gamma,
            n_components=n_components,
        )
        super().__init__(num_iterations, learning_rate, mode, random_state, verbose)
        warnings.warn(
            "SnapBoost(mode=...) is deprecated and will be removed in a "
            "future release. Use SnapBoostClassifier or SnapBoostRegressor "
            "instead.",
            FutureWarning,
            stacklevel=2,
        )

    def set_params(self, **params):
        return self._set_snapboost_params(**params)


class SnapBoostClassifier(_SnapBoostMixin, HNBMClassifier):
    """
    SnapBoost for binary classification.

    A heterogeneous Newton boosting machine that uses decision trees and
    random Fourier feature ridge regressors. Supports random or greedy learner
    selection, per-round line search, subsampling, observation weights,
    validation histories, and early stopping.
    """

    def __init__(
        self,
        num_iterations=100,
        learning_rate=0.1,
        p_tree=0.9,
        p_linear=0.0,
        min_max_depth=2,
        max_max_depth=4,
        min_samples_leaf=10,
        alpha=1.0,
        gamma=1.0,
        n_components=100,
        max_features=None,
        scale_features=True,
        kernel_gammas=None,
        kernel_types=("rbf",),
        monotonic_cst=None,
        random_state=None,
        verbose=False,
        selection_strategy="random",
        line_search=False,
        early_stopping_rounds=None,
        min_delta=0.0,
        subsample=1.0,
        objective="auto",
        objective_parameter=None,
    ):
        self._init_snapboost_params(
            p_tree=p_tree,
            p_linear=p_linear,
            min_max_depth=min_max_depth,
            max_max_depth=max_max_depth,
            min_samples_leaf=min_samples_leaf,
            alpha=alpha,
            gamma=gamma,
            n_components=n_components,
            max_features=max_features,
            scale_features=scale_features,
            kernel_gammas=kernel_gammas,
            kernel_types=kernel_types,
            monotonic_cst=monotonic_cst,
        )
        super().__init__(
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=verbose,
            selection_strategy=selection_strategy,
            line_search=line_search,
            early_stopping_rounds=early_stopping_rounds,
            min_delta=min_delta,
            subsample=subsample,
            objective=objective,
            objective_parameter=objective_parameter,
        )

    def set_params(self, **params):
        if "mode" in params:
            raise ValueError("mode cannot be set on SnapBoostClassifier.")
        return self._set_snapboost_params(**params)


class SnapBoostRegressor(_SnapBoostMixin, HNBMRegressor):
    """
    SnapBoost for regression.

    A heterogeneous Newton boosting machine that uses decision trees and
    random Fourier feature ridge regressors. Supports random or greedy learner
    selection, per-round line search, subsampling, observation weights,
    validation histories, and early stopping.
    """

    def __init__(
        self,
        num_iterations=100,
        learning_rate=0.1,
        p_tree=0.9,
        p_linear=0.0,
        min_max_depth=2,
        max_max_depth=4,
        min_samples_leaf=10,
        alpha=1.0,
        gamma=1.0,
        n_components=100,
        max_features=None,
        scale_features=True,
        kernel_gammas=None,
        kernel_types=("rbf",),
        monotonic_cst=None,
        random_state=None,
        verbose=False,
        selection_strategy="random",
        line_search=False,
        early_stopping_rounds=None,
        min_delta=0.0,
        subsample=1.0,
        objective="auto",
        objective_parameter=None,
    ):
        self._init_snapboost_params(
            p_tree=p_tree,
            p_linear=p_linear,
            min_max_depth=min_max_depth,
            max_max_depth=max_max_depth,
            min_samples_leaf=min_samples_leaf,
            alpha=alpha,
            gamma=gamma,
            n_components=n_components,
            max_features=max_features,
            scale_features=scale_features,
            kernel_gammas=kernel_gammas,
            kernel_types=kernel_types,
            monotonic_cst=monotonic_cst,
        )
        super().__init__(
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=verbose,
            selection_strategy=selection_strategy,
            line_search=line_search,
            early_stopping_rounds=early_stopping_rounds,
            min_delta=min_delta,
            subsample=subsample,
            objective=objective,
            objective_parameter=objective_parameter,
        )

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

    def _make_non_tree_learner(self, random_state, gamma, kernel):
        del random_state
        if kernel != "rbf":
            raise ValueError("Exact kernel ridge supports only the RBF kernel.")
        return KernelRidge(alpha=self.alpha, kernel="rbf", gamma=gamma)


class SnapBoost_KernelRidge(_KernelRidgePoolMixin, HNBM):
    """Deprecated exact-kernel estimator. Use the task-specific classes."""

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
        warnings.warn(
            "SnapBoost_KernelRidge is deprecated and will be removed in a "
            "future release. Use SnapBoostKernelRidgeClassifier or "
            "SnapBoostKernelRidgeRegressor instead.",
            FutureWarning,
            stacklevel=2,
        )

    def set_params(self, **params):
        return self._set_snapboost_params(**params)


class SnapBoostKernelRidgeClassifier(_KernelRidgePoolMixin, HNBMClassifier):
    """Binary SnapBoost classifier using exact RBF kernel ridge learners.

    This specialized surface is frozen in 1.0: it does not expose the adaptive
    training controls of :class:`SnapBoostClassifier`. Prefer the RFF-based
    classifier unless an exact kernel is required.
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

    def set_params(self, **params):
        if "mode" in params:
            raise ValueError("mode cannot be set on SnapBoostKernelRidgeClassifier.")
        return self._set_snapboost_params(**params)


class SnapBoostKernelRidgeRegressor(_KernelRidgePoolMixin, HNBMRegressor):
    """SnapBoost regressor using exact RBF kernel ridge learners.

    This specialized surface is frozen in 1.0: it does not expose the adaptive
    training controls of :class:`SnapBoostRegressor`. Prefer the RFF-based
    regressor unless an exact kernel is required.
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

    def set_params(self, **params):
        if "mode" in params:
            raise ValueError("mode cannot be set on SnapBoostKernelRidgeRegressor.")
        return self._set_snapboost_params(**params)
