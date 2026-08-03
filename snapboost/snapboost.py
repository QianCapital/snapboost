from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from hnbm import HNBM


class SnapBoost(HNBM):
    """
    A particular realization of a HNBM that uses decision trees and kernel ridge regressors
    Args:
        num_iterations (int): number of boosting iterations
        learning_rate (float): learning rate
        p_tree (float): probability of selecting a tree at each iteration
        min_max_depth (int): minimum maximum depth of a tree in the ensemble
        max_max_depth (int): maximum maximum depth of a tree in the ensemble
        alpha (float): L2-regularization penalty in the ridge regression
        gamma (float): RBF-kernel parameter
        mode (string): classification or regression
        random_state (int): random seed for tree fitting and learner selection
        verbose (bool): whether to show a progress bar during training
    """
    _REBUILD_PARAMS = frozenset({
        "p_tree", "alpha", "gamma", "min_max_depth", "max_max_depth", "random_state",
    })

    def __init__(self, num_iterations=100, learning_rate=0.1, p_tree=0.8,
                 min_max_depth=4, max_max_depth=8, alpha=1.0, gamma=1.0,
                 mode="classification", random_state=None, verbose=True):

        self.p_tree = p_tree
        self.min_max_depth = min_max_depth
        self.max_max_depth = max_max_depth
        self.alpha = alpha
        self.gamma = gamma

        super().__init__(num_iterations, learning_rate, mode, random_state, verbose)
        self._validate_snapboost_params()
        self._build_base_learners()

    def _validate_snapboost_params(self):
        if not 0.0 <= self.p_tree <= 1.0:
            raise ValueError(f"p_tree must be between 0 and 1, got {self.p_tree}.")
        if self.min_max_depth < 0:
            raise ValueError(
                f"min_max_depth must be >= 0, got {self.min_max_depth}."
            )
        if self.min_max_depth > self.max_max_depth:
            raise ValueError("min_max_depth must be <= max_max_depth.")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}.")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {self.gamma}.")

    def _build_base_learners(self):
        self.base_learners_ = []
        self.probabilities_ = []

        tree_seed = 42 if self.random_state is None else self.random_state
        depth_range = range(self.min_max_depth, 1 + self.max_max_depth)
        for d in depth_range:
            self.base_learners_.append(
                DecisionTreeRegressor(max_depth=d, random_state=tree_seed + d)
            )
            self.probabilities_.append(self.p_tree / len(depth_range))

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
