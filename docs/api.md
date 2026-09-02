# API Reference

## SnapBoostClassifier / SnapBoostRegressor

Recommended entry points (analogous to `XGBClassifier` / `XGBRegressor`).

```python
from snapboost import SnapBoostClassifier, SnapBoostRegressor

clf = SnapBoostClassifier(
    num_iterations=100,
    learning_rate=0.1,
    p_tree=0.9,
    min_max_depth=2,
    max_max_depth=4,
    alpha=1.0,
    gamma=1.0,
    random_state=42,
    verbose=True,
)
clf.fit(X, y)

reg = SnapBoostRegressor(num_iterations=100, random_state=42)
reg.fit(X, y)
```

### Methods

| Method | Classifier | Regressor | Description |
|--------|:----------:|:---------:|-------------|
| `fit(X, y, sample_weight=None, eval_set=None, *, eval_sample_weight=None)` | ✓ | ✓ | Train, optionally with weights and validation data |
| `predict(X)` | ✓ | ✓ | Class labels from `classes_` or continuous values |
| `predict_proba(X)` | ✓ | | Class probabilities, shape `(n_samples, n_classes)` |
| `decision_function(X)` | ✓ | | Raw logits: `(n_samples,)` binary, `(n_samples, n_classes)` multiclass |
| `staged_predict(X)` | ✓ | ✓ | Predictions after each boosting round |
| `permutation_importance(X, y)` | ✓ | ✓ | Permutation importance of original features |
| `score(X, y)` | ✓ | ✓ | Accuracy or R² |
| `evaluate(X, y)` | ✓ | ✓ | Prints and returns log loss or RMSE |

```{eval-rst}
.. autoclass:: snapboost.SnapBoostClassifier
   :members: fit, predict, predict_proba, decision_function, staged_predict, staged_predict_proba, staged_decision_function, permutation_importance, score, evaluate, set_params
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: snapboost.SnapBoostRegressor
   :members: fit, predict, staged_predict, permutation_importance, score, evaluate, set_params
   :undoc-members:
   :show-inheritance:
   :no-index:
```

Fitted adaptive models expose `base_score_`, `learner_weights_`, `history_`,
`best_iteration_`, `n_iter_`, and for classifiers `classes_` / `n_classes_`.
Multiclass `ensemble_` entries are one fitted scalar learner per class. When `early_stopping_rounds` triggers, the
best validation ensemble is restored before `fit` returns. Passing an
`eval_set` without `early_stopping_rounds` still records `best_iteration_`, but
no learners are discarded, so predictions use all `n_iter_` of them.

Models also inherit `compact(min_abs_weight=0.0, inplace=False)` from HNBM.
Compaction is explicit and never runs automatically.

## Tabular preprocessing

`make_tabular_preprocessor()` returns a dense scikit-learn `ColumnTransformer`
that median-imputes numeric data, adds optional missingness indicators, and
imputes/one-hot encodes categorical data with unseen-category support. Use it in
a normal `Pipeline`; it does not modify SnapBoost internals.


## SnapBoost (deprecated)

Accepts a `mode` parameter (`"classification"` or `"regression"`). This class
emits `FutureWarning` and will be removed in 2.0. Prefer the task-specific
classes above.

```python
from snapboost import SnapBoost

model = SnapBoost(
    num_iterations=100,
    learning_rate=0.1,
    p_tree=0.9,
    mode="classification",
    random_state=42,
)
model.fit(X, y)
```

```{eval-rst}
.. autoclass:: snapboost.SnapBoost
   :members: fit, predict, predict_proba, decision_function, score, evaluate, set_params
   :undoc-members:
   :show-inheritance:
   :no-index:
```

## RandomFourierRidgeRegressor

Ridge regression on random Fourier features approximating RBF or Laplacian kernels. Used as a non-tree learner in the SnapBoost pool.

```{eval-rst}
.. autoclass:: snapboost.RandomFourierRidgeRegressor
   :members: fit, predict
   :undoc-members:
   :show-inheritance:
   :no-index:
```

## Exact kernel ridge estimators

`SnapBoostKernelRidgeClassifier` and `SnapBoostKernelRidgeRegressor` swap the
random Fourier feature learner for an exact RBF kernel ridge learner. They keep
a frozen constructor surface (`num_iterations`, `learning_rate`, `p_tree`,
`min_max_depth`, `max_max_depth`, `min_samples_leaf`, `alpha`, `gamma`,
`random_state`, `verbose`) and so do not expose greedy selection, line search,
subsampling, or early stopping. Classification inherits binary logistic loss
and multiclass softmax from HNBM. Memory grows quadratically in the number of
samples, so prefer the RFF path unless an exact kernel is required.

```python
from snapboost import SnapBoostKernelRidgeClassifier

clf = SnapBoostKernelRidgeClassifier(num_iterations=50, gamma=0.5, random_state=42)
clf.fit(X, y)
```

```{eval-rst}
.. autoclass:: snapboost.SnapBoostKernelRidgeClassifier
   :members: fit, predict, predict_proba, decision_function, staged_predict, staged_predict_proba, staged_decision_function, permutation_importance, score, evaluate, set_params
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: snapboost.SnapBoostKernelRidgeRegressor
   :members: fit, predict, staged_predict, permutation_importance, score, evaluate, set_params
   :undoc-members:
   :show-inheritance:
   :no-index:
```

`SnapBoost_KernelRidge` is the deprecated `mode`-based equivalent. It emits
`FutureWarning` and will be removed in 2.0.

## Optional learner and preprocessing helpers

```{eval-rst}
.. autoclass:: snapboost.WeightedLinearRegressor
   :members: fit, predict
   :show-inheritance:

.. autoclass:: snapboost.WeightedKernelRidgeRegressor
   :members: fit, predict
   :show-inheritance:

.. autoclass:: snapboost.LaplacianSampler
   :members: fit, transform
   :show-inheritance:

.. autofunction:: snapboost.make_tabular_preprocessor
```

## HNBM

Abstract base classes for custom heterogeneous ensembles are provided by the [`hnbm`](https://pypi.org/project/hnbm/) package. Subclass and configure `base_learners_` / `probabilities_` before calling `fit`:

```python
from sklearn.tree import DecisionTreeRegressor
from hnbm import HNBMClassifier

class MyClassifier(HNBMClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_learners_ = [DecisionTreeRegressor(max_depth=5)]
        self.probabilities_ = [1.0]
```
