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
| `fit(X, y, sample_weight=None, eval_set=None)` | ✓ | ✓ | Train, optionally with weights and validation data |
| `predict(X)` | ✓ | ✓ | Class labels (0/1) or continuous values |
| `predict_proba(X)` | ✓ | | Class probabilities, shape `(n_samples, 2)` |
| `decision_function(X)` | ✓ | | Raw logits |
| `score(X, y)` | ✓ | ✓ | Accuracy or R² |
| `evaluate(X, y)` | ✓ | ✓ | Prints and returns log loss or RMSE |

```{eval-rst}
.. autoclass:: snapboost.SnapBoostClassifier
   :members: fit, predict, predict_proba, decision_function, score, evaluate, set_params
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: snapboost.SnapBoostRegressor
   :members: fit, predict, score, evaluate, set_params
   :undoc-members:
   :show-inheritance:
   :no-index:
```

Fitted adaptive models expose `base_score_`, `learner_weights_`, `history_`,
`best_iteration_`, and `n_iter_`. When `early_stopping_rounds` triggers, the
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


## SnapBoost (legacy)

Accepts a `mode` parameter (`"classification"` or `"regression"`). Prefer the task-specific classes above.

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

## Optional learner and preprocessing helpers

```{eval-rst}
.. autoclass:: snapboost.WeightedLinearRegressor
   :members: fit, predict
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
