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
| `fit(X, y)` | ✓ | ✓ | Train the ensemble |
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

Ridge regression on random Fourier features approximating an RBF kernel. Used as the non-tree learner in the SnapBoost pool.

```{eval-rst}
.. autoclass:: snapboost.RandomFourierRidgeRegressor
   :members: fit, predict
   :undoc-members:
   :show-inheritance:
   :no-index:
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
