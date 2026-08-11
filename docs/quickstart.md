# Quick Start

## Classification

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from snapboost import SnapBoostClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = SnapBoostClassifier(
    num_iterations=100,
    learning_rate=0.1,
    random_state=42,
)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
print("Probabilities shape:", model.predict_proba(X_test).shape)  # (n_samples, 2)
model.evaluate(X_test, y_test)  # prints log loss
```

```{note}
Classification labels may be `0`/`1` or `-1`/`+1`. Predictions are always returned as `0`/`1`.
```

## Regression

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from snapboost import SnapBoostRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = SnapBoostRegressor(
    num_iterations=100,
    learning_rate=0.1,
    random_state=42,
)
model.fit(X_train, y_train)

print("R²:", model.score(X_test, y_test))
model.evaluate(X_test, y_test)  # prints RMSE
```

## How it works

At each boosting iteration, SnapBoost samples a base learner from a fixed pool:

1. **Decision trees** with `max_depth` drawn uniformly from `[min_max_depth, max_max_depth]`
2. **One RFF ridge regressor** for smooth, global fits

Trees are chosen with probability `p_tree` (split evenly across depths); the ridge model with probability `1 - p_tree`. Each selected learner is fit to the Newton direction (gradient / Hessian, weighted by the Hessian).

Prefer `SnapBoostClassifier` / `SnapBoostRegressor` for new code. The legacy `SnapBoost(..., mode=...)` class remains available.
