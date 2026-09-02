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
Classification accepts binary and multiclass labels. Predictions retain the
original labels and probability columns follow `model.classes_`. Binary
`decision_function` is a vector of length `n_samples`; multiclass returns
shape `(n_samples, n_classes)`.
```

Multiclass example:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from snapboost import SnapBoostClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = SnapBoostClassifier(num_iterations=100, random_state=42)
model.fit(X_train, y_train)
print("Classes:", model.n_classes_)
print("Probabilities shape:", model.predict_proba(X_test).shape)  # (n_samples, 3)
```

## Staged prediction and feature importance

```python
staged = list(model.staged_predict(X_test))
importance = model.permutation_importance(X_test, y_test, n_repeats=5, random_state=42)
print(len(staged), importance.importances_mean.shape)
```

`staged_predict` (and classifier `staged_predict_proba` /
`staged_decision_function`) yields the ensemble after each boosting round.
`permutation_importance` is the feature-importance API for mixed tree and
kernel learners.

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

1. **Decision trees**, one candidate per `max_depth` in `[min_max_depth, max_max_depth]`
2. **RFF ridge regressors** for smooth, global fits, one candidate per `(kernel, gamma)` pair
3. **An optional linear ridge learner**, present only when `p_linear > 0`

The tree candidates share `p_tree` evenly, the kernel candidates share the
remaining `1 - p_tree - p_linear` evenly, and the linear learner takes
`p_linear`. Each selected learner is fit to the Newton direction (gradient /
Hessian, weighted by the Hessian). Multiclass rounds fit one scalar learner per
class from the same family.

Prefer `SnapBoostClassifier` / `SnapBoostRegressor` for new code. The legacy `SnapBoost(..., mode=...)` class remains available.

## Adaptive training and early stopping

```python
model = SnapBoostRegressor(
    num_iterations=500,
    learning_rate=0.05,
    selection_strategy="greedy",
    line_search=True,
    subsample=0.8,
    max_features=0.8,
    early_stopping_rounds=25,
    random_state=42,
)
model.fit(X_train, y_train, eval_set=(X_validation, y_validation))

print("Best iteration:", model.best_iteration_)
print("Validation loss:", model.history_["validation_loss"][-1])
```

`selection_strategy="random"` remains the default and reproduces the original
stochastic HNBM selection design. The greedy strategy fits every configured
candidate family and keeps the lowest-loss update, trading training time for a
more adaptive ensemble.

## Robust objectives and richer learner pools

```python
model = SnapBoostRegressor(
    p_tree=0.7,
    p_linear=0.1,
    kernel_gammas=(0.05, 0.5, 5.0),
    kernel_types=("rbf", "laplacian"),
    objective="quantile",
    objective_parameter=0.9,
    random_state=42,
)
model.fit(X_train, y_train, candidate_n_jobs=4)
```

All options above are disabled by default. With `p_linear=0`,
`kernel_gammas=None`, `kernel_types=("rbf",)`, and `objective="auto"`, the
existing learner pool and objective remain unchanged.

## Missing and categorical data

```python
from sklearn.pipeline import Pipeline
from snapboost import make_tabular_preprocessor

model = Pipeline([
    (
        "prepare",
        make_tabular_preprocessor(
            categorical_features=(1, 4),
            add_missing_indicators=True,
        ),
    ),
    ("snapboost", SnapBoostRegressor(random_state=42)),
])
model.fit(X_train, y_train)
```
