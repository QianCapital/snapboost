# SnapBoost Documentation

**SnapBoost** is an instance of a Heterogeneous Newton Boosting Machine (HNBM) — a generalized gradient boosting framework that supports the use of various types of learners aside from trees. Snapboost is an HNBM that mixes decision trees and kernel ridge regressors instead of trees alone. **SnapBoost** is scikit-learn compatible and built on [HNBM](https://github.com/qiancapital/hnbm).

At each boosting round, SnapBoost stochastically selects either a decision tree or an RFF ridge regressor. That mix captures both local, axis-aligned structure and smooth global patterns.

## New in 1.0.0

- Frozen `SnapBoostClassifier` / `SnapBoostRegressor` public API.
- sklearn estimator tags and `check_estimator` coverage via HNBM 1.0.
- Parameter validation at `fit`, matching the sklearn contract.
- `SnapBoost(mode=...)` and `SnapBoost_KernelRidge` are deprecated.

SnapBoost 1.0 requires HNBM 1.0 or newer. See [limitations](limitations.md)
for the binary-only, dense-input, and CART-tree contract.

Key properties:

- Scikit-learn API (`fit` / `predict` / `score`)
- Classification and regression estimators
- Heterogeneous base learners (trees + RFF ridge)
- Built on the HNBM framework

Inspired by [SnapBoost: A Heterogeneous Boosting Machine](https://arxiv.org/abs/2006.09745) (Parnell et al., NeurIPS 2020).

```bash
pip install snapboost
```

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from snapboost import SnapBoostClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = SnapBoostClassifier(num_iterations=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))
```

```{toctree}
:maxdepth: 2
:caption: Contents

installation
quickstart
api
parameters
limitations
examples
references
license
```
