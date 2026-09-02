# SnapBoost Documentation

**SnapBoost** is an instance of a Heterogeneous Newton Boosting Machine (HNBM) — a generalized gradient boosting framework that supports the use of various types of learners aside from trees. Snapboost is an HNBM that mixes decision trees and kernel ridge regressors instead of trees alone. **SnapBoost** is scikit-learn compatible and built on [HNBM](https://github.com/qiancapital/hnbm).

At each boosting round, SnapBoost stochastically selects either a decision tree or an RFF ridge regressor. That mix captures both local, axis-aligned structure and smooth global patterns.

## New in 1.2.0

- Native multiclass classification via HNBM softmax Newton boosting: one
  scalar learner per class each round, `predict_proba` of shape
  `(n_samples, n_classes)`, and `n_classes_` on the fitted estimator.
- Binary logistic classification is unchanged (scalar `decision_function`).
- Requires HNBM 1.2.0 or newer.

## New in 1.1.0

- Staged prediction (`staged_predict`, `staged_predict_proba`,
  `staged_decision_function`) and `permutation_importance`.
- Original-label `eval_metric` and `eval_sample_weight` for validation loss
  and early stopping.
- `gamma="scale"` (sklearn variance-based kernel coefficient) and
  `alpha_linear` for the optional linear family.
- Requires HNBM 1.1 or newer.

## New in 1.0.0

- Frozen `SnapBoostClassifier` / `SnapBoostRegressor` public API.
- sklearn estimator tags and `check_estimator` coverage via HNBM 1.0.
- Parameter validation at `fit`, matching the sklearn contract.
- `SnapBoost(mode=...)` and `SnapBoost_KernelRidge` are deprecated.

See [limitations](limitations.md) for the dense-input and CART-tree contract.
Classification supports binary logistic loss and multiclass softmax.

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
