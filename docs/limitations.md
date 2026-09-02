# Limitations

SnapBoost is a scikit-learn realization of the NeurIPS 2020 HNBM idea, not a
drop-in replacement for IBM Snap ML or XGBoost. The following constraints are
part of the 1.x contract.

## No multilabel or multioutput targets

`SnapBoostClassifier` supports binary logistic loss and multiclass softmax.
Binary models keep a scalar `decision_function`. Multiclass models return
shape `(n_samples, n_classes)` and store one scalar learner per class in
each boosting round. Multilabel and multioutput targets raise `ValueError`.

## No monotonic constraints under softmax

`monotonic_cst` is rejected with `ValueError` when `fit` receives more than two
classes. Constraints would bind each class score separately, and every score
can rise with a feature while no class probability does, so the constraint
would not mean what it appears to mean. scikit-learn rejects the same
combination. Binary classification and regression are unaffected.

Multiclass also takes a longer Newton step than XGBoost at the same
`learning_rate`, because the softmax Hessian keeps the undamped diagonal. Tune
`learning_rate` separately for multiclass instead of reusing binary settings,
or guard the run with `early_stopping_rounds`. See MATH.md 4.2.1.

## Dense numeric inputs

Training and prediction accept dense numeric arrays. Sparse matrices, native
NaNs, and native categorical splits are not supported inside the booster. Use
`make_tabular_preprocessor` (or any sklearn `Pipeline`) for missing values and
categoricals before `fit`.

## CART trees, not histogram BDTs

Tree learners are scikit-learn `DecisionTreeRegressor` objects. The original
SnapBoost paper used histogram-based binary decision trees in C++. Accuracy can
still match a heterogeneous ensemble; wall-clock performance will not match
Snap ML or XGBoost on large data.

## Exact kernel ridge is specialized

`SnapBoostKernelRidgeClassifier` and `SnapBoostKernelRidgeRegressor` keep a
frozen constructor surface without greedy selection, line search, subsampling,
or early stopping. They inherit binary and multiclass classification from HNBM
but have quadratic memory cost. Prefer the default RFF path unless the dataset
is small and an exact RBF kernel is required.

## At least two classes are required

Classifiers raise `ValueError` when `y` contains a single class. Fit a
`DummyClassifier` for degenerate targets.

## Sample-weight equivalence

Fitting with integer `sample_weight` is not equivalent to repeating rows.
Hessian-weighted Newton updates and ridge regularization both depend on the
weight scale. sklearn's `check_sample_weight_equivalence_on_dense_data` is an
expected failure of this estimator family.
