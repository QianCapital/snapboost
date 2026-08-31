# Limitations

SnapBoost 1.0 is a scikit-learn realization of the NeurIPS 2020 HNBM idea, not
a drop-in replacement for IBM Snap ML or XGBoost. The following constraints are
part of the 1.0 contract.

## Binary classification only

`SnapBoostClassifier` supports exactly two classes and logistic loss. Multiclass
targets raise `ValueError` matching `"Only binary classification is supported."`

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
or early stopping. They have quadratic memory cost. Prefer the default RFF path
unless the dataset is small and an exact RBF kernel is required.

## Sample-weight equivalence

Fitting with integer `sample_weight` is not equivalent to repeating rows.
Hessian-weighted Newton updates and ridge regularization both depend on the
weight scale. sklearn's `check_sample_weight_equivalence_on_dense_data` is an
expected failure of this estimator family.
