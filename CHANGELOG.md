# Changelog

## 1.0.0

Release date: 2026-08-30

- Freeze `SnapBoostClassifier` and `SnapBoostRegressor` as the 1.0 public API.
- Require HNBM 1.0 or newer and inherit its sklearn estimator contract.
- Delay SnapBoost-specific parameter validation until `fit`.
- Deprecate `SnapBoost(mode=...)` and `SnapBoost_KernelRidge`.
- Freeze the exact-kernel estimators as a specialized surface without adaptive
  training controls.
- Add sklearn `check_estimator` coverage for the recommended estimators.
- Document limitations (binary classification, dense inputs, CART trees,
  pipeline-only missing values and categoricals).
- Mark the package as typed (`py.typed`) and raise the coverage floor to 90%.

### Compatibility

- Requires HNBM 1.0.0 or newer.
- Invalid constructor values are stored and rejected at `fit` instead of at
  construction.
- `SnapBoost` and `SnapBoost_KernelRidge` emit `FutureWarning` and will be
  removed in 2.0.
- Default training remains random HNBM selection without line search,
  subsampling, or early stopping.

## 0.2.1

Release date: 2026-08-30

- Avoid constructing a kernel family when `p_tree + p_linear` already sums to
  one and rounding leaves a negligible residual probability.
- Add ruff, mypy, and coverage configuration with a `dev` extra.
- Build the Docker image on a supported Python version.
- Clarify that best-ensemble restoration happens only when
  `early_stopping_rounds` triggers.

### Compatibility

- Requires HNBM 0.3.1 or newer.
- Via HNBM, predicting on a dataframe whose columns are reordered or renamed
  relative to fit now raises `ValueError` instead of returning wrong values.

## 0.2.0

Release date: 2026-08-11

- Add adaptive greedy learner selection through HNBM 0.3.
- Add opt-in pseudo-Huber and quantile objectives.
- Add RBF/Laplacian multi-bandwidth random-feature pools.
- Add an optional weighted raw linear learner family.
- Add a pipeline-friendly missing/categorical preprocessing helper.
- Add optional monotonic tree constraints with version guarding.
- Add custom metrics, callbacks, parallel candidates, and model compaction.
- Add per-round line search, validation history, and early stopping.
- Add sample weights, row subsampling, and tree feature subsampling.
- Standardize RFF inputs by default.
- Generate a distinct reproducible RFF basis for each boosting round.
- Avoid constructing disabled tree families when `p_tree=0`.
- Harden tree `max_features` validation for invalid array-like inputs.
- Preserve the original stochastic SnapBoost algorithm as the default mode.

### Compatibility

- Requires HNBM 0.3.0 or newer.
- Existing default behavior remains random learner selection without line
  search, subsampling, or early stopping.
- Recommended task-specific estimators expose the adaptive controls; legacy
  and exact-kernel constructor signatures remain stable.

