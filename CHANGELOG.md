# Changelog

## 1.2.0

Release date: 2026-09-01

- Inherit native multiclass classification from HNBM 1.2: softmax Newton
  boosting with one scalar learner per class each round.
- Binary logistic classification is unchanged (scalar `decision_function`).
- Document multiclass labels, probability shapes, and the `n_classes_`
  fitted attribute.
- Document the exact kernel ridge estimators and `LaplacianSampler` in the API
  reference, and record `objective` / `objective_parameter` in the README
  parameter table.
- Export `WeightedKernelRidgeRegressor`, matching the already-public
  `RandomFourierRidgeRegressor` and `WeightedLinearRegressor`.

### Fixed

- Correct the documented learner-pool sampling rule. Tree candidates share
  `p_tree` evenly, kernel candidates share `1 - p_tree - p_linear` evenly, and
  the linear learner takes `p_linear`; the previous "`p_tree` versus
  `1 - p_tree`" description ignored per-depth splitting and the extra kernel and
  linear families.
- Inherit the HNBM 1.2 fix that raises `ValueError` for single-class targets.
- Raise `ValueError` when `monotonic_cst` is combined with a multiclass target.
  Constraints bind each class score separately, and every score can rise with a
  feature while no class probability does, so the fit used to return a model
  whose monotonicity guarantee held for nothing observable. scikit-learn
  rejects the same combination. Binary and regression fits are unaffected.
- Inherit the HNBM 1.2 fixes that keep `history_` aligned with `ensemble_`
  after early stopping and that report the early-stopping round to callbacks.
- Correct the softmax Hessian attribution in MATH.md and note that multiclass
  takes a longer Newton step than XGBoost at the same `learning_rate`.
- Document that the pseudo-Huber `delta` must match the residual scale. The
  Newton working response grows like `residual³ / delta²`, so the default
  `delta=1.0` diverges on targets that are not roughly unit-scale.

### Compatibility

- Requires HNBM 1.2.0 or newer.
- Multiclass `ensemble_` entries are length-`n_classes` lists of scalar
  learners. Binary ensembles remain one learner per round.
- Single-class classification targets now raise `ValueError`.

### Packaging and tooling

- Add PyPI classifiers.
- Ship `LICENSE`, `MATH.md`, and `CONTRIBUTING.md` in the sdist.

## 1.1.0

Release date: 2026-09-01

- Require HNBM 1.1 and inherit staged prediction, permutation importance,
  original-label `eval_metric`, and `eval_sample_weight`.
- Accept `gamma="scale"` (and `"scale"` entries in `kernel_gammas`).
- Add `alpha_linear` so the optional linear family can use a different ridge
  penalty than the RFF learners.
- Run sklearn `check_estimator` on the RFF path (`p_tree=0`).
- Add type hints on public preprocessing and kernel-gamma helpers.

## 1.0.0

Release date: 2026-08-30

- Freeze `SnapBoostClassifier` and `SnapBoostRegressor` as the 1.0 public API.
- Require HNBM 1.0 or newer and inherit its sklearn estimator contract.
- Delay SnapBoost-specific parameter validation until `fit`.
- Deprecate `SnapBoost(mode=...)` and `SnapBoost_KernelRidge`.
- Freeze the exact-kernel estimators as a specialized surface without adaptive
  training controls.
- Add sklearn `check_estimator` coverage for the recommended estimators.
- Reject all-zero `sample_weight` with a sklearn-compatible error message.
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

