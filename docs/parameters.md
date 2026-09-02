# Parameters

## Shared

The core parameters are accepted by all SnapBoost estimators. Adaptive controls
are exposed by the recommended `SnapBoostClassifier` and
`SnapBoostRegressor` classes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_iterations` | `int` | `100` | Number of boosting rounds |
| `learning_rate` | `float` | `0.1` | Shrinkage applied to each learner's contribution |
| `random_state` | `int` or `None` | `None` | Seed for learner selection and tree fitting |
| `verbose` | `bool` | `False` | Show a tqdm progress bar during training |
| `selection_strategy` | `{"random", "greedy"}` | `"random"` | Sample one learner or select the lowest-loss candidate |
| `line_search` | `bool` | `False` | Select a contribution weight per boosting round |
| `subsample` | `float` | `1.0` | Fraction of rows used to fit each learner |
| `early_stopping_rounds` | positive `int` or `None` | `None` | Validation patience before restoring the best ensemble |
| `min_delta` | `float` | `0.0` | Minimum validation-loss improvement |
| `objective` | `str` | `"auto"` | Loss to optimize. Classifiers accept `"auto"` and `"log_loss"`; regressors also accept `"squared_error"`, `"pseudo_huber"`, and `"quantile"` |
| `objective_parameter` | `float` or `None` | `None` | Pseudo-Huber delta (default `1.0`) or quantile level (default `0.5`); ignored otherwise |

Classifiers select logistic loss for binary targets and softmax for multiclass
targets under both `objective="auto"` and `objective="log_loss"`.

```{warning}
The Newton working response for pseudo-Huber grows like `residual³ / delta²`,
so the default `delta=1.0` diverges on targets that are not roughly unit-scale.
Standardize `y`, or set `objective_parameter` to about the residual scale, as
with `huber_slope` in XGBoost's `reg:pseudohubererror`.
```

The legacy `SnapBoost` class also accepts `mode` (`"classification"` or `"regression"`)
and is deprecated. The legacy `SnapBoost_KernelRidge` class is also
deprecated; it is the one exception to the table above and defaults to
`verbose=True`. Invalid constructor values are rejected at `fit`, not at
construction.

## SnapBoost-specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `p_tree` | `float` | `0.9` | Total probability of the tree families, split evenly across depths |
| `p_linear` | `float` | `0.0` | Probability allocated to the optional weighted linear family |
| `min_max_depth` | `int` | `2` | Minimum `max_depth` for trees in the pool |
| `max_max_depth` | `int` | `4` | Maximum `max_depth` for trees in the pool |
| `min_samples_leaf` | `int` | `10` | Minimum samples per leaf for decision trees |
| `alpha` | `float` | `1.0` | L2 regularization for the RFF ridge regressor |
| `alpha_linear` | `float` or `None` | `None` | L2 penalty for the optional linear family; defaults to `alpha` |
| `gamma` | `float` or `"scale"` | `1.0` | Kernel coefficient, or sklearn's variance-based `'scale'` |
| `n_components` | `int` | `100` | Number of random Fourier features |
| `scale_features` | `bool` | `True` | Standardize inputs used by the RFF learner |
| `max_features` | `None`, `int`, `float`, or `str` | `None` | Features considered by each tree split |
| `kernel_gammas` | sequence or `None` | `None` | Optional kernel bandwidth pool |
| `kernel_types` | sequence | `("rbf",)` | Any combination of `"rbf"` and `"laplacian"` |
| `monotonic_cst` | sequence or `None` | `None` | Per-feature monotonic tree constraints when supported; binary and regression only |

```{tip}
On problems with both piecewise and smooth structure, try `p_tree` around `0.8`–`0.9`. Setting `p_tree=1.0` disables the ridge learner; `p_tree=0.0` uses ridge only.
```

## Label conventions

For binary and multiclass classification, original labels are accepted.
Predictions use those labels and probability columns follow `classes_` order.
Binary `decision_function` is one-dimensional; multiclass returns one column
per class.

## Fit-time data

`fit(X, y, sample_weight=None, eval_set=None, *, eval_sample_weight=None)`
accepts non-negative observation weights, one validation pair, and optional
validation weights. `early_stopping_rounds` has an effect only when
`eval_set=(X_validation, y_validation)` is provided. `eval_metric(y, raw)`
receives the original labels, not the internal `{-1, +1}` encoding.

`eval_metric` and `callbacks` are optional fit-time arguments.
`candidate_n_jobs` controls threaded candidate fitting in greedy mode only.
After `fit`, `staged_predict` (and classifier `staged_predict_proba` /
`staged_decision_function`) yield the ensemble after each round.
`permutation_importance(X, y)` is the recommended feature-importance API for
mixed tree and kernel learners.
