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
| `verbose` | `bool` | `True` | Show a tqdm progress bar during training |
| `selection_strategy` | `{"random", "greedy"}` | `"random"` | Sample one learner or select the lowest-loss candidate |
| `line_search` | `bool` | `False` | Select a contribution weight per boosting round |
| `subsample` | `float` | `1.0` | Fraction of rows used to fit each learner |
| `early_stopping_rounds` | positive `int` or `None` | `None` | Validation patience before restoring the best ensemble |
| `min_delta` | `float` | `0.0` | Minimum validation-loss improvement |
| `objective` | `str` | `"auto"` | Regression objective: squared error, pseudo-Huber, or quantile |
| `objective_parameter` | `float` or `None` | `None` | Pseudo-Huber delta or quantile level |

The legacy `SnapBoost` class also accepts `mode` (`"classification"` or `"regression"`).

## SnapBoost-specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `p_tree` | `float` | `0.9` | Probability of selecting a decision tree (vs. RFF ridge) |
| `p_linear` | `float` | `0.0` | Probability allocated to the optional weighted linear family |
| `min_max_depth` | `int` | `2` | Minimum `max_depth` for trees in the pool |
| `max_max_depth` | `int` | `4` | Maximum `max_depth` for trees in the pool |
| `min_samples_leaf` | `int` | `10` | Minimum samples per leaf for decision trees |
| `alpha` | `float` | `1.0` | L2 regularization for the RFF ridge regressor |
| `gamma` | `float` | `1.0` | RBF kernel coefficient for random Fourier features |
| `n_components` | `int` | `100` | Number of random Fourier features |
| `scale_features` | `bool` | `True` | Standardize inputs used by the RFF learner |
| `max_features` | `None`, `int`, `float`, or `str` | `None` | Features considered by each tree split |
| `kernel_gammas` | sequence or `None` | `None` | Optional kernel bandwidth pool |
| `kernel_types` | sequence | `("rbf",)` | Any combination of `"rbf"` and `"laplacian"` |
| `monotonic_cst` | sequence or `None` | `None` | Per-feature monotonic tree constraints when supported |

```{tip}
On problems with both piecewise and smooth structure, try `p_tree` around `0.8`–`0.9`. Setting `p_tree=1.0` disables the ridge learner; `p_tree=0.0` uses ridge only.
```

## Label conventions

For binary classification, any two distinct labels are accepted. Predictions
use the original labels and probability columns follow `classes_` order.

## Fit-time data

`fit(X, y, sample_weight=None, eval_set=None)` accepts non-negative observation
weights and one validation pair. `early_stopping_rounds` has an effect only
when `eval_set=(X_validation, y_validation)` is provided.

`eval_metric` and `callbacks` are optional fit-time arguments.
`candidate_n_jobs` controls threaded candidate fitting in greedy mode only.
