# Parameters

## Shared

These parameters are accepted by `HNBM`, `SnapBoostClassifier`, `SnapBoostRegressor`, and the legacy `SnapBoost` class.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_iterations` | `int` | `100` | Number of boosting rounds |
| `learning_rate` | `float` | `0.1` | Shrinkage applied to each learner's contribution |
| `random_state` | `int` or `None` | `None` | Seed for learner selection and tree fitting |
| `verbose` | `bool` | `True` | Show a tqdm progress bar during training |

The legacy `SnapBoost` class also accepts `mode` (`"classification"` or `"regression"`).

## SnapBoost-specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `p_tree` | `float` | `0.9` | Probability of selecting a decision tree (vs. RFF ridge) |
| `min_max_depth` | `int` | `2` | Minimum `max_depth` for trees in the pool |
| `max_max_depth` | `int` | `4` | Maximum `max_depth` for trees in the pool |
| `min_samples_leaf` | `int` | `10` | Minimum samples per leaf for decision trees |
| `alpha` | `float` | `1.0` | L2 regularization for the RFF ridge regressor |
| `gamma` | `float` | `1.0` | RBF kernel coefficient for random Fourier features |
| `n_components` | `int` | `100` | Number of random Fourier features |

```{tip}
On problems with both piecewise and smooth structure, try `p_tree` around `0.8`–`0.9`. Setting `p_tree=1.0` disables the ridge learner; `p_tree=0.0` uses ridge only.
```

## Label conventions

For classification, labels may be `0`/`1` or `-1`/`+1`. Predictions are returned as `0`/`1`.
