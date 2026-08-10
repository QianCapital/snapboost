# SnapBoost

[![PyPI version](https://img.shields.io/pypi/v/snapboost.svg)](https://pypi.org/project/snapboost/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-compatible-blue.svg)](https://scikit-learn.org/)

**Heterogeneous Newton Boosting Machine (HNBM)** — a gradient boosting framework that mixes decision trees and kernel ridge regressors instead of trees alone. The core [HNBM](https://github.com/qiancapital-dev/hnbm) framework is provided by the `hnbm` package; SnapBoost is a concrete implementation built on top of it.

Unlike XGBoost and LightGBM, which rely exclusively on decision trees as base learners, SnapBoost stochastically selects from a heterogeneous pool of learners at each boosting iteration. This lets the model capture both local, axis-aligned structure (trees) and smooth, global patterns (RBF kernel ridge).

This package is a Python/scikit-learn reimplementation inspired by [SnapBoost: A Heterogeneous Boosting Machine](https://arxiv.org/abs/2006.09745) (Parnell et al., NeurIPS 2020). See [REFERENCES.md](REFERENCES.md) for papers, related work, and citation details.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Examples & Results](#examples--results)
- [API Reference](#api-reference)
  - [SnapBoostClassifier / SnapBoostRegressor](#snapboostclassifier--snapboostregressor)
  - [SnapBoost](#snapboost)
  - [HNBM](#hnbm)
- [Parameters](#parameters)
- [Docker](#docker)
- [Development](#development)
- [References & Citation](#references--citation)
- [License](#license)

---

## Features

| Tag | Description |
|-----|-------------|
| `gradient-boosting` | Second-order Newton boosting with gradient and Hessian weighting |
| `heterogeneous-learners` | Mixes decision trees and kernel ridge regressors in one ensemble |
| `classification` | Binary classification with logistic loss |
| `regression` | Continuous targets with mean squared error loss |
| `scikit-learn` | Implements the scikit-learn estimator API (`fit`, `predict`, `score`, …) |
| `randomized-ensemble` | Stochastic base-learner selection per iteration |

---

## Installation

**From PyPI** (recommended):

```bash
pip install snapboost
```

**From source**:

```bash
git clone https://github.com/qiancapital/snapboost.git
cd snapboost
pip install .
```

**Requirements**: Python ≥ 3.8, NumPy, scikit-learn, tqdm, [`hnbm`](https://pypi.org/project/hnbm/) ≥ 0.2.1.

---

## Quick Start

### Classification

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

### Regression

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

---

## Examples & Results

Interactive Jupyter notebooks in [`static/`](static/) walk through classification, regression, and hyperparameter exploration. Each notebook trains SnapBoost and compares it against **XGBoost** and **LightGBM** on the same splits.

| Notebook | Dataset | SnapBoost | XGBoost | LightGBM |
|----------|---------|-----------|---------|----------|
| [Classification.ipynb](static/Classification.ipynb) | [Breast Cancer Wisconsin](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) | 97.2% accuracy | 95.8% | 96.5% |
| [Regression.ipynb](static/Regression.ipynb) | [Diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) | R² 0.44, RMSE 55.7 | R² 0.38, RMSE 58.4 | R² 0.40, RMSE 57.7 |
| [Parameter_Exploration.ipynb](static/Parameter_Exploration.ipynb) | Synthetic (piecewise + smooth) | R² 0.986, RMSE 0.170 | R² 0.986, RMSE 0.174 | R² 0.987, RMSE 0.167 |

Run the notebooks locally:

```bash
pip install ".[examples]"
jupyter notebook static/
```

### Classification

On the Breast Cancer dataset (250 boosting rounds), SnapBoost achieves the highest test accuracy and fewest misclassifications among the three boosters:

![Test accuracy and error count vs XGBoost and LightGBM](static/classification_comparison.png)

Confusion matrix for SnapBoost on the held-out test set:

![SnapBoost classification confusion matrix](static/classification_confusion_matrix.png)

### Regression

On the Diabetes dataset (100 boosting rounds), SnapBoost improves R² and RMSE over tree-only baselines:

![R², RMSE, and MAE comparison on Diabetes dataset](static/regression_comparison.png)

Predicted vs. actual disease progression on the test set:

![Predicted vs actual scatter plot](static/regression_predicted_vs_actual.png)

SnapBoost fitted curve along BMI (other features held at training medians):

![BMI vs target with SnapBoost fit](static/regression_bmi_fit.png)

Residual distribution:

![Regression residual histogram](static/regression_residuals.png)

### Parameter exploration

On a synthetic dataset mixing piecewise-linear and sinusoidal structure, the notebook sweeps `p_tree`, tree depth ranges, and kernel ridge parameters. A mixed ensemble (`p_tree=0.8`) outperforms trees-only (`p_tree=1.0`, RMSE 0.174) and ridge-only (`p_tree=0.0`, RMSE 0.366):

![Learned functions along one axis for different p_tree values](static/parameter_exploration_predictions.png)

See [Parameter_Exploration.ipynb](static/Parameter_Exploration.ipynb) for the full sweeps and baseline comparison tables.

---

## API Reference

### SnapBoostClassifier / SnapBoostRegressor

The recommended entry points (similar to `XGBClassifier` / `XGBRegressor`). A concrete HNBM that builds an ensemble from:

- **Decision trees** with depths sampled uniformly from `[min_max_depth, max_max_depth]`
- **One RFF ridge regressor** for smooth global fits

At each iteration, a learner is chosen with probability `p_tree` for trees (split evenly across depths) and `1 - p_tree` for the ridge model.

```python
from snapboost import SnapBoostClassifier, SnapBoostRegressor

clf = SnapBoostClassifier(
    num_iterations=100,
    learning_rate=0.1,
    p_tree=0.8,
    min_max_depth=4,
    max_max_depth=8,
    alpha=1.0,
    gamma=1.0,
    random_state=42,
    verbose=True,
)
clf.fit(X, y)

reg = SnapBoostRegressor(num_iterations=100, random_state=42)
reg.fit(X, y)
```

**Methods**

| Method | Classifier | Regressor | Description |
|--------|------------|-----------|-------------|
| `fit(X, y)` | ✓ | ✓ | Train the ensemble |
| `predict(X)` | ✓ | ✓ | Class labels (0/1) or continuous values |
| `predict_proba(X)` | ✓ | | Class probabilities, shape `(n_samples, 2)` |
| `decision_function(X)` | ✓ | | Raw logits |
| `score(X, y)` | ✓ | ✓ | Accuracy or R² |
| `evaluate(X, y)` | ✓ | ✓ | Prints and returns log loss or RMSE |

### SnapBoost

Legacy class that accepts a `mode` parameter (`"classification"` or `"regression"`). Prefer `SnapBoostClassifier` or `SnapBoostRegressor` for new code.

```python
from snapboost import SnapBoost

model = SnapBoost(
    num_iterations=100,
    learning_rate=0.1,
    p_tree=0.8,
    min_max_depth=4,
    max_max_depth=8,
    alpha=1.0,
    gamma=1.0,
    mode="classification",  # or "regression"
    random_state=42,
    verbose=True,
)
model.fit(X, y)
```

### HNBM

The abstract base class for building custom heterogeneous ensembles. Provided by the [`hnbm`](https://pypi.org/project/hnbm/) package — subclass or configure `base_learners_` and `probabilities_` before calling `fit`:

```python
from sklearn.tree import DecisionTreeRegressor
from hnbm import HNBMClassifier, HNBMRegressor

class MyClassifier(HNBMClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_learners_ = [DecisionTreeRegressor(max_depth=5)]
        self.probabilities_ = [1.0]
```

---

## Parameters

### Shared (`HNBM` / `SnapBoostClassifier` / `SnapBoostRegressor`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_iterations` | `int` | `100` | Number of boosting rounds |
| `learning_rate` | `float` | `0.1` | Shrinkage applied to each learner's contribution |
| `random_state` | `int` or `None` | `None` | Seed for learner selection and tree fitting |
| `verbose` | `bool` | `False` | Show a tqdm progress bar during training |

The legacy `SnapBoost` class also accepts a `mode` parameter (`"classification"` or `"regression"`).

### SnapBoost-specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `p_tree` | `float` | `0.9` | Probability of selecting a decision tree (vs. ridge) |
| `min_max_depth` | `int` | `2` | Minimum `max_depth` for trees in the pool |
| `max_max_depth` | `int` | `4` | Maximum `max_depth` for trees in the pool |
| `alpha` | `float` | `1.0` | L2 regularization for the RFF ridge regressor |
| `gamma` | `float` | `1.0` | RBF kernel coefficient for random Fourier features |
| `n_components` | `int` | `100` | Number of random Fourier features |

**Label conventions (classification)**: accepts `0`/`1` or `-1`/`+1`. Predictions are returned as `0`/`1`.

---

## Docker

Build and run a container with SnapBoost pre-installed:

```bash
docker build -t snapboost .
docker run --rm snapboost
```

The default command verifies the import:

```
SnapBoost ready
```

---

## Development

```bash
git clone https://github.com/qiancapital/snapboost.git
cd snapboost
pip install -e ".[examples,test]"
pytest
jupyter notebook static/   # optional: run example notebooks
```

Releases are published to PyPI via GitHub Actions when a GitHub release is created.

---

## References & Citation

If you use this package or the HNBM framework in research, please cite the original SnapBoost paper:

> Thomas Parnell, Andreea Anghel, Małgorzata Łazuka, Nikolas Ioannou, Sebastian Kurella, Peshal Agarwal, Nikolaos Papandreou, and Haralampos Pozidis. **SnapBoost: A Heterogeneous Boosting Machine.** *Advances in Neural Information Processing Systems*, 33, 2020.

```bibtex
@inproceedings{parnell2020snapboost,
  title     = {{SnapBoost}: A Heterogeneous Boosting Machine},
  author    = {Parnell, Thomas and Anghel, Andreea and {\L}azuka, Ma{\l}gorzata and Ioannou, Nikolas and Kurella, Sebastian and Agarwal, Peshal and Papandreou, Nikolaos and Pozidis, Haralampos},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  pages     = {20872--20883},
  year      = {2020},
  eprint    = {2006.09745},
  doi       = {10.48550/arXiv.2006.09745}
}
```

**Links:** [arXiv:2006.09745](https://arxiv.org/abs/2006.09745) · [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2020/hash/7fd3b80fb1884e2927df46a7139bb8bf-Abstract.html) · [IBM Research](https://research.ibm.com/publications/snapboost-a-heterogeneous-boosting-machine)

For the full bibliography, related heterogeneous-boosting literature (KTBoost, DeepBoost, etc.), and notes on how this repo relates to the original IBM Snap ML implementation, see **[REFERENCES.md](REFERENCES.md)**. Additional BibTeX entries are in **[CITATION.bib](CITATION.bib)**.

---

## License

MIT — See [LICENSE](LICENSE) for full text.
