# SnapBoost

[![PyPI version](https://img.shields.io/pypi/v/snapboost.svg)](https://pypi.org/project/snapboost/)
[![Python versions](https://img.shields.io/pypi/pyversions/snapboost.svg)](https://pypi.org/project/snapboost/)
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
- [API Reference](#api-reference)
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

**Requirements**: Python ≥ 3.6, NumPy, scikit-learn, tqdm (see `requirements.txt` for pinned versions).

---

## Quick Start

### Classification

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from snapboost import SnapBoost

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = SnapBoost(
    num_iterations=100,
    learning_rate=0.1,
    mode="classification",
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
from snapboost import SnapBoost

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = SnapBoost(
    num_iterations=100,
    learning_rate=0.1,
    mode="regression",
    random_state=42,
)
model.fit(X_train, y_train)

print("R²:", model.score(X_test, y_test))
model.evaluate(X_test, y_test)  # prints RMSE
```

---

## API Reference

### SnapBoost

The main entry point. A concrete HNBM that builds an ensemble from:

- **Decision trees** with depths sampled uniformly from `[min_max_depth, max_max_depth]`
- **One kernel ridge regressor** (RBF kernel) for smooth global fits

At each iteration, a learner is chosen with probability `p_tree` for trees (split evenly across depths) and `1 - p_tree` for the ridge model.

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

**Methods**

| Method | Mode | Description |
|--------|------|-------------|
| `fit(X, y)` | both | Train the ensemble |
| `predict(X)` | both | Class labels (0/1) or continuous values |
| `predict_proba(X)` | classification | Class probabilities, shape `(n_samples, 2)` |
| `decision_function(X)` | classification | Raw logits |
| `score(X, y)` | both | Accuracy (classification) or R² (regression) |
| `evaluate(X, y)` | both | Prints and returns log loss or RMSE |

### HNBM

The abstract base class for building custom heterogeneous ensembles. Provided by the [`hnbm`](https://pypi.org/project/hnbm/) package — subclass or configure `base_learners_` and `probabilities_` before calling `fit`:

```python
from sklearn.tree import DecisionTreeRegressor
from hnbm import HNBM

class MyBoost(HNBM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_learners_ = [DecisionTreeRegressor(max_depth=5)]
        self.probabilities_ = [1.0]
```

---

## Parameters

### Shared (`HNBM` / `SnapBoost`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_iterations` | `int` | `100` | Number of boosting rounds |
| `learning_rate` | `float` | `0.1` | Shrinkage applied to each learner's contribution |
| `mode` | `str` | `"classification"` | `"classification"` or `"regression"` |
| `random_state` | `int` or `None` | `None` | Seed for learner selection and tree fitting |
| `verbose` | `bool` | `True` | Show a tqdm progress bar during training |

### SnapBoost-specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `p_tree` | `float` | `0.8` | Probability of selecting a decision tree (vs. ridge) |
| `min_max_depth` | `int` | `4` | Minimum `max_depth` for trees in the pool |
| `max_max_depth` | `int` | `8` | Maximum `max_depth` for trees in the pool |
| `alpha` | `float` | `1.0` | L2 regularization for `KernelRidge` |
| `gamma` | `float` | `1.0` | RBF kernel coefficient for `KernelRidge` |

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
pip install -r requirements.txt
pip install -e .
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
