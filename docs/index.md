# SnapBoost

```{raw} html
<div class="qc-hero">
  <div class="qc-hero-kicker">Qian Capital · Open Source</div>
  <div class="qc-hero-title" aria-hidden="true">SnapBoost</div>
  <p class="qc-hero-lead">A heterogeneous Newton boosting machine that mixes decision trees and random Fourier feature ridge regressors — scikit-learn compatible, built on HNBM.</p>
  <div class="qc-meta">
    <span class="qc-chip">v0.1.6</span>
    <span class="qc-chip">Python ≥ 3.8</span>
    <span class="qc-chip">scikit-learn</span>
    <span class="qc-chip">MIT</span>
  </div>
</div>
```

**SnapBoost** is a concrete [HNBM](https://github.com/qiancapital-dev/hnbm) implementation: at each boosting round it stochastically selects either a decision tree or an RFF ridge regressor. That mix captures both local, axis-aligned structure and smooth global patterns.

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
:caption: Documentation

installation
quickstart
api
parameters
examples
references
```

```{toctree}
:maxdepth: 1
:caption: Project

license
```
