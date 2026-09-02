# Contributing to SnapBoost

## Development

```bash
python -m pip install -e "../hnbm"
python -m pip install -e ".[dev]"
python -m ruff check .
python -m mypy
python -m pytest --cov --cov-report=term-missing
```

Publish **HNBM 1.x before SnapBoost 1.x**. SnapBoost 1.2 requires `hnbm>=1.2.0`.

## Semantic versioning

SnapBoost 1.0 freezes `SnapBoostClassifier` and `SnapBoostRegressor`:

- Defaults of the random HNBM algorithm stay (`selection_strategy="random"`,
  `line_search=False`, `subsample=1.0`, trees + RFF ridge).
- New constructor arguments are additive and default to current behavior.
- Removing or renaming a public parameter requires a deprecation cycle and a
  major version.
- `SnapBoost(mode=...)` and `SnapBoost_KernelRidge` are deprecated and will be
  removed in 2.0.
- Exact-kernel task-specific classes are a frozen specialized surface.

See [limitations](docs/limitations.md) for behavior that will not change in 1.x
without a major version.

## Release checklist

1. HNBM 1.x is on PyPI.
2. Tests, ruff, mypy, coverage floor, and `check_estimator` pass.
3. Changelog section is dated (not "staged").
4. `__version__` matches the GitHub release tag (`v1.x.y`).
5. Add `vX.Y.Z` to `docs/versions.json` and push `master`. GitHub Pages
   deploys only from `master`/`main` (environment protection blocks tags).
   The docs workflow rebuilds `/vX.Y.Z/` from that list.
