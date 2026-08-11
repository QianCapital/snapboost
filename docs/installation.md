# Installation

## From PyPI

```bash
pip install snapboost
```

## From source

```bash
git clone https://github.com/qiancapital/snapboost.git
cd snapboost
pip install .
```

## Requirements

| Dependency | Constraint |
|------------|------------|
| Python | ≥ 3.9 |
| NumPy | ≥ 1.20 |
| scikit-learn | ≥ 1.0 |
| tqdm | ≥ 4.50 |
| [hnbm](https://pypi.org/project/hnbm/) | ≥ 0.1.1 |

## Docker

Build and run a container with SnapBoost pre-installed:

```bash
docker build -t snapboost .
docker run --rm snapboost
```

The default command verifies the import and prints `SnapBoost ready`.

## Development install

```bash
git clone https://github.com/qiancapital/snapboost.git
cd snapboost
pip install -r requirements.txt
pip install -e .

# Optional: example notebooks
jupyter notebook static/
```

To build these docs locally:

```bash
pip install -r docs/requirements.txt
cd docs && make html
# open _build/html/index.html
```
