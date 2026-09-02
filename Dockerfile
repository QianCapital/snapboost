FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml setup.py setup.cfg MANIFEST.in README.md LICENSE ./
COPY CHANGELOG.md CONTRIBUTING.md MATH.md REFERENCES.md CITATION.bib requirements.txt ./
COPY snapboost/ snapboost/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

CMD ["python", "-c", "from snapboost import SnapBoostClassifier, SnapBoostRegressor, HNBM; print('SnapBoost ready')"]
