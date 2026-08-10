FROM python:3.8-slim

WORKDIR /app

COPY pyproject.toml setup.py setup.cfg MANIFEST.in README.md LICENSE ./
COPY snapboost/ snapboost/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

CMD ["python", "-c", "from snapboost import SnapBoostClassifier, SnapBoostRegressor, HNBM; print('SnapBoost ready')"]
