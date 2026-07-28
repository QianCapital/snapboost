FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt setup.py setup.cfg MANIFEST.in README.md LICENSE ./
COPY snapboost/ snapboost/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

CMD ["python", "-c", "from snapboost import SnapBoost, HNBM; print('SnapBoost ready')"]
