FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt setup.py setup.cfg MANIFEST.in README.md ./
COPY snapboost/ snapboost/

RUN pip install --no-cache-dir .

CMD ["python", "-c", "from snapboost import SnapBoost; print('SnapBoost ready')"]
