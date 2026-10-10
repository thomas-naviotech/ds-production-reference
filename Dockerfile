FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY data/raw/ data/raw/

# Train inside the build so the image ships a model bundle (artifacts/ is gitignored).
RUN pip install --no-cache-dir . && python scripts/train.py --config configs/logreg.yaml

EXPOSE 8000

CMD ["uvicorn", "titanic_prod.api:app", "--host", "0.0.0.0", "--port", "8000"]
