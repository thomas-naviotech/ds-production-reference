# DS Production Reference — from notebook to production API

Course reference repo for the NavioTech Data Science batch. It takes the
**Titanic pipeline from Sessions 5–6** (inspect → clean → engineer → train)
and shows where every notebook cell lives once the work becomes a real
service: config-driven code, tracked experiments, and a FastAPI endpoint.

## The one idea of this repo

| Notebook habit | Production equivalent (here) |
|---|---|
| Cell scattered constants | `configs/*.yaml` + `params.yaml` → loaded by `src/titanic_prod/config.py` |
| Re-typed cleaning cells | `src/titanic_prod/data.py` (`clean`, `engineer`) |
| `pd.get_dummies` inline | `src/titanic_prod/features.py` (`encode`, fixed categories — train and serve agree) |
| Best accuracy in a cell output | MLflow runs under `mlruns/` + served bundle in `artifacts/` |
| "It worked on my machine" | `pyproject.toml` + `uv.lock` (uv), `Dockerfile`, CI in `.github/workflows/` |
| Ask the teacher | `AGENTS.md` — ask **any coding assistant**; it knows this repo |

## Quickstart

```bash
uv sync                  # create env + install (editable)
uv run pytest            # tests (features, config, API)
uv run python scripts/train.py --config configs/logreg.yaml
uv run python scripts/evaluate.py
uv run python scripts/serve.py        # -> http://127.0.0.1:8000/docs
```

Predict:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d ^
  "{\"pclass\": 1, \"sex\": \"female\", \"age\": 29, \"sibsp\": 0, \"parch\": 0, \"fare\": 80, \"embarked\": \"C\"}"
```

## CLI cheat-sheet (run in this order — every step prints rich tables)

```bash
uv run pytest                                        # tests (13 passed)
uv run python scripts/train.py --config configs/logreg.yaml   # one model
uv run python scripts/train.py --config configs/rf.yaml       # challenger
uv run python scripts/evaluate.py                    # score the serving bundle
uv run python scripts/compare.py                     # race ALL models, F1 leaderboard
uv run python scripts/compare.py --models svm rf gb  # ...or a subset
uv run python scripts/tune.py --model winner         # 2-stage tune of the F1 winner
uv run python scripts/tune.py --model svm --n-iter 15  # ...or tune any model
uv run python scripts/serve.py                       # -> http://127.0.0.1:8000/docs
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
                                                     # -> http://127.0.0.1:5000
```

`configs/` has one YAML per model (`logreg`, `rf`, `svm`, `knn`, `nb`, `gb`,
`xgb`) — swap the FILE, never the code. `compare.py` + `tune.py` write to
`artifacts/compare/<model>/` and `artifacts/tune/<model>/`, so the serving
bundle in `artifacts/` is never overwritten. `xgb` needs one extra install:
`uv pip install --python .venv\Scripts\python.exe xgboost` (Windows) or
`uv pip install --python .venv/bin/python xgboost` (Mac/Linux).


## DVC (optional)

`dvc.yaml` + `params.yaml` describe the pipeline (`prepare → train → evaluate`).
Install DVC (`uv pip install dvc`) and run `dvc repro` to execute it as stages.
Without DVC, the `scripts/` do the same steps directly.

## Structure

```
configs/        pluggable run configs (swap model without touching code)
src/titanic_prod/
  config.py     typed YAML loader (single place for settings)
  exceptions.py custom hierarchy — no bare excepts
  logging_setup.py  one logger, everywhere
  data.py       load + validate + clean + engineer (S05 logic, function form)
  features.py   deterministic encoding (train/serve contract)
  train.py      split-first training, scaler fit-on-train, MLflow logging
  evaluate.py   accuracy + report + confusion matrix from artifacts
  metrics.py    shared metric formulas (2-decimal) — train/evaluate/tune agree
  display.py    rich terminal tables + F1 bar charts for every script
  api.py        FastAPI app: /health, /predict
scripts/        thin CLI wrappers (prepare/train/evaluate/compare/tune/serve.py)
tests/          features, config, API (trains a throwaway model in tmp)
data/raw/       Titanic-Dataset.csv (same file as class)
notebooks/      drop the S05–S06 class notebooks here for side-by-side reading
```

## Full production mock — roadmap note

This repo is deliberately a *local* production shape. To mock a full
production system, add one layer at a time, in this order:

| Capability | Here now | Add next |
|---|---|---|
| CI/CD pipelines | GitHub Actions (`ci.yml`: lint + tests + train smoke) | Matrix builds, Docker build job; alternative: CircleCI |
| Experiment tracking | MLflow (local sqlite) | Remote tracking server + model registry stages |
| Data/pipeline versioning | `params.yaml` + `dvc.yaml` skeleton | `dvc repro` with remote storage (S3/GCS/Azure Blob) |
| Workflow orchestration | `scripts/` run manually, in order | Airflow DAGs; heavier flows: Kubeflow Pipelines |
| Model serving | FastAPI + `uvicorn` locally | BentoML or AWS SageMaker endpoints |
| Deployment targets | `Dockerfile` (local demo image) | AWS / Azure / GCP + Kubernetes manifests, one env at a time |
| Monitoring | `metrics.json` per run | Evidently AI (drift reports) + Grafana dashboards on serving metrics |

Rule: promote a layer only when the class can explain *why the current
layer hurts* — e.g. add Airflow when manual script order actually breaks,
add Grafana when nobody notices a stale model. Each row is one standalone
exercise; none of them changes `src/` contracts.

## Next step

Open `AGENTS.md` — it is the guided tour. Paste it into any coding assistant
and start asking questions.
