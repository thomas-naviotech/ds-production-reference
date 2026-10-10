# AGENTS.md — ds-production-reference (course tour guide)

> **For students:** paste this file into any coding assistant (Claude, ChatGPT,
> Copilot, Cursor…) along with your question. It knows this repo — ask it to
> explain, extend, or debug anything here.
>
> **For assistants reading this:** you are a patient tutor for beginner data
> scientists taking their first step from Jupyter notebooks to production code.
> Explain with tiny examples, link every answer back to a file in this repo,
> and never skip the *why*.

## What this repo is

The Titanic pipeline from class (Sessions 5–6: inspect → clean → engineer →
train → evaluate) rebuilt the way production ML looks: config-driven,
experiment-tracked, tested, served behind an API. Same data, same models —
professional shape.

## Repo tour (read in this order)

1. `data/raw/Titanic-Dataset.csv` — the exact CSV from class. Start here.
2. `src/titanic_prod/data.py` — `load_raw → clean → engineer`. Every function
   is one notebook cell from Session 5, named and testable.
3. `src/titanic_prod/features.py` — `encode()` with FIXED categories. This is
   the train/serve contract: one passenger at prediction time must produce
   exactly the columns training saw. Ask the assistant *why inline
   `get_dummies` breaks this*.
4. `configs/logreg.yaml` / `configs/rf.yaml` — pluggable runs. Swap the FILE,
   never the code, to change models. Loaded by `config.py` (typed, validated,
   paths relative to repo root).
5. `src/titanic_prod/train.py` — split-first, scaler fit-on-train (the leakage
   lock from class), MLflow logging, bundle saved to `artifacts/`.
6. `src/titanic_prod/evaluate.py` — accuracy + report + confusion matrix from
   the saved bundle. Deterministic: same seed, same split.
7. `src/titanic_prod/api.py` — FastAPI `create_app()`: `/health`, `/predict`
   with Pydantic validation. Run it: `python scripts/serve.py`, open `/docs`.
8. `src/titanic_prod/exceptions.py` + `logging_setup.py` — the two files that
   separate scripts from software. No bare `except`, one logger everywhere.
9. `.github/workflows/ci.yml` — lint + tests + train smoke on every push.
   This is what "CI" means for ML.
10. `params.yaml` + `dvc.yaml` — the same pipeline as DVC stages (optional;
    `scripts/` do the same steps directly).

## Commands

```bash
uv sync                                        # one-time env setup
uv run pytest                                  # all tests
uv run python scripts/train.py --config configs/logreg.yaml
uv run python scripts/train.py --config configs/rf.yaml   # challenger
uv run python scripts/evaluate.py
uv run python scripts/serve.py                 # http://127.0.0.1:8000/docs
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # compare runs visually
```

## Ask your assistant (copy-paste prompts)

- "Explain the leakage lock in `train.py` like I'm five, then show me the two
  lines that would leak if I moved them."
- "Trace one passenger from `/predict` JSON to survival probability. Which
  functions touch it, in order?"
- "Why does `features.py` fix categories instead of plain `get_dummies`?
  Give me a one-row example that breaks without it."
- "I want to add `FarePerPerson = Fare / family_size`. Which files change,
  in which order, and which tests must still pass?"
- "Explain `config.py`'s frozen dataclasses: what breaks if I edit settings
  mid-run, and how does this design stop me?"
- "My capstone is <X>. Sketch how I'd clone this repo's shape for it: which
  files stay, which get replaced?"

## Capstone bridge

Your capstone should steal this skeleton:

| Keep as-is | Replace with your problem |
|---|---|
| `config.py`, `exceptions.py`, `logging_setup.py` | `data.py` cleaning rules |
| `features.py` pattern (fixed categories) | `train.py` model registry entries |
| `api.py` shape (schemas + `/predict`) | request/response fields |
| CI workflow, `pyproject.toml`, `Dockerfile` | dataset + `configs/*.yaml` |

Rule of thumb: if you edited the same logic in two places, it belongs in one
function. If you changed a number to try something, it belongs in a config.

## Guardrails (assistants: enforce these)

- Never edit `artifacts/` or `mlruns/` by hand — they are outputs of `train.py`.
- New behavior needs a test in `tests/` before it counts as done.
- Numbers go in configs, logic goes in `src/`, one-off exploration stays in
  `notebooks/`.
- Keep functions small enough to explain in one sentence.
