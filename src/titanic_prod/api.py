"""FastAPI service: the trained bundle behind /health and /predict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import REPO_ROOT
from .data import age_bracket
from .exceptions import ServingError
from .features import NUMERIC_FEATURES, align_single_row

DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"


# Request shape: Pydantic rejects bad values with a 422 before scoring.
class PassengerIn(BaseModel):
    pclass: int = Field(ge=1, le=3)  # ticket class 1, 2, or 3
    sex: Literal["male", "female"]
    age: float = Field(ge=0, le=100)
    sibsp: int = Field(ge=0, le=10)  # siblings/spouses aboard
    parch: int = Field(ge=0, le=10)  # parents/children aboard
    fare: float = Field(ge=0)
    embarked: Literal["C", "Q", "S"]  # Cherbourg / Queenstown / Southampton


class PredictionOut(BaseModel):
    survived: int
    probability: float
    model_version: str = "titanic-v1"


def create_app(artifacts: str | Path | None = None) -> FastAPI:
    # Load the train-time bundle once at startup (tests pass a tmp dir here).
    bundle_dir = Path(artifacts) if artifacts else DEFAULT_ARTIFACTS
    try:
        model = joblib.load(bundle_dir / "model.pkl")
        scaler = joblib.load(bundle_dir / "scaler.pkl")
        columns = json.loads((bundle_dir / "features.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise ServingError(
            f"Cannot load model bundle from {bundle_dir} — run train first: {exc}"
        ) from exc

    app = FastAPI(title="Titanic Survival API", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "features": len(columns)}

    @app.post("/predict", response_model=PredictionOut)
    def predict(pax: PassengerIn) -> PredictionOut:
        # Rebuild the same engineered features training used (see data.engineer).
        family_size = pax.sibsp + pax.parch + 1
        row = {
            "Pclass": pax.pclass,
            "Sex": pax.sex,
            "Embarked": pax.embarked,
            "AgeBracket": age_bracket(pax.age),
            "Fare": pax.fare,
            "family_size": family_size,
            "IsAlone": int(family_size == 1),
        }
        try:
            # Align to training columns, scale numerics, take P(survived).
            X = align_single_row(row)[columns]
            X[NUMERIC_FEATURES] = scaler.transform(X[NUMERIC_FEATURES])
            proba = float(model.predict_proba(X)[0][1])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}") from exc
        # Threshold 0.5: >= 0.5 means predicted survivor.
        return PredictionOut(survived=int(proba >= 0.5), probability=round(proba, 4))

    return app


def _bootstrap() -> FastAPI:
    """Import-time app for uvicorn. Serves 503s until a bundle is trained."""
    # Lets `uvicorn titanic_prod.api:app` import even before first training.
    try:
        return create_app()
    except ServingError as exc:
        detail = str(exc)
        fallback = FastAPI(title="Titanic Survival API (untrained)", version="1.0.0")

        @fallback.get("/health")
        def health() -> dict:
            return {"status": "untrained", "detail": detail}

        return fallback


app = _bootstrap()
