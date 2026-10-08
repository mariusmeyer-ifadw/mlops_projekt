"""
Musterlösung zu main_v1.py (Tag 14) – NUR für den Kursleiter.
"""
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
import mlflow
from mlflow import MlflowClient
import numpy as np

app = FastAPI(title="Iris-Klassifikator-API")

MODEL_NAME = "iris-classifier"


def load_champion():
    """Lädt das Modell, auf das @champion gerade zeigt, und gibt
    (Modell, Versionsnummer) zurück. Die Versionsnummer wird einmal
    aufgelöst und dann direkt geladen, damit beides garantiert zusammenpasst."""
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}"), version


model, model_version = load_champion()

v1_router = APIRouter(prefix="/v1")


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictionResponse(BaseModel):
    prediction: int
    model_version: str


@v1_router.get("/health")
def health():
    return {"status": "ok"}


@v1_router.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])

    try:
        prediction = model.predict(data)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Modell nicht verfügbar: {e}")

    return PredictionResponse(prediction=int(prediction[0]), model_version=str(model_version))


@app.post("/admin/reload")
def reload_model():
    global model, model_version
    try:
        # Erst laden, dann beide Variablen auf einmal tauschen: Scheitert das
        # Laden, bleibt das alte Modell aktiv.
        model, model_version = load_champion()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Neu laden fehlgeschlagen: {e}")
    return {"status": "reloaded", "model_name": MODEL_NAME, "model_version": str(model_version)}


app.include_router(v1_router)
