"""
Konsolidierte Musterlösung: main.py nach Tag 15 – NUR für den Kursleiter.

Baut auf main_v1_loesung.py (Tag 14) auf und ergänzt:
- Logging-Middleware
- API-Key-Absicherung (nur für /admin/reload, s. Kursweite Entscheidung Tag 15)
- Background Task für Prediction-Logging
- globaler Exception-Handler für ValueError (strukturierte Fehlerantwort)

Das ist die Referenz dafür, wie main.py am ENDE von Tag 15 aussehen sollte -
zum Abgleich, falls TN in der Übung nicht weiterkommen.
"""
import time
from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import mlflow
from mlflow import MlflowClient
import numpy as np

app = FastAPI(title="Iris-Klassifikator-API")

MODEL_NAME = "iris-classifier"


def load_champion():
    """Lädt das Modell, auf das @champion gerade zeigt, und gibt
    (Modell, Versionsnummer) zurück (siehe Tag 14)."""
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}"), version


model, model_version = load_champion()

v1_router = APIRouter(prefix="/v1")

# Nur zu Übungszwecken im Code: In einem echten Projekt gehört der Schlüssel
# in eine Umgebungsvariable bzw. ein Secret und nie ins Git-Repository.
API_KEY = "mein-geheimer-schluessel"
api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Ungültiger API-Key")


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    print(f"{request.method} {request.url.path} - {duration:.3f}s")
    return response


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": f"Ungültige Eingabe: {exc}"})


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictionResponse(BaseModel):
    prediction: int
    model_version: str


def log_prediction(input_data: dict, prediction: int):
    # Läuft NACH dem Zurückschicken der Antwort - beeinflusst die
    # Response-Zeit nicht.
    print(f"Prediction geloggt: input={input_data}, prediction={prediction}")


@v1_router.get("/health")
def health():
    return {"status": "ok"}


@v1_router.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures, background_tasks: BackgroundTasks):
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

    result = int(prediction[0])
    background_tasks.add_task(log_prediction, features.model_dump(), result)

    return PredictionResponse(prediction=result, model_version=str(model_version))


# Nur dieser administrative Endpoint ist API-Key-geschützt - /v1/predict
# und /v1/health bleiben bewusst offen (siehe Kursweite Entscheidung Tag 15,
# damit test_main.py ohne Anpassung funktioniert).
@app.post("/admin/reload", dependencies=[Depends(verify_api_key)])
def reload_model():
    global model, model_version
    try:
        # Erst laden, dann beide Variablen auf einmal tauschen (siehe Tag 14).
        model, model_version = load_champion()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Neu laden fehlgeschlagen: {e}")
    return {"status": "reloaded", "model_name": MODEL_NAME, "model_version": str(model_version)}


app.include_router(v1_router)
