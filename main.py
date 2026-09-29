"""
Musterlösung zu main_mit_registry.py (Tag 8) – NUR für den Kursleiter.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import numpy as np

app = FastAPI(title="Iris-Klassifikator-API (Registry-Version)")

model = mlflow.pyfunc.load_model("models:/iris-classifier@champion")
print(f"Geladenes Modell: run_id={model.metadata.run_id}")


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/health")
def health():
    return {"status": "ok", "model_run_id": model.metadata.run_id}


@app.post("/predict")
def predict(features: IrisFeatures):
    data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
