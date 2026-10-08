"""
Musterlösung zu test_main.py (Tag 15) – NUR für den Kursleiter.
"""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="module")
def client():
    # Zwei Dinge müssen gemockt werden, BEVOR main.py importiert wird:
    # - MlflowClient: löst den @champion-Alias zur Versionsnummer auf
    # - mlflow.pyfunc.load_model: lädt das eigentliche Modell
    with patch("mlflow.pyfunc.load_model") as mock_load_model, \
         patch("mlflow.MlflowClient") as mock_client:
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_load_model.return_value = mock_model
        mock_client.return_value.get_model_version_by_alias.return_value.version = "1"

        from main import app
        from fastapi.testclient import TestClient
        yield TestClient(app)


def test_health(client):
    response = client.get("/v1/health")
    assert response.status_code == 200


def test_predict_happy_path(client):
    response = client.post("/v1/predict", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    })
    assert response.status_code == 200
    assert response.json() == {"prediction": 1, "model_version": "1"}


def test_predict_invalid_input(client):
    response = client.post("/v1/predict", json={
        "sepal_length": "nicht-eine-zahl",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    })
    assert response.status_code == 422


def test_reload_requires_api_key(client):
    # Bonus (Übung "Wenn ihr schneller fertig seid")
    response = client.post("/admin/reload")
    assert response.status_code == 401


def test_reload_with_api_key(client):
    response = client.post("/admin/reload", headers={"X-API-Key": "mein-geheimer-schluessel"})
    assert response.status_code == 200
    assert response.json()["model_version"] == "1"
