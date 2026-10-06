"""
Musterlösung zu batch_inference.py (Tag 12) – NUR für den Kursleiter.
"""
import pandas as pd
import mlflow
from prefect import flow, task

mlflow.set_tracking_uri("http://localhost:5000")

FEATURE_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


@task
def load_data():
    return pd.read_csv("data/iris.csv")


@task
def load_champion_model():
    return mlflow.pyfunc.load_model("models:/iris-classifier@champion")


@task
def run_batch_predictions(df, model):
    predictions = model.predict(df[FEATURE_COLS])
    df["prediction"] = predictions
    return df


@flow
def batch_inference():
    df = load_data()
    model = load_champion_model()
    result = run_batch_predictions(df, model)
    print(result[["sepal_length", "petal_length", "prediction"]].head(10))
    print(f"Batch-Inference abgeschlossen: {len(result)} Vorhersagen.")


if __name__ == "__main__":
    batch_inference()
