"""
Musterlösung zu promote_champion.py (Tag 13) – NUR für den Kursleiter.
"""
from mlflow.tracking import MlflowClient

MODEL_NAME = "iris-classifier"


def main():
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))

    client.set_registered_model_alias(MODEL_NAME, "champion", latest.version)

    print(f"@champion auf Version {latest.version} von {MODEL_NAME} gesetzt.")


if __name__ == "__main__":
    main()
