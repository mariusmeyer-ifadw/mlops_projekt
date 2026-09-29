"""
Trainingsskript für das durchgängige Kursprojekt – mit MLflow-Tracking
gegen den containerisierten Server (Tag 7).
"""
import joblib
import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris-klassifikator")

N_ESTIMATORS = 100


def main():
    print("Lade Daten...")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        mlflow.log_param("n_estimators", N_ESTIMATORS)

        print("Trainiere Modell...")
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42)
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"Accuracy auf Testdaten: {accuracy:.3f}")

        mlflow.log_metric("accuracy", accuracy)

        joblib.dump(model, "model.pkl")
        mlflow.log_artifact("model.pkl")

        print("Modell gespeichert und in MLflow getrackt.")


if __name__ == "__main__":
    main()
