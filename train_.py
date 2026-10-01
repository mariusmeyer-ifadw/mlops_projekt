"""
Musterlösung zu train_mit_registry.py (Tag 8) – NUR für den Kursleiter.
"""
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris-klassifikator")

N_ESTIMATORS = 2


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

        mlflow.sklearn.log_model(model, "model", registered_model_name="iris-classifier",
                                 skops_trusted_types=["sklearn.tree._tree.Tree"],)

        print("Modell trainiert, getrackt und registriert.")


if __name__ == "__main__":
    main()
