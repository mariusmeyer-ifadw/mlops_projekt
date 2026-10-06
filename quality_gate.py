"""
Musterlösung zu quality_gate.py (Tag 13) – NUR für den Kursleiter.
"""
import sys
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
THRESHOLD = 0.9


def main():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("iris-klassifikator")

    runs = client.search_runs(
        experiment.experiment_id, order_by=["start_time DESC"], max_results=1
    )
    latest_run = runs[0]

    accuracy = latest_run.data.metrics["accuracy"]

    print(f"Neueste Accuracy: {accuracy:.3f} (Schwellenwert: {THRESHOLD})")

    if accuracy < THRESHOLD:
        print("Quality Gate NICHT bestanden.")
        sys.exit(1)

    print("Quality Gate bestanden.")


if __name__ == "__main__":
    main()
