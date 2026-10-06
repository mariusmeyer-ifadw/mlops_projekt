"""
Musterlösung zu pipeline.py (Tag 12) – NUR für den Kursleiter.
"""
import random
import subprocess

from prefect import flow, task


@task
def validate_data():
    print("Validiere Daten...")
    subprocess.run(["python", "validate_data.py"], check=True)


@task
def train_model():
    print("Trainiere Modell...")
    subprocess.run(["python", "train.py"], check=True)


@task(retries=3, retry_delay_seconds=5)
def instabiler_schritt():
    print("Führe absichtlich instabilen Schritt aus...")
    if random.random() < 0.6:
        raise RuntimeError("Simulierter, kurzzeitiger Fehler")
    print("Instabiler Schritt erfolgreich.")


@flow
def pipeline():
    validate_data()
    train_model()
    instabiler_schritt()


if __name__ == "__main__":
    pipeline()
