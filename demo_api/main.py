"""
Minimale Demo-API für Tag 16 (Cloud-Deployment).

Warum nicht das Image aus Tag 14/15? Dessen main.py lädt das Modell beim Import
(load_champion() braucht MLflow). Auf der Cloud-Instanz gibt es keinen MLflow-Stack -
der Container würde sofort wieder beenden und /v1/health wäre nie erreichbar.
Diese Demo hat bewusst keine MLflow-Abhängigkeit: Es geht um das Deployment-Prinzip.
"""
import socket

from fastapi import FastAPI

app = FastAPI(title="MLOps-Demo-API (Tag 16)")


@app.get("/v1/health")
def health():
    return {"status": "ok"}


@app.get("/v1/info")
def info():
    # Zeigt, auf welchem Rechner der Container läuft (Container-ID bzw. Hostname).
    return {"service": "mlops-demo", "host": socket.gethostname()}
