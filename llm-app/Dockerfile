# Dockerfile – Tag 17
# Ollama selbst läuft NICHT in diesem Container, sondern nativ auf dem
# Host. Diese App braucht deshalb --add-host beim docker run (siehe
# Übungsanleitung Schritt 6). Unter Linux muss Ollama außerdem auf
# 0.0.0.0 lauschen (Übungsanleitung Schritt 6), sonst erreicht der
# Container es nicht.

FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install uv && uv pip install --system --no-cache-dir -r requirements.txt

COPY prompt_template.txt .
COPY dokumente.txt .
COPY rag_utils.py .
COPY llm_app.py .

# Wichtig: host.docker.internal, NICHT localhost - der Container muss
# auf den Host zugreifen, wo Ollama läuft.
ENV OLLAMA_HOST=http://host.docker.internal:11434

# Python puffert print() in Containern ohne Terminal. Ohne diese Zeile erscheinen
# Meldungen wie "N Dokumente ... eingebettet" nicht (oder viel zu spät) im Log.
ENV PYTHONUNBUFFERED=1

EXPOSE 8001

CMD ["uvicorn", "llm_app:app", "--host", "0.0.0.0", "--port", "8001"]
