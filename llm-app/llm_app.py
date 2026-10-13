"""
Musterlösung zu llm_app.py (Tag 17) – NUR für den Kursleiter.
"""
import os
from jinja2 import Template
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama

import rag_utils

app = FastAPI(title="LLM-Anwendung mit RAG")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
ollama_client = ollama.Client(host=OLLAMA_HOST)

CHAT_MODEL = "llama3.2:1b"

with open("prompt_template.txt", encoding="utf-8") as f:
    prompt_template = Template(f.read())

# Chroma läuft im Arbeitsspeicher und ist beim Start leer. Deshalb werden die
# Dokumente hier beim Start eingebettet (lokal und im Container gleich).
# rag_utils.embed() nutzt den Standard-Client der ollama-Bibliothek; der liest
# dieselbe Umgebungsvariable OLLAMA_HOST wie ollama_client oben.
DOKUMENTE = "dokumente.txt"
if os.path.exists(DOKUMENTE):
    try:
        anzahl = rag_utils.load_documents(DOKUMENTE)
        print(f"{anzahl} Dokumente aus {DOKUMENTE} eingebettet.")
    except Exception as e:
        print(f"Warnung: Dokumente konnten nicht eingebettet werden: {e}")


class ChatRequest(BaseModel):
    message: str


def generate_stream(prompt: str):
    try:
        stream = ollama_client.chat(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            yield chunk["message"]["content"]
    except Exception:
        yield "Fehler: LLM-Dienst nicht erreichbar. Läuft Ollama?"


@app.post("/chat")
def chat(request: ChatRequest):
    # Der Kontext wird VOR dem Streaming geholt. Ist Ollama nicht erreichbar,
    # scheitert schon das Einbetten der Frage - das muss hier abgefangen werden,
    # sonst wird daraus ein unverständlicher 500er.
    try:
        context_chunks = rag_utils.retrieve(request.message)
    except Exception:
        raise HTTPException(
            status_code=503, detail="LLM-Dienst nicht erreichbar. Läuft Ollama?"
        )
    context = "\n".join(context_chunks)

    prompt = prompt_template.render(context=context, question=request.message)

    return StreamingResponse(generate_stream(prompt), media_type="text/plain")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ollama_host": OLLAMA_HOST,
        "dokumente": rag_utils.collection.count(),
    }