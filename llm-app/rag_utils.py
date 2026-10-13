"""
Musterlösung zu rag_utils.py (Tag 17) – NUR für den Kursleiter.
"""
import ollama
import chromadb
from chromadb.config import Settings

# anonymized_telemetry=False: schaltet die anonyme Nutzungsstatistik von Chroma ab.
# Sonst erscheint bei jedem Start die Warnung "Failed to send telemetry event ...".
# Chroma läuft hier nur im Arbeitsspeicher: Was ein Prozess einbettet, kennt kein
# anderer Prozess. Deshalb lädt llm_app.py die Datei dokumente.txt beim Start.
client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection("dokumente")

EMBED_MODEL = "nomic-embed-text"


def embed(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def add_document(doc_id: str, text: str):
    # upsert statt add: dieselbe ID darf mehrfach eingebettet werden
    collection.upsert(ids=[doc_id], embeddings=[embed(text)], documents=[text])


def load_documents(path: str) -> int:
    """Bettet jede nicht-leere Zeile der Datei als eigenes Dokument ein."""
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    for i, line in enumerate(lines):
        add_document(f"dok-{i}", line)
    return len(lines)


def retrieve(query: str, n_results: int = 2) -> list[str]:
    anzahl = collection.count()
    if anzahl == 0:
        return []
    results = collection.query(
        query_embeddings=[embed(query)], n_results=min(n_results, anzahl)
    )
    return results["documents"][0]
