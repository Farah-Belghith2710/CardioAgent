"""
RAG - step 2 (search): the chunks of Qdrant that are the closest to a question (cosine similarity of the embeddings).

    from rag_engine.search import search
    search("Que signifie un souffle systolique ?", k=4, category="clinique")

    python -m rag_engine.search "Qu'est-ce que le deuxième bruit du cœur ?"     (quick test in the terminal)

The Cardiologue agent searches the "clinique" documents; the Analyste agent may search the "technique" ones.
"""
import sys
import threading

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag_engine.ingest import COLLECTION, EMBEDDING_MODEL, QDRANT_URL

# the embedding model and the Qdrant client are made once (the first search), then reused
_model = None
_client = None


_lock = threading.Lock()    # the warm-up thread of the orchestrator and a first question may ask at the same time


def get_model():
    global _model
    with _lock:
        if _model is None:
            _model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _model


def get_client():
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, timeout=10)
    return _client


def search(question, k=4, category=None):
    # 1. the question -> 1 vector (the same model as the chunks)
    vector = list(get_model().embed([question]))[0].tolist()

    # 2. only the chunks of one category, if asked (Qdrant filter on the payload)
    query_filter = None
    if category is not None:
        query_filter = Filter(must=[FieldCondition(key="category", match=MatchValue(value=category))])

    # 3. the k closest chunks
    response = get_client().query_points(collection_name=COLLECTION, query=vector, limit=k,
                                         query_filter=query_filter, with_payload=True)
    results = []
    for point in response.points:
        results.append({
            "text": point.payload["text"],
            "source": point.payload["source"],
            "title": point.payload["title"],
            "page": point.payload["page"],
            "category": point.payload["category"],
            "score": round(point.score, 4),      # cosine similarity: 1 = the same meaning
        })
    return results


if __name__ == "__main__":
    question = " ".join(sys.argv[1:])
    if question == "":
        question = "Qu'est-ce qu'un souffle systolique ?"
    print("Question:", question)
    for result in search(question):
        print("\n[{:.3f}] {} (p. {}, {})".format(result["score"], result["title"], result["page"], result["category"]))
        print(result["text"][:300], "...")
