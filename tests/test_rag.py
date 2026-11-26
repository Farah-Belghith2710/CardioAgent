"""
Tests of the RAG (rag_engine/): cleaning and chunking of the text, the ingestion and the search with a FAKE
embedding model and a FAKE Qdrant (no download, no database), and the metrics of the evaluation.
"""
import numpy as np

import rag_engine.evaluate_rag as evaluation
import rag_engine.ingest as ingest
import rag_engine.search as rag
from rag_engine.documents import describe


# ==================== cleaning and chunking ====================
def test_find_repeated_lines_finds_headers_and_footers():
    pages = ["Cours de cardiologie\nB1 est le premier bruit.\n1",
             "Cours de cardiologie\nB2 est le second bruit.\n2",
             "Cours de cardiologie\nLe souffle est un bruit en plus.\n3"]
    assert "Cours de cardiologie" in ingest.find_repeated_lines(pages)


def test_clean_page_removes_page_numbers_copyright_and_repeated_lines():
    text = "Cours de cardiologie\nLe souffle systo-\nlique est entendu.\n12\n© 2024 Université, all rights reserved"
    cleaned = ingest.clean_page(text, {"Cours de cardiologie"})
    assert cleaned == "Le souffle systolique est entendu."


def test_cover_page_of_hal_is_skipped():
    assert ingest.is_cover_page("HAL Id: hal-04166622 ... HAL is a multi-disciplinary open access archive")
    assert not ingest.is_cover_page("Le premier bruit du cœur")


def test_make_chunks_sizes_and_overlap():
    words = ["mot" + str(i) for i in range(600)]
    text = " ".join(words)                                   # ~4000 characters
    chunks = ingest.make_chunks(text)
    assert len(chunks) > 4
    for chunk in chunks:
        assert len(chunk) <= ingest.CHUNK_SIZE
    for i in range(len(chunks) - 1):
        last_words = chunks[i].split()[-3:]                  # 2 neighbour chunks share their border
        assert " ".join(last_words) in chunks[i + 1]
    for word in words:                                       # nothing is lost
        assert any(word in chunk.split() for chunk in chunks)


def test_short_text_is_one_chunk():
    assert ingest.make_chunks("B1 et B2.") == ["B1 et B2."]


def test_describe_known_and_new_documents():
    assert describe("challenge2016.pdf")[1] == "technique"
    assert describe("Auscultation-cardiaque-normale-4-4.pdf")[1] == "clinique"
    assert describe("nouveau_cours.pdf") == ("nouveau_cours", "clinique")


# ==================== ingestion and search with fakes ====================
class FakeEmbedding:
    def __init__(self, model_name=None):
        self.model_name = model_name

    def embed(self, texts, batch_size=32):
        for text in texts:
            yield np.array([len(text), 1.0, 0.0], dtype=np.float32)


class FakePoint:
    def __init__(self, payload, score):
        self.payload = payload
        self.score = score


class FakeQdrant:
    # remembers what the code asks, like a tiny Qdrant
    def __init__(self, url=None, timeout=None):
        self.points = []
        self.calls = []

    def collection_exists(self, name):
        return True

    def delete_collection(self, name):
        self.calls.append("delete")

    def create_collection(self, name, vectors_config=None):
        self.calls.append(("create", vectors_config.size))

    def create_payload_index(self, name, field_name=None, field_schema=None):
        self.calls.append(("index", field_name))

    def upsert(self, name, points=None):
        self.points.extend(points)

    def count(self, name):
        class Count:
            count = len(self.points)
        return Count()

    def query_points(self, collection_name=None, query=None, limit=None, query_filter=None, with_payload=None):
        self.calls.append(("query", limit, query_filter))

        class Response:
            points = [FakePoint({"text": "B1 = fermeture des valves", "source": "a.pdf", "title": "Cours A",
                                 "page": 4, "category": "clinique"}, 0.91234)]
        return Response()


def test_ingest_main_with_fakes(monkeypatch):
    fake_client = FakeQdrant()
    monkeypatch.setattr(ingest, "TextEmbedding", FakeEmbedding)
    monkeypatch.setattr(ingest, "QdrantClient", lambda url=None, timeout=None: fake_client)
    monkeypatch.setattr(ingest, "read_documents", lambda folder: [
        {"text": "Premier passage " * 10, "source": "a.pdf", "title": "A", "page": 1, "category": "clinique"},
        {"text": "Deuxième passage " * 10, "source": "b.pdf", "title": "B", "page": 2, "category": "technique"},
    ])
    ingest.main()
    assert fake_client.calls[:3] == ["delete", ("create", 3), ("index", "category")]
    assert len(fake_client.points) == 2
    assert fake_client.points[0].payload["source"] == "a.pdf"


def test_read_documents_with_fake_pdf(monkeypatch, tmp_path):
    (tmp_path / "cours.pdf").write_bytes(b"%PDF fake")
    (tmp_path / "notes.txt").write_text("not a pdf")
    monkeypatch.setattr(ingest, "read_pages", lambda path: ["Le premier bruit B1 correspond à la fermeture des "
                                                            "valves mitrale et tricuspide au début de la systole."])
    chunks = ingest.read_documents(str(tmp_path))
    assert len(chunks) == 1
    assert chunks[0]["source"] == "cours.pdf" and chunks[0]["page"] == 1 and chunks[0]["category"] == "clinique"


def test_search_with_a_category_filter(monkeypatch):
    fake_client = FakeQdrant()
    monkeypatch.setattr(rag, "get_model", lambda: FakeEmbedding())
    monkeypatch.setattr(rag, "get_client", lambda: fake_client)
    results = rag.search("Qu'est-ce que B1 ?", k=3, category="clinique")
    assert results[0]["title"] == "Cours A" and results[0]["score"] == 0.9123
    kind, limit, query_filter = fake_client.calls[-1]
    assert limit == 3 and query_filter.must[0].key == "category"
    rag.search("Qu'est-ce que B1 ?")
    assert fake_client.calls[-1][2] is None                       # no category -> no filter


# ==================== the metrics of the evaluation ====================
def test_summarize_computes_hit_at_k_and_mrr():
    rows = [{"rank": 1, "keyword_found": True}, {"rank": 3, "keyword_found": True},
            {"rank": 0, "keyword_found": False}, {"rank": 2, "keyword_found": True}]
    summary = evaluation.summarize(rows)
    assert summary["hit@1"] == 0.25 and summary["hit@3"] == 0.75 and summary["hit@5"] == 0.75
    assert summary["mrr"] == round((1 + 1 / 3 + 0 + 1 / 2) / 4, 3)
    assert summary["keyword_hit"] == 0.75


def test_evaluate_with_a_fake_search(monkeypatch):
    def fake_search(question, k=5, category=None):
        return [{"source": "challenge2016.pdf", "page": 1, "text": "2435 recordings sensitivity specificity"}]

    monkeypatch.setattr(evaluation, "search", fake_search)
    rows = evaluation.evaluate()
    assert len(rows) == len(evaluation.QUESTIONS)
    challenge_rows = [row for row in rows if "challenge2016.pdf" in row["expected"]]
    assert all(row["rank"] == 1 for row in challenge_rows)
