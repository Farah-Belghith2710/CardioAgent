"""
RAG - step 1 (indexing): read every PDF of "article for rag/", clean the text, cut it into chunks, turn each chunk
into a vector (embedding) and store the vectors in Qdrant (Service 2: vector-db).

    python -m rag_engine.ingest                                              (repo root, Qdrant on localhost:6333)
    docker compose run --rm agent-orchestrator python -m rag_engine.ingest   (inside the platform)

A new PDF dropped in the folder is indexed at the next run (the collection is rebuilt from zero each time).
Embeddings: fastembed + sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (French + English, 384 numbers
per text, ONNX: no PyTorch). Qdrant collection "cardio_guidelines", cosine similarity.
"""
import os
import re
import uuid
from collections import Counter

from fastembed import TextEmbedding
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag_engine.documents import describe

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_FOLDER = os.getenv("RAG_DOCS_FOLDER", os.path.join(HERE, "..", "article for rag"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "cardio_guidelines"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 800       # characters per chunk (~150 words: a paragraph)
CHUNK_OVERLAP = 150    # 2 neighbour chunks share 150 characters, so a sentence cut in 2 is complete in one of them
MIN_CHUNK = 80         # a smaller piece (a title alone) is not indexed


def read_pages(path):
    # the text of each page of a PDF (pypdf)
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return pages


def find_repeated_lines(pages):
    # the lines found on at least half of the pages = headers, footers, copyright lines -> removed by clean_page()
    counts = Counter()
    for text in pages:
        lines = set()
        for line in text.split("\n"):
            if line.strip() != "":
                lines.add(line.strip())
        counts.update(lines)
    repeated = set()
    if len(pages) >= 3:
        for line in counts:
            if counts[line] >= len(pages) / 2 and len(line) < 120:
                repeated.add(line)
    return repeated


def is_cover_page(text):
    # the cover page added by the HAL archive is not medical content
    return "HAL Id" in text and "open access archive" in text


def clean_page(text, repeated):
    kept = []
    for line in text.split("\n"):
        line = line.strip()
        lower = line.lower()
        if line == "" or line in repeated:
            continue
        if line.isdigit():                                  # a page number alone
            continue
        if "©" in line or "copyright" in lower or "all rights reserved" in lower or "downloaded from" in lower:
            continue
        kept.append(line)
    text = "\n".join(kept)
    text = re.sub(r"-\n(\w)", r"\1", text)    # a word cut at the end of a line: "physio-\nlogique" -> "physiologique"
    text = re.sub(r"\s+", " ", text)          # all the spaces and line breaks -> 1 space
    return text.strip()


def make_chunks(text):
    # pieces of ~CHUNK_SIZE characters, cut at a space (not in the middle of a word), with CHUNK_OVERLAP in common
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            space = text.rfind(" ", start + CHUNK_SIZE // 2, end)
            if space != -1:
                end = space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
        next_space = text.find(" ", start, end)    # start the next chunk at the beginning of a word
        if next_space != -1:
            start = next_space + 1
    return chunks


def read_documents(folder):
    # all the chunks of all the PDFs of the folder, with their source, title, page and category
    chunks = []
    for file_name in sorted(os.listdir(folder)):
        if not file_name.lower().endswith(".pdf"):
            continue
        title, category = describe(file_name)
        pages = read_pages(os.path.join(folder, file_name))
        repeated = find_repeated_lines(pages)
        count = 0
        for page_number in range(1, len(pages) + 1):
            text = pages[page_number - 1]
            if is_cover_page(text):
                continue
            for piece in make_chunks(clean_page(text, repeated)):
                if len(piece) >= MIN_CHUNK:
                    chunks.append({"text": piece, "source": file_name, "title": title, "page": page_number,
                                   "category": category})
                    count = count + 1
        print(file_name, ":", len(pages), "pages ->", count, "chunks (" + category + ")")
    return chunks


def main():
    chunks = read_documents(DOCS_FOLDER)
    print("total:", len(chunks), "chunks")

    # the embeddings: 1 vector of 384 numbers per chunk (the model is downloaded the first time, ~220 MB)
    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    texts = []
    for chunk in chunks:
        texts.append(chunk["text"])
    vectors = list(model.embed(texts, batch_size=32))

    # Qdrant: the collection is rebuilt from zero (so a removed PDF disappears too)
    client = QdrantClient(url=QDRANT_URL, timeout=120)    # creating a collection can take ~15 s on a busy PC
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE))
    client.create_payload_index(COLLECTION, field_name="category", field_schema="keyword")    # fast filter

    points = []
    for i in range(len(chunks)):
        # the same id for the same chunk at every run (uuid5 = a fixed id made from a text)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunks[i]["source"] + "#" + str(chunks[i]["page"]) + "#" + str(i)))
        points.append(PointStruct(id=point_id, vector=vectors[i].tolist(), payload=chunks[i]))
    for start in range(0, len(points), 256):
        client.upsert(COLLECTION, points=points[start:start + 256])
    print("Qdrant:", client.count(COLLECTION).count, "chunks in the collection", COLLECTION, "at", QDRANT_URL)


if __name__ == "__main__":
    main()
