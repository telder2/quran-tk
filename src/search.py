"""Semantic search and cross-reference API over the ChromaDB vector store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = Path(__file__).parent.parent / "db"
COLLECTION_NAME = "quran_sahih"
MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level singletons — loaded once on first call.
_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


@dataclass
class SearchResult:
    surah: int
    ayah: int
    surah_name: str
    surah_name_arabic: str
    text: str
    juz: int
    revelation_place: str
    score: float  # cosine similarity in [0, 1]; higher is more similar


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading model '{MODEL_NAME}' ...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        if not DB_DIR.exists():
            raise RuntimeError(
                "Vector DB not found. Run: python -m src.embed"
            )
        client = chromadb.PersistentClient(path=str(DB_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _chroma_to_results(
    query_result: dict,
    exclude_id: str | None = None,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    ids = query_result["ids"][0]
    metadatas = query_result["metadatas"][0]
    distances = query_result["distances"][0]

    for doc_id, meta, dist in zip(ids, metadatas, distances):
        if doc_id == exclude_id:
            continue
        # ChromaDB cosine distance is 1 - similarity; convert back.
        score = round(1.0 - dist, 4)
        results.append(
            SearchResult(
                surah=meta["surah"],
                ayah=meta["ayah"],
                surah_name=meta["surah_name"],
                surah_name_arabic=meta["surah_name_arabic"],
                text=meta["text"],
                juz=meta["juz"],
                revelation_place=meta["revelation_place"],
                score=score,
            )
        )
    return results


def semantic_search(query: str, k: int = 10) -> list[SearchResult]:
    """Return the k ayahs most semantically similar to query."""
    model = _get_model()
    collection = _get_collection()
    embedding = model.encode(query).tolist()
    raw = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["metadatas", "distances"],
    )
    return _chroma_to_results(raw)


def find_similar(surah: int, ayah: int, k: int = 10) -> list[SearchResult]:
    """Return the k ayahs most similar to the given ayah, excluding itself."""
    collection = _get_collection()
    focal_id = f"{surah}:{ayah}"

    # Retrieve the stored embedding for the focal ayah.
    stored = collection.get(ids=[focal_id], include=["embeddings"])
    if stored["embeddings"] is None or len(stored["embeddings"]) == 0:
        raise ValueError(f"Ayah {focal_id} not found in the vector store.")

    embedding = stored["embeddings"][0]
    # Request one extra so we can drop the focal ayah from results.
    raw = collection.query(
        query_embeddings=[embedding],
        n_results=k + 1,
        include=["metadatas", "distances"],
    )
    results = _chroma_to_results(raw, exclude_id=focal_id)
    return results[:k]
