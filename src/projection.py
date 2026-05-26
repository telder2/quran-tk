"""UMAP 2-D projection of all ayah embeddings, with numpy cache."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DB_DIR = Path(__file__).parent.parent / "db"
CACHE_DIR = Path(__file__).parent.parent / "cache"
COORDS_FILE = CACHE_DIR / "umap_coords.npy"
META_FILE = CACHE_DIR / "umap_meta.npy"   # parallel array of (surah, ayah, juz, place)

UMAP_PARAMS = dict(
    n_neighbors=20,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
)

COLLECTION_NAME = "quran_sahih"


@dataclass
class ProjectionResult:
    """Parallel arrays; index i corresponds to the same ayah in every field."""
    x: np.ndarray          # shape (N,)
    y: np.ndarray          # shape (N,)
    surahs: np.ndarray     # int, shape (N,)
    ayahs: np.ndarray      # int, shape (N,)
    juzs: np.ndarray       # int, shape (N,)
    places: list[str]      # "Mecca" or "Medina", len N
    surah_names: list[str] # len N
    texts: list[str]       # len N


def _fetch_from_chroma() -> tuple[np.ndarray, list[dict]]:
    """Pull all embeddings and metadata from ChromaDB in order."""
    import chromadb

    if not DB_DIR.exists():
        raise RuntimeError("Vector DB not found. Run: python -m src.embed")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    total = collection.count()
    print(f"Fetching {total:,} embeddings from ChromaDB ...")

    # Chroma returns at most 5 461 items per call; page in batches.
    BATCH = 2000
    all_embeddings: list[list[float]] = []
    all_meta: list[dict] = []
    offset = 0

    while offset < total:
        result = collection.get(
            limit=BATCH,
            offset=offset,
            include=["embeddings", "metadatas"],
        )
        all_embeddings.extend(result["embeddings"])
        all_meta.extend(result["metadatas"])
        offset += len(result["ids"])
        print(f"  Fetched {offset:,}/{total:,}", end="\r")

    print()
    return np.array(all_embeddings, dtype=np.float32), all_meta


def build(rebuild: bool = False) -> ProjectionResult:
    """Return a ProjectionResult, using the cache when available."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if COORDS_FILE.exists() and META_FILE.exists() and not rebuild:
        print(f"Loading UMAP projection from cache ({COORDS_FILE.name}) ...")
        coords = np.load(COORDS_FILE)
        meta_arr = np.load(META_FILE, allow_pickle=True)
        return _meta_arr_to_result(coords, meta_arr)

    embeddings, all_meta = _fetch_from_chroma()

    print(f"Running UMAP on {len(embeddings):,} vectors ...")
    print(f"  params: {UMAP_PARAMS}")
    import umap

    reducer = umap.UMAP(**UMAP_PARAMS)
    coords = reducer.fit_transform(embeddings).astype(np.float32)

    # Build parallel metadata array for caching.
    meta_arr = np.array(
        [
            (
                m["surah"],
                m["ayah"],
                m["juz"],
                m["revelation_place"],
                m["surah_name"],
                m["text"],
            )
            for m in all_meta
        ],
        dtype=object,
    )

    np.save(COORDS_FILE, coords)
    np.save(META_FILE, meta_arr)
    print(f"Cached coords → {COORDS_FILE}")
    print(f"Cached meta   → {META_FILE}")

    return _meta_arr_to_result(coords, meta_arr)


def _meta_arr_to_result(coords: np.ndarray, meta_arr: np.ndarray) -> ProjectionResult:
    return ProjectionResult(
        x=coords[:, 0],
        y=coords[:, 1],
        surahs=meta_arr[:, 0].astype(int),
        ayahs=meta_arr[:, 1].astype(int),
        juzs=meta_arr[:, 2].astype(int),
        places=list(meta_arr[:, 3]),
        surah_names=list(meta_arr[:, 4]),
        texts=list(meta_arr[:, 5]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or reload UMAP projection.")
    parser.add_argument("--rebuild", action="store_true", help="Recompute even if cache exists.")
    args = parser.parse_args()
    result = build(rebuild=args.rebuild)
    print(f"Projection shape: {result.x.shape[0]:,} points")
    print(f"x range: [{result.x.min():.2f}, {result.x.max():.2f}]")
    print(f"y range: [{result.y.min():.2f}, {result.y.max():.2f}]")
