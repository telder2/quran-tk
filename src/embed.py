"""Build the ChromaDB vector store from quran_en.json using contextual chunking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_FILE = Path(__file__).parent.parent / "data" / "quran_en.json"
TAFSIR_FILE = Path(__file__).parent.parent / "data" / "tafsir_en.json"
FINETUNED_MODEL = Path(__file__).parent.parent / "models" / "quran-finetuned"
DB_DIR = Path(__file__).parent.parent / "db"
COLLECTION_NAME = "quran_sahih"
BASE_MODEL = "all-MiniLM-L6-v2"

# Context window: how many ayahs before/after to include in the embedding text.
WINDOW = 1

# Max chars of tafsir excerpt appended to each ayah's embedding text.
TAFSIR_EXCERPT_CHARS = 300


def _resolve_model() -> str:
    """Use the fine-tuned model if it exists, otherwise the base model."""
    if FINETUNED_MODEL.exists():
        print(f"Fine-tuned model found — using {FINETUNED_MODEL}")
        return str(FINETUNED_MODEL)
    return BASE_MODEL


def _load_ayahs() -> list[dict]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run: python -m src.fetch_quran"
        )
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def _load_tafsir() -> dict[str, str]:
    """
    Load tafsir and expand to all ayahs via fallback lookup.

    Ibn Kathir groups consecutive ayahs under the first ayah's key
    (e.g., 2:8 covers 2:8-11). For any ayah without a direct entry,
    walk backwards within the same surah to find the covering entry.
    """
    if not TAFSIR_FILE.exists():
        return {}
    raw: dict[str, str] = json.loads(TAFSIR_FILE.read_text(encoding="utf-8"))
    quran = json.loads(DATA_FILE.read_text(encoding="utf-8"))

    expanded: dict[str, str] = {}
    for r in quran:
        key = f"{r['surah']}:{r['ayah']}"
        if key in raw:
            expanded[key] = raw[key]
        else:
            # Walk back within the same surah to find the covering tafsir entry.
            for prev_ayah in range(r["ayah"] - 1, 0, -1):
                prev_key = f"{r['surah']}:{prev_ayah}"
                if prev_key in raw:
                    expanded[key] = raw[prev_key]
                    break
    return expanded


def _build_index(ayahs: list[dict]) -> dict[tuple[int, int], int]:
    """Map (surah, ayah) -> list position for O(1) neighbour lookup."""
    return {(r["surah"], r["ayah"]): i for i, r in enumerate(ayahs)}


def _contextual_text(
    ayahs: list[dict],
    idx: int,
    index: dict[tuple[int, int], int],
    tafsir: dict[str, str],
    with_tafsir: bool,
) -> str:
    """
    Build the embedding text for ayahs[idx] using ±WINDOW neighbours,
    never crossing surah boundaries. Optionally appends a tafsir excerpt.
    """
    focal = ayahs[idx]
    focal_surah = focal["surah"]

    parts: list[str] = []

    for offset in range(-WINDOW, WINDOW + 1):
        neighbour_idx = idx + offset
        if neighbour_idx < 0 or neighbour_idx >= len(ayahs):
            continue
        neighbour = ayahs[neighbour_idx]
        if neighbour["surah"] != focal_surah:
            continue
        if offset == 0:
            tag = f"[{focal['surah']}:{focal['ayah']}]"
            parts.append(f"{tag} {neighbour['text']}")
        else:
            parts.append(neighbour["text"])

    if with_tafsir:
        key = f"{focal['surah']}:{focal['ayah']}"
        if key in tafsir:
            excerpt = tafsir[key][:TAFSIR_EXCERPT_CHARS].rsplit(" ", 1)[0]
            parts.append(f"| {excerpt}")

    return " ".join(parts)


def build(rebuild: bool = False, with_tafsir: bool = False) -> None:
    ayahs = _load_ayahs()
    index = _build_index(ayahs)
    tafsir = _load_tafsir() if with_tafsir else {}

    if with_tafsir and not tafsir:
        print("WARNING: --with-tafsir set but tafsir_en.json not found. "
              "Run: python -m src.fetch_tafsir")

    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        if not rebuild:
            collection = client.get_collection(COLLECTION_NAME)
            print(
                f"Collection '{COLLECTION_NAME}' already exists "
                f"({collection.count()} docs). Use --rebuild to re-embed."
            )
            return
        print(f"Deleting existing collection '{COLLECTION_NAME}' ...")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    model_name = _resolve_model()
    print(f"Loading model '{model_name}' ...")
    model = SentenceTransformer(model_name)

    tafsir_note = f" + tafsir context" if (with_tafsir and tafsir) else ""
    print(f"Building embedding texts (window={WINDOW}{tafsir_note}) ...")
    texts = [
        _contextual_text(ayahs, i, index, tafsir, with_tafsir)
        for i in range(len(ayahs))
    ]
    ids = [f"{r['surah']}:{r['ayah']}" for r in ayahs]
    metadatas = [
        {
            "surah": r["surah"],
            "ayah": r["ayah"],
            "surah_name": r["surah_name"],
            "surah_name_arabic": r["surah_name_arabic"],
            "text": r["text"],
            "juz": r["juz"],
            "revelation_place": r["revelation_place"],
        }
        for r in ayahs
    ]

    BATCH = 512
    total = len(ayahs)
    for start in range(0, total, BATCH):
        end = min(start + BATCH, total)
        embeddings = model.encode(texts[start:end], show_progress_bar=False).tolist()
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings,
            metadatas=metadatas[start:end],
        )
        print(f"  Embedded {end:,}/{total:,}", end="\r")

    print(f"\nDone. Collection '{COLLECTION_NAME}' has {collection.count():,} documents.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Quran vector DB.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete and re-embed from scratch.")
    parser.add_argument("--with-tafsir", action="store_true",
                        help="Append tafsir excerpt to each ayah's embedding text.")
    args = parser.parse_args()
    build(rebuild=args.rebuild, with_tafsir=args.with_tafsir)
