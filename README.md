# Quran Semantic Explorer

A local desktop application for exploring the English Quran through meaning rather than keyword search.

Three capabilities:

- **Semantic search** — type a query in natural English and retrieve the most relevant ayahs across all 6,236 verses.
- **UMAP map** — every ayah as a 2D point, coloured by revelation place, surah, or juz. Click any point to read it in context.
- **Cross-reference** — given any ayah, surface its ten closest semantic neighbours.

All computation runs locally. No internet connection is required after the initial data download.

---

## Requirements

- Python 3.11 or later
- ~1 GB disk space (model weights + ChromaDB)
- ~2 minutes for the first embedding run; subsequent starts are instant

---

## Install

```bash
pip install -r requirements.txt
```

---

## Setup (run once)

**Step 1 — Download the Sahih International translation:**

```bash
python -m src.fetch_quran
```

This downloads from tanzil.net and writes `data/quran_en.json` (6,236 ayahs, validated).

**Step 2 — Build the vector database:**

```bash
python -m src.embed
```

Downloads `all-MiniLM-L6-v2` (~90 MB) and embeds all ayahs into a local ChromaDB collection using contextual chunking (each ayah is embedded together with its immediate neighbours for richer representation).

---

## Run

```bash
python -m src.app
```

The UMAP projection (`cache/umap_coords.npy`) is built the first time you open the Map tab and cached for all subsequent runs.

---

## Rebuild options

Re-embed from scratch (e.g. after changing the window size in `embed.py`):

```bash
python -m src.embed --rebuild
```

Recompute the UMAP projection (e.g. after re-embedding):

```bash
python -m src.projection --rebuild
```

---

## Using from a notebook

Each module is independently importable:

```python
from src.search import semantic_search, find_similar

results = semantic_search("trust in God during difficulty", k=10)
for r in results:
    print(f"{r.surah_name} {r.surah}:{r.ayah}  {r.score:.3f}")
    print(f"  {r.text}\n")

neighbours = find_similar(surah=2, ayah=255, k=10)
```

---

## Project layout

```
├── data/quran_en.json        # Sahih International (built by fetch_quran)
├── db/                       # ChromaDB persistent store (gitignored)
├── cache/umap_coords.npy     # UMAP projection cache (gitignored)
├── src/
│   ├── fetch_quran.py        # Download and parse the translation
│   ├── embed.py              # Build the vector DB
│   ├── search.py             # semantic_search / find_similar API
│   ├── projection.py         # UMAP build and cache
│   └── app.py                # Tkinter UI entry point
└── tests/
    └── test_search.py        # Sanity checks
```

---

## Running the tests

```bash
python -m pytest tests/ -v
```

---

## Screenshot

<!-- Add screenshot here once the application is running -->

---

## Translation source

Sahih International translation via [tanzil.net](https://tanzil.net).
