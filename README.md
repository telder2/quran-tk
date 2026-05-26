# Quran Semantic Explorer

A local desktop application for exploring the Sahih International English translation through meaning rather than keyword search.

---

## Capabilities

**Search** — type a query in natural English and retrieve the most relevant ayahs across all 6,236 verses. Click any result to read it in context with surrounding ayahs.

**Map** — every ayah as a 2D point on a UMAP scatter plot. Colour by revelation place, surah, juz, or auto-detected semantic neighbourhood. Zoom and pan with the toolbar. Draw a freehand lasso to select a region — the reading pane lists every ayah inside the boundary. Click any point to open it.

**Cross-reference** — enter any ayah reference (e.g. `2:255`) to surface its ten closest semantic neighbours with similarity scores. Accessible from the reading pane's "Find Similar" button.

All computation runs locally. No internet connection is required after the initial data download.

---

## Requirements

- Python 3.11 or later
- ~1 GB disk space (model weights + ChromaDB)

---

## Install

```bash
pip install -r requirements.txt
```

---

## Quick start

The translation, tafsir, and training pairs are already included in the repository. To get a working app in two commands:

```bash
python -m src.embed
python -m src.app
```

`embed.py` downloads `all-MiniLM-L6-v2` (~90 MB) and embeds all 6,236 ayahs into a local ChromaDB collection. The UMAP projection is built the first time you open the Map tab and cached for all subsequent runs.

---

## Enhanced embeddings (recommended)

The repository also includes a fine-tuning pipeline that significantly improves semantic search quality by training the embedding model on scholarly cross-references extracted from Ibn Kathir's tafsir.

**Step 1 — Fine-tune the model:**

```bash
python -m src.train
```

Uses `data/training_pairs.json` (4,474 cross-reference pairs derived from Ibn Kathir) to fine-tune `all-MiniLM-L6-v2` with `MultipleNegativesRankingLoss`. Saves the model to `models/quran-finetuned/`. Takes ~7 minutes on Apple Silicon (MPS), longer on CPU.

**Step 2 — Rebuild embeddings with the fine-tuned model and tafsir context:**

```bash
python -m src.embed --rebuild --with-tafsir
python -m src.projection --rebuild
```

`embed.py` automatically detects and uses the fine-tuned model if it exists. `--with-tafsir` appends a scholarly excerpt from Ibn Kathir to each ayah's embedding text, giving the model richer thematic context.

---

## Rebuilding from scratch

Re-download the translation:
```bash
python -m src.fetch_quran
```

Re-download Ibn Kathir tafsir from quran.com:
```bash
python -m src.fetch_tafsir
```

Re-extract cross-reference pairs from the tafsir:
```bash
python -m src.extract_pairs
```

---

## Rebuild options

```bash
python -m src.embed --rebuild              # re-embed with current model
python -m src.embed --rebuild --with-tafsir  # re-embed with tafsir context
python -m src.projection --rebuild         # recompute UMAP from current embeddings
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
├── data/
│   ├── quran_en.json         # Sahih International translation (6,236 ayahs)
│   ├── tafsir_en.json        # Ibn Kathir (Abridged) English from quran.com
│   └── training_pairs.json   # 4,474 cross-reference pairs for fine-tuning
├── db/                       # ChromaDB persistent store (gitignored)
├── cache/                    # UMAP projection cache (gitignored)
├── models/                   # Fine-tuned model (gitignored)
├── src/
│   ├── fetch_quran.py        # Download and parse the translation
│   ├── fetch_tafsir.py       # Download Ibn Kathir tafsir from quran.com
│   ├── extract_pairs.py      # Extract cross-reference pairs from tafsir
│   ├── train.py              # Fine-tune the embedding model
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

<!-- Add screenshot here -->

---

## Sources

- Translation: Sahih International via [tanzil.net](https://tanzil.net)
- Tafsir: Ibn Kathir (Abridged) English via [quran.com](https://quran.com)
