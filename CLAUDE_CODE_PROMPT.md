# Quran Semantic Explorer — Build Prompt

## Project goal

Build a local Python application that lets me explore the English Quran semantically. Three core capabilities:

1. **Semantic search** — type a query in natural English, get the most relevant ayahs.
2. **UMAP map** — see all 6,236 ayahs as a 2D scatter, click any point to read.
3. **Cross-reference explorer** — given any ayah, surface its semantic neighbors.

UI is Tkinter (use `customtkinter` for a modern look). All data and models run locally — no API keys required for v1.

## Stack

- Python 3.11+
- `sentence-transformers` (`all-MiniLM-L6-v2` — fast, local, good enough)
- `chromadb` for vector storage (persistent client, single local dir)
- `umap-learn` for 2D projection
- `customtkinter` for the UI
- `matplotlib` embedded in Tkinter via `FigureCanvasTkAgg` for the map
- Data source: Sahih International translation from tanzil.net (plain text, one ayah per line). Include a small `fetch_quran.py` that downloads and parses it into a normalized JSON.

## Repo layout I want

```
quran-explorer/
├── README.md
├── requirements.txt
├── data/
│   └── quran_en.json           # built by fetch script
├── db/                          # chromadb persistent dir (gitignored)
├── cache/
│   └── umap_coords.npy          # cached projection (gitignored)
├── src/
│   ├── __init__.py
│   ├── fetch_quran.py           # download + parse Sahih International
│   ├── embed.py                 # build vector DB with prev/next context
│   ├── search.py                # semantic search API
│   ├── projection.py            # UMAP build + cache
│   └── app.py                   # Tkinter entry point
└── tests/
    └── test_search.py           # a few sanity checks
```

## Build phases — pause after each, summarize what you did, ask before moving on

### Phase 1 — Data
- Write `fetch_quran.py` that downloads Sahih International from tanzil.net, parses it into `data/quran_en.json` with this schema per ayah:
  ```
  {surah, ayah, surah_name, surah_name_arabic, text, juz, revelation_place}
  ```
- Source the surah names and revelation place from a static reference (hardcode the 114-row table; it never changes).
- Validate: exactly 6,236 ayahs, no nulls in any field. Print a summary.

### Phase 2 — Embeddings
- Write `embed.py` implementing **contextual chunking**: each row embeds `[prev ayah] + [focal ayah with surah:ayah tag] + [next ayah]`, but stores only the focal ayah's text and metadata. Never cross surah boundaries.
- Window size = 1 (one prev, one next). Make it a `WINDOW` constant at top of file.
- Use cosine distance in Chroma (`metadata={"hnsw:space": "cosine"}`).
- ID format: `"{surah}:{ayah}"`.
- Idempotent: skip if collection already populated, with a `--rebuild` flag.

### Phase 3 — Search
- Write `search.py` exposing a clean function:
  ```python
  def semantic_search(query: str, k: int = 10) -> list[SearchResult]
  ```
  where `SearchResult` is a dataclass with `surah, ayah, surah_name, text, score`.
- Also expose `find_similar(surah: int, ayah: int, k: int = 10)` for the cross-reference feature. It should retrieve the focal ayah's embedding and search, excluding the focal itself from results.
- Add `tests/test_search.py` with 3-4 sanity queries: "patience in hardship" should return ayahs from Al-Baqarah's sabr passages; "creation of the heavens" should return cosmological ayahs. Don't assert exact IDs — assert presence of expected surahs in top-10.

### Phase 4 — Projection
- Write `projection.py` that loads all embeddings from Chroma, runs UMAP with `n_neighbors=20, min_dist=0.1, metric="cosine", random_state=42`, and caches the result to `cache/umap_coords.npy`.
- Reload from cache unless `--rebuild` is passed.

### Phase 5 — UI
- `app.py` is the entry point. Use `customtkinter` with a dark theme by default.
- Three tabs:
  1. **Search** — entry field, results list. Each result shows `Surah Name 2:155`, the ayah text, and similarity score. Click a result to open it in the right-side reading pane with prev/next context (same logic as embedding window, but ±2 ayahs).
  2. **Map** — embedded matplotlib scatter of UMAP coords. Color-by selector: revelation place / surah / juz. Click a point to open in the reading pane. Pin random_state so the layout is stable across runs.
  3. **Cross-reference** — pick a surah and ayah from two dropdowns (or type "2:255"), see top-10 semantic neighbors with scores.
- Right-side reading pane is shared across tabs: shows current ayah large, prev/next in smaller text below, and a "find similar" button that jumps to the cross-reference tab pre-filled.

### Phase 6 — Polish
- README with: what it is, install steps (`pip install -r requirements.txt`, run `python -m src.fetch_quran`, `python -m src.embed`, `python -m src.app`), and a screenshot section (leave a placeholder).
- `.gitignore` for `db/`, `cache/`, `__pycache__/`, `.venv/`.
- Make sure first-run UX is clean: if `data/quran_en.json` or the Chroma DB is missing, `app.py` should print a clear "run these commands first" message rather than crashing.

## Things I care about — please respect these

- **Don't over-engineer.** No async, no FastAPI layer, no Docker. This is a local desktop app.
- **Keep modules small and importable.** I want to be able to `from src.search import semantic_search` in a Jupyter notebook later.
- **Print, don't log.** No `logging` setup for v1; `print()` is fine.
- **Type hints everywhere**, dataclasses for return types.
- **No hidden network calls** at app runtime. All network access happens in `fetch_quran.py` and only there.
- **Be respectful in copy.** This is sacred text. No emoji in the UI, no marketing tone in the README. Plain, dignified language.

## Where I want you to stop and ask

- If tanzil.net's format is different from what you expect, show me a sample of the raw data before parsing.
- Before Phase 5, show me a wireframe (ASCII is fine) of the Tkinter layout so I can adjust before you write the UI code.
- If you find yourself wanting to add a dependency that isn't in the stack above, ask first.

## Definition of done

- I can run three commands and have a working app.
- Searching "trust in God during difficulty" returns recognizably relevant ayahs (tawakkul, sabr themes) in the top 10.
- The UMAP map renders, points are clickable, color modes work.
- Cross-reference from 2:255 (Ayat al-Kursi) returns thematically related ayahs about divine attributes.
- Tests pass.

Start with Phase 1.
