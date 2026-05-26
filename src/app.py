"""Quran Semantic Explorer — Tkinter entry point."""

from __future__ import annotations

import json
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional

DATA_FILE = Path(__file__).parent.parent / "data" / "quran_en.json"
DB_DIR = Path(__file__).parent.parent / "db"

# ---------------------------------------------------------------------------
# Pre-flight: fail early with a clear message rather than a cryptic traceback.
# ---------------------------------------------------------------------------
def _preflight() -> None:
    missing: list[str] = []
    if not DATA_FILE.exists():
        missing.append(f"  {DATA_FILE}  →  run: python -m src.fetch_quran")
    if not DB_DIR.exists() or not any(DB_DIR.iterdir()):
        missing.append(f"  {DB_DIR}/    →  run: python -m src.embed")
    if missing:
        print("Missing required files. Please run the setup commands first:\n")
        for m in missing:
            print(m)
        sys.exit(1)


_preflight()

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from src.search import SearchResult, find_similar, semantic_search

# ---------------------------------------------------------------------------
# Quran data helpers
# ---------------------------------------------------------------------------
_AYAHS: list[dict] = json.loads(DATA_FILE.read_text(encoding="utf-8"))
_INDEX: dict[tuple[int, int], dict] = {(r["surah"], r["ayah"]): r for r in _AYAHS}
_SURAH_NAMES: list[str] = []
_seen: set[int] = set()
for _r in _AYAHS:
    if _r["surah"] not in _seen:
        _SURAH_NAMES.append(_r["surah_name"])
        _seen.add(_r["surah"])


def _get_context(surah: int, ayah: int, window: int = 2) -> list[dict]:
    """Return ayahs in [ayah-window, ayah+window], clamped within the surah."""
    results: list[dict] = []
    for offset in range(-window, window + 1):
        key = (surah, ayah + offset)
        if key in _INDEX and _INDEX[key]["surah"] == surah:
            results.append(_INDEX[key])
    return results


# ---------------------------------------------------------------------------
# Color maps for the scatter plot
# ---------------------------------------------------------------------------
_PLACE_COLORS = {"Mecca": "#4e9af1", "Medina": "#f1844e"}

N_CLUSTERS = 20

def _compute_neighborhoods(proj):
    """
    Cluster the 2D UMAP coords and label each cluster with its top TF-IDF terms.
    Returns (cluster_labels, cluster_names, cluster_centers).
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    coords = np.column_stack([proj.x, proj.y])
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(coords)

    # Build one document per cluster from the ayah texts it contains.
    cluster_docs = [""] * N_CLUSTERS
    for i, k in enumerate(labels):
        cluster_docs[k] += " " + proj.texts[i]

    # Combine English stopwords with near-universal Quran translation words
    # that appear in almost every ayah and therefore carry no cluster signal.
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    quran_common = {
        "allah", "lord", "say", "said", "shall", "unto", "upon", "thy",
        "thee", "thou", "hath", "indeed", "verily", "truly", "surely",
        "those", "them", "their", "people", "know", "knew", "will", "come",
        "came", "make", "made", "take", "took", "give", "gave", "man",
        "men", "thing", "things", "way", "day", "days", "time",
    }
    stop_words = list(ENGLISH_STOP_WORDS | quran_common)

    vec = TfidfVectorizer(stop_words=stop_words, max_features=8000, sublinear_tf=True)
    tfidf = vec.fit_transform(cluster_docs)
    terms = vec.get_feature_names_out()

    cluster_names: dict[int, str] = {}
    for k in range(N_CLUSTERS):
        row = tfidf[k].toarray()[0]
        top = row.argsort()[-3:][::-1]
        cluster_names[k] = " · ".join(terms[i] for i in top)

    return labels, cluster_names, km.cluster_centers_


def _build_color_arrays(proj, mode: str, neighborhood_cache: dict):
    """Return (colors, labels_for_legend) for the chosen color mode."""
    import numpy as np
    if mode == "Revelation Place":
        colors = [_PLACE_COLORS.get(p, "#888") for p in proj.places]
        legend = list(_PLACE_COLORS.items())
    elif mode == "Juz":
        cmap = plt.get_cmap("tab20")
        colors = [cmap((j - 1) / 30) for j in proj.juzs]
        legend = []
    elif mode == "Neighborhoods":
        if "labels" not in neighborhood_cache:
            labels, names, centers = _compute_neighborhoods(proj)
            neighborhood_cache["labels"] = labels
            neighborhood_cache["names"] = names
            neighborhood_cache["centers"] = centers
        labels = neighborhood_cache["labels"]
        cmap = plt.get_cmap("tab20")
        colors = [cmap(k / N_CLUSTERS) for k in labels]
        legend = []
    else:  # Surah
        cmap = plt.get_cmap("nipy_spectral")
        colors = [cmap((s - 1) / 114) for s in proj.surahs]
        legend = []
    return colors, legend


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class QuranApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Quran Semantic Explorer")
        self.geometry("1200x750")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._proj = None          # loaded lazily
        self._scatter = None       # matplotlib PathCollection
        self._fig = None
        self._ax = None
        self._canvas = None
        self._lasso = None
        self._lasso_mode = False
        self._color_mode = tk.StringVar(value="Revelation Place")
        self._neighborhood_cache: dict = {}
        self._current_ayah: Optional[tuple[int, int]] = None

        self._build_layout()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=2, minsize=340)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # Left panel: tabview
        self._tabs = ctk.CTkTabview(self, width=340)
        self._tabs.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self._tabs.add("Search")
        self._tabs.add("Map")
        self._tabs.add("Cross-ref")

        self._build_search_tab()
        self._build_map_tab()
        self._build_crossref_tab()

        # Right panel: reading pane
        self._build_reading_pane()

    # ------------------------------------------------------------------
    # Search tab
    # ------------------------------------------------------------------
    def _build_search_tab(self) -> None:
        tab = self._tabs.tab("Search")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        entry_frame = ctk.CTkFrame(tab, fg_color="transparent")
        entry_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        entry_frame.grid_columnconfigure(0, weight=1)

        self._search_entry = ctk.CTkEntry(
            entry_frame, placeholder_text="Search the Quran...", height=36
        )
        self._search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self._search_entry.bind("<Return>", lambda _: self._do_search())

        ctk.CTkButton(
            entry_frame, text="Go", width=48, height=36, command=self._do_search
        ).grid(row=0, column=1)

        self._search_results_frame = ctk.CTkScrollableFrame(tab)
        self._search_results_frame.grid(row=1, column=0, sticky="nsew")
        self._search_results_frame.grid_columnconfigure(0, weight=1)

    def _do_search(self) -> None:
        query = self._search_entry.get().strip()
        if not query:
            return
        for w in self._search_results_frame.winfo_children():
            w.destroy()
        ctk.CTkLabel(
            self._search_results_frame, text="Searching...", text_color="gray"
        ).grid(row=0, column=0, pady=4)
        self.update_idletasks()

        def run() -> None:
            results = semantic_search(query, k=15)
            self.after(0, lambda: self._populate_search_results(results))

        threading.Thread(target=run, daemon=True).start()

    def _populate_search_results(self, results: list[SearchResult]) -> None:
        for w in self._search_results_frame.winfo_children():
            w.destroy()
        for i, r in enumerate(results):
            self._make_result_card(
                self._search_results_frame, i, r.surah, r.ayah,
                r.surah_name, r.text, r.score
            )

    def _make_result_card(
        self,
        parent,
        row: int,
        surah: int,
        ayah: int,
        surah_name: str,
        text: str,
        score: Optional[float] = None,
    ) -> None:
        card = ctk.CTkFrame(parent, corner_radius=6)
        card.grid(row=row, column=0, sticky="ew", pady=3, padx=2)
        card.grid_columnconfigure(0, weight=1)

        header_text = f"{surah_name}  {surah}:{ayah}"
        if score is not None:
            header_text += f"    {score:.3f}"
        header = ctk.CTkLabel(
            card,
            text=header_text,
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
        )
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))

        snippet = text if len(text) <= 100 else text[:97] + "..."
        body = ctk.CTkLabel(
            card, text=snippet, wraplength=290, justify="left",
            font=ctk.CTkFont(size=11), anchor="w", text_color="#cccccc",
        )
        body.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))

        for widget in (card, header, body):
            widget.bind(
                "<Button-1>",
                lambda _, s=surah, a=ayah: self._open_ayah(s, a),
            )

    # ------------------------------------------------------------------
    # Map tab
    # ------------------------------------------------------------------
    def _build_map_tab(self) -> None:
        tab = self._tabs.tab("Map")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        ctrl = ctk.CTkFrame(tab, fg_color="transparent")
        ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        ctk.CTkLabel(ctrl, text="Color by:").pack(side="left", padx=(0, 6))
        ctk.CTkOptionMenu(
            ctrl,
            variable=self._color_mode,
            values=["Revelation Place", "Surah", "Juz", "Neighborhoods"],
            command=lambda _: self._redraw_map(),
            width=160,
        ).pack(side="left")

        self._lasso_btn = ctk.CTkButton(
            ctrl, text="Select Region", width=120, height=28,
            command=self._toggle_lasso_mode,
        )
        self._lasso_btn.pack(side="left", padx=(12, 0))

        tab.grid_rowconfigure(2, weight=0)

        self._map_load_btn = ctk.CTkButton(
            tab, text="Load Map", height=36, command=self._load_map_async,
        )
        self._map_load_btn.grid(row=1, column=0, pady=20)

        self._map_status = ctk.CTkLabel(
            tab, text="", text_color="gray", font=ctk.CTkFont(size=11)
        )
        self._map_status.grid(row=2, column=0)

    def _load_map_async(self) -> None:
        if self._proj is not None:
            return
        self._map_load_btn.configure(state="disabled", text="Loading...")
        self._map_status.configure(text="Building projection from cache (or running UMAP — this takes ~30s first time)...")

        def run() -> None:
            try:
                from src.projection import build
                proj = build()
                self.after(0, lambda: self._render_map(proj))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.after(0, lambda: self._map_status.configure(
                    text=f"Error: {exc}", text_color="#f08080"
                ))

        threading.Thread(target=run, daemon=True).start()

    def _render_map(self, proj) -> None:
        self._proj = proj
        self._map_load_btn.grid_forget()
        self._map_status.grid_forget()

        tab = self._tabs.tab("Map")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_rowconfigure(2, weight=0)
        tab.grid_rowconfigure(3, weight=0)

        self._fig, self._ax = plt.subplots(figsize=(5, 4))
        self._fig.patch.set_facecolor("#1a1a2e")
        self._ax.set_facecolor("#1a1a2e")
        self._ax.tick_params(colors="#666")
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#333")

        self._canvas = FigureCanvasTkAgg(self._fig, master=tab)
        self._canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        toolbar = NavigationToolbar2Tk(self._canvas, tab, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=2, column=0, sticky="ew")

        self._canvas.mpl_connect("button_press_event", self._on_map_click)

        from matplotlib.widgets import LassoSelector
        self._lasso = LassoSelector(
            self._ax,
            self._on_lasso_select,
            useblit=False,
            props={"color": "#f1c40f", "linewidth": 1.5},
        )
        self._lasso.set_active(False)

        self._redraw_map()

    def _redraw_map(self) -> None:
        if self._proj is None or self._ax is None:
            return
        self._ax.clear()
        self._ax.set_facecolor("#1a1a2e")
        self._ax.tick_params(colors="#666")

        proj = self._proj
        mode = self._color_mode.get()
        colors, legend_items = _build_color_arrays(proj, mode, self._neighborhood_cache)

        self._scatter = self._ax.scatter(
            proj.x, proj.y, c=colors, s=4, alpha=0.7, linewidths=0,
        )
        if legend_items:
            from matplotlib.patches import Patch
            handles = [Patch(color=c, label=lbl) for lbl, c in legend_items]
            self._ax.legend(
                handles=handles, loc="upper left",
                facecolor="#2a2a3e", edgecolor="#555", labelcolor="white",
                fontsize=8,
            )

        if mode == "Neighborhoods" and "names" in self._neighborhood_cache:
            names = self._neighborhood_cache["names"]
            centers = self._neighborhood_cache["centers"]
            for k, (cx, cy) in enumerate(centers):
                self._ax.text(
                    cx, cy, names[k],
                    fontsize=6.5, color="white", ha="center", va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="#0d0d1a", alpha=0.75,
                        edgecolor="none",
                    ),
                    zorder=10,
                )

        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._fig.tight_layout(pad=0.5)
        self._canvas.draw()

    def _toggle_lasso_mode(self) -> None:
        self._lasso_mode = not self._lasso_mode
        if self._lasso:
            self._lasso.set_active(self._lasso_mode)
        if self._lasso_mode:
            self._lasso_btn.configure(text="Cancel Selection", fg_color="#8b0000")
        else:
            self._lasso_btn.configure(text="Select Region", fg_color=["#3b8ed0", "#1f6aa5"])

    def _on_lasso_select(self, verts) -> None:
        import numpy as np
        from matplotlib.path import Path

        proj = self._proj
        path = Path(verts)
        points = np.column_stack([proj.x, proj.y])
        mask = path.contains_points(points)
        indices = np.where(mask)[0]

        # Exit lasso mode and highlight selected points.
        self._lasso_mode = False
        self._lasso.set_active(False)
        self._lasso_btn.configure(text="Select Region", fg_color=["#3b8ed0", "#1f6aa5"])

        self._redraw_map()
        if len(indices) > 0:
            self._ax.scatter(
                proj.x[indices], proj.y[indices],
                s=22, c="#f1c40f", alpha=0.95, linewidths=0, zorder=5,
            )
            self._canvas.draw()

        self._show_selection(indices)

    def _on_map_click(self, event) -> None:
        if event.inaxes != self._ax or self._proj is None:
            return
        # Don't open ayahs while zoom/pan or lasso mode is active.
        if self._fig.canvas.toolbar.mode or self._lasso_mode:
            return
        import numpy as np
        proj = self._proj
        # Find nearest point to the click in data coordinates.
        dists = (proj.x - event.xdata) ** 2 + (proj.y - event.ydata) ** 2
        idx = int(np.argmin(dists))
        self._open_ayah(int(proj.surahs[idx]), int(proj.ayahs[idx]))

    # ------------------------------------------------------------------
    # Cross-ref tab
    # ------------------------------------------------------------------
    def _build_crossref_tab(self) -> None:
        tab = self._tabs.tab("Cross-ref")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        ctrl = ctk.CTkFrame(tab, fg_color="transparent")
        ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ctrl.grid_columnconfigure(0, weight=1)
        ctrl.grid_columnconfigure(1, weight=0)
        ctrl.grid_columnconfigure(2, weight=0)

        self._xref_entry = ctk.CTkEntry(
            ctrl, placeholder_text='e.g. 2:255  or  36:58', height=36,
        )
        self._xref_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self._xref_entry.bind("<Return>", lambda _: self._do_crossref())

        ctk.CTkButton(
            ctrl, text="Find", width=60, height=36, command=self._do_crossref
        ).grid(row=0, column=1)

        self._xref_results_frame = ctk.CTkScrollableFrame(tab)
        self._xref_results_frame.grid(row=1, column=0, sticky="nsew")
        self._xref_results_frame.grid_columnconfigure(0, weight=1)

    def _do_crossref(self) -> None:
        raw = self._xref_entry.get().strip()
        if not raw:
            return
        try:
            surah_s, ayah_s = raw.split(":")
            surah, ayah = int(surah_s.strip()), int(ayah_s.strip())
        except ValueError:
            self._show_xref_error("Enter a reference like  2:255")
            return
        if (surah, ayah) not in _INDEX:
            self._show_xref_error(f"{surah}:{ayah} not found")
            return

        for w in self._xref_results_frame.winfo_children():
            w.destroy()
        ctk.CTkLabel(
            self._xref_results_frame, text="Searching...", text_color="gray"
        ).grid(row=0, column=0, pady=4)
        self.update_idletasks()
        self._open_ayah(surah, ayah)

        def run() -> None:
            results = find_similar(surah, ayah, k=10)
            self.after(0, lambda: self._populate_xref_results(results))

        threading.Thread(target=run, daemon=True).start()

    def _populate_xref_results(self, results: list[SearchResult]) -> None:
        for w in self._xref_results_frame.winfo_children():
            w.destroy()
        for i, r in enumerate(results):
            self._make_result_card(
                self._xref_results_frame, i, r.surah, r.ayah,
                r.surah_name, r.text, r.score,
            )

    def _show_xref_error(self, msg: str) -> None:
        for w in self._xref_results_frame.winfo_children():
            w.destroy()
        ctk.CTkLabel(
            self._xref_results_frame, text=msg, text_color="#f08080"
        ).grid(row=0, column=0, pady=8)

    def prefill_crossref(self, surah: int, ayah: int) -> None:
        """Called by the reading pane's 'Find Similar' button."""
        self._tabs.set("Cross-ref")
        self._xref_entry.delete(0, "end")
        self._xref_entry.insert(0, f"{surah}:{ayah}")
        self._do_crossref()

    # ------------------------------------------------------------------
    # Reading pane (right column)
    # ------------------------------------------------------------------
    def _build_reading_pane(self) -> None:
        pane = ctk.CTkFrame(self)
        pane.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        pane.grid_columnconfigure(0, weight=1)
        pane.grid_rowconfigure(1, weight=1)

        # Header label
        self._pane_header = ctk.CTkLabel(
            pane, text="Select an ayah to read",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        )
        self._pane_header.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 4))

        # Scrollable context area
        self._context_frame = ctk.CTkScrollableFrame(pane)
        self._context_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))
        self._context_frame.grid_columnconfigure(0, weight=1)

        # Find Similar button
        self._find_similar_btn = ctk.CTkButton(
            pane,
            text="Find Similar  →  Cross-ref",
            height=32,
            state="disabled",
            command=self._on_find_similar,
        )
        self._find_similar_btn.grid(row=2, column=0, pady=(0, 10), padx=14, sticky="ew")

    def _show_selection(self, indices) -> None:
        """Populate the reading pane with all ayahs inside a lasso selection."""
        for w in self._context_frame.winfo_children():
            w.destroy()
        self._find_similar_btn.configure(state="disabled")

        n = len(indices)
        if n == 0:
            self._pane_header.configure(text="No ayahs in selection")
            return

        proj = self._proj
        self._pane_header.configure(text=f"{n} ayah{'s' if n != 1 else ''} selected")

        sorted_indices = sorted(
            indices,
            key=lambda i: (int(proj.surahs[i]), int(proj.ayahs[i])),
        )
        for row, i in enumerate(sorted_indices):
            self._make_result_card(
                self._context_frame,
                row,
                int(proj.surahs[i]),
                int(proj.ayahs[i]),
                proj.surah_names[i],
                proj.texts[i],
            )

    def _open_ayah(self, surah: int, ayah: int) -> None:
        self._current_ayah = (surah, ayah)
        context = _get_context(surah, ayah, window=2)

        ayah_data = _INDEX.get((surah, ayah))
        if not ayah_data:
            return

        self._pane_header.configure(
            text=f"{ayah_data['surah_name']}  {surah}:{ayah}  —  Juz {ayah_data['juz']}"
        )

        for w in self._context_frame.winfo_children():
            w.destroy()

        for i, ctx in enumerate(context):
            is_focal = ctx["surah"] == surah and ctx["ayah"] == ayah
            ref = f"{ctx['surah_name']} {ctx['surah']}:{ctx['ayah']}"

            ref_lbl = ctk.CTkLabel(
                self._context_frame,
                text=ref,
                font=ctk.CTkFont(size=10, weight="bold" if is_focal else "normal"),
                text_color="#aaaaff" if is_focal else "#888888",
                anchor="w",
            )
            ref_lbl.grid(row=i * 2, column=0, sticky="ew", padx=8, pady=(8 if is_focal else 4, 0))

            text_lbl = ctk.CTkLabel(
                self._context_frame,
                text=ctx["text"],
                wraplength=420,
                justify="left",
                anchor="w",
                font=ctk.CTkFont(size=15 if is_focal else 12),
                text_color="#ffffff" if is_focal else "#aaaaaa",
            )
            text_lbl.grid(row=i * 2 + 1, column=0, sticky="ew", padx=8, pady=(0, 4 if is_focal else 2))

        self._find_similar_btn.configure(state="normal")

    def _on_find_similar(self) -> None:
        if self._current_ayah:
            self.prefill_crossref(*self._current_ayah)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    app = QuranApp()
    app.mainloop()


if __name__ == "__main__":
    main()
