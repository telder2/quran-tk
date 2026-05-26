"""Download Ibn Kathir (Abridged) English tafsir from quran.com API."""

from __future__ import annotations

import json
import time
from html.parser import HTMLParser
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_FILE = DATA_DIR / "tafsir_en.json"

TAFSIR_ID = 169          # Ibn Kathir Abridged, English
API_BASE = "https://api.quran.com/api/v4"
PER_PAGE = 300           # high enough to get all ayahs of any surah in one shot
DELAY = 0.4              # seconds between chapter requests


class _StripHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(" ".join(self._parts).split())


def strip_html(html: str) -> str:
    p = _StripHTML()
    p.feed(html)
    return p.get_text()


def _fetch_chapter(chapter: int, session: requests.Session) -> dict[str, str]:
    """Return {verse_key: plain_text} for all ayahs in a chapter."""
    url = f"{API_BASE}/tafsirs/{TAFSIR_ID}/by_chapter/{chapter}"
    page = 1
    results: dict[str, str] = {}
    while True:
        resp = session.get(url, params={"per_page": PER_PAGE, "page": page}, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        for entry in data.get("tafsirs", []):
            key = entry.get("verse_key", "")
            text = strip_html(entry.get("text", ""))
            if key and text:
                results[key] = text
        pg = data.get("pagination", {})
        if pg.get("next_page") is None:
            break
        page += 1
    return results


def build(out_file: Path = OUT_FILE) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    all_tafsir: dict[str, str] = {}

    for chapter in range(1, 115):
        entries = _fetch_chapter(chapter, session)
        all_tafsir.update(entries)
        print(f"  Chapter {chapter:3d}/114 — {len(entries)} ayahs", end="\r")
        time.sleep(DELAY)

    print(f"\nFetched {len(all_tafsir):,} tafsir entries")

    # Validate
    missing = [f"{s}:{a}" for s in range(1, 115) for a in range(1, 500)
               if f"{s}:{a}" not in all_tafsir and _expected(s, a)]
    if missing:
        print(f"WARNING: {len(missing)} ayahs have no tafsir entry")
        for m in missing[:10]:
            print(" ", m)

    out_file.write_text(json.dumps(all_tafsir, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_file}")


# Surah ayah counts for validation (same as fetch_quran.py)
_AYAH_COUNTS = [
    0,7,286,200,176,120,165,206,75,129,109,123,111,43,52,99,128,111,110,98,
    135,112,78,118,64,77,227,93,88,69,60,34,30,73,54,45,83,182,88,75,85,54,
    53,89,59,37,35,38,29,18,45,60,49,62,55,78,96,29,22,24,13,14,11,11,18,12,
    12,30,52,52,44,28,28,20,56,40,31,50,40,46,42,29,19,36,25,22,17,19,26,30,
    20,15,21,11,8,8,19,5,8,8,11,11,8,3,9,5,4,7,3,6,3,5,4,5,6,
]

def _expected(surah: int, ayah: int) -> bool:
    if surah < 1 or surah > 114:
        return False
    return 1 <= ayah <= _AYAH_COUNTS[surah]


if __name__ == "__main__":
    build()
