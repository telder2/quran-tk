"""Extract cross-reference pairs from Ibn Kathir tafsir text.

Ibn Kathir constantly cites other ayahs to explain the current one.
Each citation is a scholarly judgement that two ayahs are semantically linked —
exactly the training signal we need for contrastive fine-tuning.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
TAFSIR_FILE = DATA_DIR / "tafsir_en.json"
QURAN_FILE = DATA_DIR / "quran_en.json"
OUT_FILE = DATA_DIR / "training_pairs.json"

# Match patterns like "2:255", "112:1", etc. in the tafsir text.
_AYAH_REF = re.compile(r"\b(\d{1,3}):(\d{1,3})\b")


def _load_valid_keys(quran_file: Path) -> set[str]:
    ayahs = json.loads(quran_file.read_text(encoding="utf-8"))
    return {f"{r['surah']}:{r['ayah']}" for r in ayahs}


def build(out_file: Path = OUT_FILE) -> None:
    if not TAFSIR_FILE.exists():
        raise FileNotFoundError(
            f"{TAFSIR_FILE} not found. Run: python -m src.fetch_tafsir"
        )

    tafsir: dict[str, str] = json.loads(TAFSIR_FILE.read_text(encoding="utf-8"))
    valid_keys = _load_valid_keys(QURAN_FILE)

    pairs: set[tuple[str, str]] = set()

    for anchor_key, text in tafsir.items():
        if anchor_key not in valid_keys:
            continue
        for m in _AYAH_REF.finditer(text):
            cited_key = f"{m.group(1)}:{m.group(2)}"
            if cited_key == anchor_key:
                continue
            if cited_key not in valid_keys:
                continue
            # Store in canonical order so (A,B) and (B,A) collapse to one pair.
            pair = tuple(sorted([anchor_key, cited_key]))
            pairs.add(pair)

    pair_list = [list(p) for p in sorted(pairs)]
    out_file.write_text(
        json.dumps(pair_list, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Extracted {len(pair_list):,} unique cross-reference pairs")

    # Quick breakdown
    anchors = {p[0] for p in pair_list} | {p[1] for p in pair_list}
    print(f"Ayahs involved in at least one pair: {len(anchors):,} / 6,236")

    # Most-cited ayahs
    from collections import Counter
    cited_counts: Counter = Counter()
    for a, b in pair_list:
        cited_counts[a] += 1
        cited_counts[b] += 1
    print("Most cited ayahs:")
    for key, count in cited_counts.most_common(5):
        print(f"  {key}  cited {count} times")


if __name__ == "__main__":
    build()
