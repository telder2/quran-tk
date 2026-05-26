"""Fine-tune the sentence transformer on Quran cross-reference pairs.

Uses MultipleNegativesRankingLoss: for each (anchor, positive) pair in the batch,
every other positive in the same batch acts as a hard negative. No negative
labelling required — the loss function handles it automatically.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
QURAN_FILE = DATA_DIR / "quran_en.json"
TAFSIR_FILE = DATA_DIR / "tafsir_en.json"
PAIRS_FILE = DATA_DIR / "training_pairs.json"
MODEL_OUT = Path(__file__).parent.parent / "models" / "quran-finetuned"

BASE_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 16
EPOCHS = 3
WARMUP_RATIO = 0.1


def _load_texts(with_tafsir: bool) -> dict[str, str]:
    """Return {verse_key: embedding_text} optionally enriched with tafsir."""
    ayahs = json.loads(QURAN_FILE.read_text(encoding="utf-8"))
    texts = {f"{r['surah']}:{r['ayah']}": r["text"] for r in ayahs}

    if with_tafsir and TAFSIR_FILE.exists():
        raw_tafsir: dict[str, str] = json.loads(TAFSIR_FILE.read_text(encoding="utf-8"))
        ayahs_list = json.loads(QURAN_FILE.read_text(encoding="utf-8"))
        # Expand with fallback: ayahs not directly in tafsir inherit from
        # the nearest preceding entry in the same surah.
        tafsir: dict[str, str] = {}
        for r in ayahs_list:
            key = f"{r['surah']}:{r['ayah']}"
            if key in raw_tafsir:
                tafsir[key] = raw_tafsir[key]
            else:
                for prev in range(r["ayah"] - 1, 0, -1):
                    pkey = f"{r['surah']}:{prev}"
                    if pkey in raw_tafsir:
                        tafsir[key] = raw_tafsir[pkey]
                        break
        enriched = 0
        for key in texts:
            if key in tafsir:
                excerpt = tafsir[key][:300].rsplit(" ", 1)[0]
                texts[key] = f"{texts[key]} | {excerpt}"
                enriched += 1
        print(f"Tafsir context added to {enriched:,} ayahs")

    return texts


def build(with_tafsir: bool = True) -> None:
    if not PAIRS_FILE.exists():
        raise FileNotFoundError(
            f"{PAIRS_FILE} not found. Run: python -m src.extract_pairs"
        )

    pairs = json.loads(PAIRS_FILE.read_text(encoding="utf-8"))
    texts = _load_texts(with_tafsir)

    # Filter to pairs where both keys have text.
    valid = [(a, b) for a, b in pairs if a in texts and b in texts]
    print(f"Training pairs: {len(valid):,}  (dropped {len(pairs)-len(valid)} with missing text)")

    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    print(f"Loading base model '{BASE_MODEL}' ...")
    model = SentenceTransformer(BASE_MODEL)

    train_examples = [InputExample(texts=[texts[a], texts[b]]) for a, b in valid]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    total_steps = math.ceil(len(train_examples) / BATCH_SIZE) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    print(f"Steps: {total_steps:,}  Warmup: {warmup_steps}  Epochs: {EPOCHS}  Batch: {BATCH_SIZE}")

    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        output_path=str(MODEL_OUT),
        show_progress_bar=True,
    )
    print(f"\nFine-tuned model saved to {MODEL_OUT}")
    print("Now rebuild embeddings: python -m src.embed --rebuild")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune embedding model on tafsir pairs.")
    parser.add_argument(
        "--no-tafsir", action="store_true",
        help="Train on raw ayah text only (no tafsir context in training texts)."
    )
    args = parser.parse_args()
    build(with_tafsir=not args.no_tafsir)
