"""Download and parse the Sahih International translation from tanzil.net."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_FILE = DATA_DIR / "quran_en.json"
TANZIL_URL = "https://tanzil.net/trans/en.sahih"

# 114-row reference table: (surah_number, name_english, name_arabic, ayah_count, revelation_place)
SURAH_REF: list[tuple[int, str, str, int, str]] = [
    (1,   "Al-Fatihah",     "الفاتحة",    7,   "Mecca"),
    (2,   "Al-Baqarah",     "البقرة",     286, "Medina"),
    (3,   "Al-Imran",       "آل عمران",   200, "Medina"),
    (4,   "An-Nisa",        "النساء",     176, "Medina"),
    (5,   "Al-Ma'idah",     "المائدة",    120, "Medina"),
    (6,   "Al-An'am",       "الأنعام",    165, "Mecca"),
    (7,   "Al-A'raf",       "الأعراف",    206, "Mecca"),
    (8,   "Al-Anfal",       "الأنفال",    75,  "Medina"),
    (9,   "At-Tawbah",      "التوبة",     129, "Medina"),
    (10,  "Yunus",          "يونس",       109, "Mecca"),
    (11,  "Hud",            "هود",        123, "Mecca"),
    (12,  "Yusuf",          "يوسف",       111, "Mecca"),
    (13,  "Ar-Ra'd",        "الرعد",      43,  "Medina"),
    (14,  "Ibrahim",        "إبراهيم",    52,  "Mecca"),
    (15,  "Al-Hijr",        "الحجر",      99,  "Mecca"),
    (16,  "An-Nahl",        "النحل",      128, "Mecca"),
    (17,  "Al-Isra",        "الإسراء",    111, "Mecca"),
    (18,  "Al-Kahf",        "الكهف",      110, "Mecca"),
    (19,  "Maryam",         "مريم",       98,  "Mecca"),
    (20,  "Ta-Ha",          "طه",         135, "Mecca"),
    (21,  "Al-Anbiya",      "الأنبياء",   112, "Mecca"),
    (22,  "Al-Hajj",        "الحج",       78,  "Medina"),
    (23,  "Al-Mu'minun",    "المؤمنون",   118, "Mecca"),
    (24,  "An-Nur",         "النور",      64,  "Medina"),
    (25,  "Al-Furqan",      "الفرقان",    77,  "Mecca"),
    (26,  "Ash-Shu'ara",    "الشعراء",    227, "Mecca"),
    (27,  "An-Naml",        "النمل",      93,  "Mecca"),
    (28,  "Al-Qasas",       "القصص",      88,  "Mecca"),
    (29,  "Al-Ankabut",     "العنكبوت",   69,  "Mecca"),
    (30,  "Ar-Rum",         "الروم",      60,  "Mecca"),
    (31,  "Luqman",         "لقمان",      34,  "Mecca"),
    (32,  "As-Sajdah",      "السجدة",     30,  "Mecca"),
    (33,  "Al-Ahzab",       "الأحزاب",    73,  "Medina"),
    (34,  "Saba",           "سبأ",        54,  "Mecca"),
    (35,  "Fatir",          "فاطر",       45,  "Mecca"),
    (36,  "Ya-Sin",         "يس",         83,  "Mecca"),
    (37,  "As-Saffat",      "الصافات",    182, "Mecca"),
    (38,  "Sad",            "ص",          88,  "Mecca"),
    (39,  "Az-Zumar",       "الزمر",      75,  "Mecca"),
    (40,  "Ghafir",         "غافر",       85,  "Mecca"),
    (41,  "Fussilat",       "فصلت",       54,  "Mecca"),
    (42,  "Ash-Shura",      "الشورى",     53,  "Mecca"),
    (43,  "Az-Zukhruf",     "الزخرف",     89,  "Mecca"),
    (44,  "Ad-Dukhan",      "الدخان",     59,  "Mecca"),
    (45,  "Al-Jathiyah",    "الجاثية",    37,  "Mecca"),
    (46,  "Al-Ahqaf",       "الأحقاف",    35,  "Mecca"),
    (47,  "Muhammad",       "محمد",       38,  "Medina"),
    (48,  "Al-Fath",        "الفتح",      29,  "Medina"),
    (49,  "Al-Hujurat",     "الحجرات",    18,  "Medina"),
    (50,  "Qaf",            "ق",          45,  "Mecca"),
    (51,  "Adh-Dhariyat",   "الذاريات",   60,  "Mecca"),
    (52,  "At-Tur",         "الطور",      49,  "Mecca"),
    (53,  "An-Najm",        "النجم",      62,  "Mecca"),
    (54,  "Al-Qamar",       "القمر",      55,  "Mecca"),
    (55,  "Ar-Rahman",      "الرحمن",     78,  "Medina"),
    (56,  "Al-Waqi'ah",     "الواقعة",    96,  "Mecca"),
    (57,  "Al-Hadid",       "الحديد",     29,  "Medina"),
    (58,  "Al-Mujadila",    "المجادلة",   22,  "Medina"),
    (59,  "Al-Hashr",       "الحشر",      24,  "Medina"),
    (60,  "Al-Mumtahanah",  "الممتحنة",   13,  "Medina"),
    (61,  "As-Saf",         "الصف",       14,  "Medina"),
    (62,  "Al-Jumu'ah",     "الجمعة",     11,  "Medina"),
    (63,  "Al-Munafiqun",   "المنافقون",  11,  "Medina"),
    (64,  "At-Taghabun",    "التغابن",    18,  "Medina"),
    (65,  "At-Talaq",       "الطلاق",     12,  "Medina"),
    (66,  "At-Tahrim",      "التحريم",    12,  "Medina"),
    (67,  "Al-Mulk",        "الملك",      30,  "Mecca"),
    (68,  "Al-Qalam",       "القلم",      52,  "Mecca"),
    (69,  "Al-Haqqah",      "الحاقة",     52,  "Mecca"),
    (70,  "Al-Ma'arij",     "المعارج",    44,  "Mecca"),
    (71,  "Nuh",            "نوح",        28,  "Mecca"),
    (72,  "Al-Jinn",        "الجن",       28,  "Mecca"),
    (73,  "Al-Muzzammil",   "المزمل",     20,  "Mecca"),
    (74,  "Al-Muddaththir", "المدثر",     56,  "Mecca"),
    (75,  "Al-Qiyamah",     "القيامة",    40,  "Mecca"),
    (76,  "Al-Insan",       "الإنسان",    31,  "Medina"),
    (77,  "Al-Mursalat",    "المرسلات",   50,  "Mecca"),
    (78,  "An-Naba",        "النبأ",      40,  "Mecca"),
    (79,  "An-Nazi'at",     "النازعات",   46,  "Mecca"),
    (80,  "Abasa",          "عبس",        42,  "Mecca"),
    (81,  "At-Takwir",      "التكوير",    29,  "Mecca"),
    (82,  "Al-Infitar",     "الانفطار",   19,  "Mecca"),
    (83,  "Al-Mutaffifin",  "المطففين",   36,  "Mecca"),
    (84,  "Al-Inshiqaq",    "الانشقاق",   25,  "Mecca"),
    (85,  "Al-Buruj",       "البروج",     22,  "Mecca"),
    (86,  "At-Tariq",       "الطارق",     17,  "Mecca"),
    (87,  "Al-A'la",        "الأعلى",     19,  "Mecca"),
    (88,  "Al-Ghashiyah",   "الغاشية",    26,  "Mecca"),
    (89,  "Al-Fajr",        "الفجر",      30,  "Mecca"),
    (90,  "Al-Balad",       "البلد",      20,  "Mecca"),
    (91,  "Ash-Shams",      "الشمس",      15,  "Mecca"),
    (92,  "Al-Layl",        "الليل",      21,  "Mecca"),
    (93,  "Ad-Duha",        "الضحى",      11,  "Mecca"),
    (94,  "Ash-Sharh",      "الشرح",      8,   "Mecca"),
    (95,  "At-Tin",         "التين",      8,   "Mecca"),
    (96,  "Al-Alaq",        "العلق",      19,  "Mecca"),
    (97,  "Al-Qadr",        "القدر",      5,   "Mecca"),
    (98,  "Al-Bayyinah",    "البينة",     8,   "Medina"),
    (99,  "Az-Zalzalah",    "الزلزلة",    8,   "Medina"),
    (100, "Al-Adiyat",      "العاديات",   11,  "Mecca"),
    (101, "Al-Qari'ah",     "القارعة",    11,  "Mecca"),
    (102, "At-Takathur",    "التكاثر",    8,   "Mecca"),
    (103, "Al-Asr",         "العصر",      3,   "Mecca"),
    (104, "Al-Humazah",     "الهمزة",     9,   "Mecca"),
    (105, "Al-Fil",         "الفيل",      5,   "Mecca"),
    (106, "Quraysh",        "قريش",       4,   "Mecca"),
    (107, "Al-Ma'un",       "الماعون",    7,   "Mecca"),
    (108, "Al-Kawthar",     "الكوثر",     3,   "Mecca"),
    (109, "Al-Kafirun",     "الكافرون",   6,   "Mecca"),
    (110, "An-Nasr",        "النصر",      3,   "Medina"),
    (111, "Al-Masad",       "المسد",      5,   "Mecca"),
    (112, "Al-Ikhlas",      "الإخلاص",    4,   "Mecca"),
    (113, "Al-Falaq",       "الفلق",      5,   "Mecca"),
    (114, "An-Nas",         "الناس",      6,   "Mecca"),
]

# Juz boundary: (surah, ayah) marks the first ayah of each juz (1-indexed).
JUZ_STARTS: list[tuple[int, int]] = [
    (1,  1),   # Juz 1
    (2,  142), # Juz 2
    (2,  253), # Juz 3
    (3,  93),  # Juz 4
    (4,  24),  # Juz 5
    (4,  148), # Juz 6
    (5,  82),  # Juz 7
    (6,  111), # Juz 8
    (7,  88),  # Juz 9
    (8,  41),  # Juz 10
    (9,  93),  # Juz 11
    (11, 6),   # Juz 12
    (12, 53),  # Juz 13
    (15, 1),   # Juz 14
    (17, 1),   # Juz 15
    (18, 75),  # Juz 16
    (21, 1),   # Juz 17
    (23, 1),   # Juz 18
    (25, 21),  # Juz 19
    (27, 56),  # Juz 20
    (29, 46),  # Juz 21
    (33, 31),  # Juz 22
    (36, 28),  # Juz 23
    (39, 32),  # Juz 24
    (41, 47),  # Juz 25
    (46, 1),   # Juz 26
    (51, 31),  # Juz 27
    (58, 1),   # Juz 28
    (67, 1),   # Juz 29
    (78, 1),   # Juz 30
]


def _build_lookup_maps() -> tuple[
    dict[int, tuple[str, str, str]],
    dict[tuple[int, int], int],
]:
    """Return (surah_info, ayah_to_juz) lookup maps built from the reference tables."""
    surah_info: dict[int, tuple[str, str, str]] = {
        num: (name_en, name_ar, place)
        for num, name_en, name_ar, _, place in SURAH_REF
    }

    # Convert JUZ_STARTS to a flat ordered list of (surah, ayah) with juz number.
    juz_starts_indexed = [(s, a, j + 1) for j, (s, a) in enumerate(JUZ_STARTS)]

    # Build per-ayah juz by iterating over all ayahs in Quran order.
    ayah_to_juz: dict[tuple[int, int], int] = {}
    juz_idx = 0
    for num, _, _, count, _ in SURAH_REF:
        for ayah in range(1, count + 1):
            # Advance juz pointer if we've reached the next juz boundary.
            while (
                juz_idx + 1 < len(juz_starts_indexed)
                and (num, ayah) >= (
                    juz_starts_indexed[juz_idx + 1][0],
                    juz_starts_indexed[juz_idx + 1][1],
                )
            ):
                juz_idx += 1
            ayah_to_juz[(num, ayah)] = juz_starts_indexed[juz_idx][2]

    return surah_info, ayah_to_juz


def _download_translation() -> str:
    """Fetch the Sahih International text from tanzil.net."""
    print(f"Downloading from {TANZIL_URL} ...")
    resp = requests.get(TANZIL_URL, timeout=30)
    resp.raise_for_status()
    return resp.text


def _parse_translation(raw: str) -> dict[tuple[int, int], str]:
    """Parse tanzil pipe-delimited translation into {(surah, ayah): text}."""
    ayahs: dict[tuple[int, int], str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("|"):
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        try:
            s, a = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        ayahs[(s, a)] = parts[2].strip()
    return ayahs


def build(out_file: Path = OUT_FILE) -> None:
    """Download, parse, validate, and write quran_en.json."""
    out_file.parent.mkdir(parents=True, exist_ok=True)

    raw = _download_translation()

    # Show a sample before committing to parse — catch unexpected formats early.
    sample_lines = [l for l in raw.splitlines() if l.strip() and not l.startswith("#")][:3]
    print("Sample lines from download:")
    for ln in sample_lines:
        print(" ", ln)

    translation = _parse_translation(raw)
    surah_info, ayah_to_juz = _build_lookup_maps()

    records: list[dict] = []
    for num, name_en, name_ar, count, place in SURAH_REF:
        for ayah in range(1, count + 1):
            key = (num, ayah)
            text = translation.get(key, "").strip()
            juz = ayah_to_juz.get(key, 0)
            records.append(
                {
                    "surah": num,
                    "ayah": ayah,
                    "surah_name": name_en,
                    "surah_name_arabic": name_ar,
                    "text": text,
                    "juz": juz,
                    "revelation_place": place,
                }
            )

    # Validation
    assert len(records) == 6236, f"Expected 6,236 ayahs, got {len(records)}"
    null_fields: list[str] = []
    for r in records:
        for field, val in r.items():
            if val is None or val == "" or val == 0:
                null_fields.append(f"{r['surah']}:{r['ayah']} — {field} = {val!r}")
    if null_fields:
        print("WARNING: blank/null fields found:")
        for f in null_fields[:20]:
            print(" ", f)
        sys.exit(1)

    out_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(records):,} ayahs to {out_file}")

    # Summary
    meccan = sum(1 for r in records if r["revelation_place"] == "Mecca")
    medinan = sum(1 for r in records if r["revelation_place"] == "Medina")
    juz_counts = {}
    for r in records:
        juz_counts[r["juz"]] = juz_counts.get(r["juz"], 0) + 1
    print(f"Meccan ayahs : {meccan:,}")
    print(f"Medinan ayahs: {medinan:,}")
    print(f"Juz range    : {min(juz_counts)} – {max(juz_counts)}")
    print(f"Unique surahs: {len({r['surah'] for r in records})}")


if __name__ == "__main__":
    build()
