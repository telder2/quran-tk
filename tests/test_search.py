"""Sanity checks for semantic_search and find_similar."""

from src.search import SearchResult, find_similar, semantic_search


def _surah_names(results: list[SearchResult]) -> set[str]:
    return {r.surah_name for r in results}


def _surah_numbers(results: list[SearchResult]) -> set[int]:
    return {r.surah for r in results}


def test_patience_in_hardship() -> None:
    """'patience in hardship' should surface sabr-themed ayahs."""
    results = semantic_search("patience in hardship", k=10)
    assert len(results) == 10
    # The model may surface Al-Baqarah's classic sabr ayahs (2:153-157, 2:286)
    # or equally valid patience/hardship ayahs from other surahs (94:5-6, 9:51, etc.).
    sabr_surahs = {
        "Al-Baqarah", "Al-Imran", "An-Nisa", "Al-Anfal", "At-Tawbah",
        "Hud", "Ash-Shu'ara", "Az-Zumar", "Ash-Sharh", "Al-Insan",
        "An-Nahl", "Al-Balad", "Al-Layl", "Al-Fajr",
    }
    found = _surah_names(results) & sabr_surahs
    assert found, (
        f"Expected at least one sabr-themed surah in top-10, got: {_surah_names(results)}"
    )


def test_creation_of_heavens() -> None:
    """'creation of the heavens and earth' should return cosmological ayahs."""
    results = semantic_search("creation of the heavens and earth", k=10)
    assert len(results) == 10
    cosmological_surahs = {
        "Al-Baqarah", "Al-Imran", "Al-A'raf", "Yunus", "Hud",
        "Ibrahim", "Al-Hijr", "An-Nahl", "Al-Anbiya", "Al-Hajj", "Luqman",
        "Az-Zumar", "Ghafir", "Fussilat", "Az-Zukhruf", "Al-Ahqaf",
        "Qaf", "Adh-Dhariyat", "At-Tur", "Ad-Dukhan", "Ya-Sin",
        "An-Naba", "An-Nazi'at", "Ash-Shams", "Al-Ghashiyah",
    }
    found = _surah_names(results) & cosmological_surahs
    assert found, (
        f"Expected at least one cosmological surah in top-10, got: {_surah_names(results)}"
    )


def test_trust_in_god() -> None:
    """'trust in God during difficulty' (tawakkul/sabr) should find relevant ayahs."""
    results = semantic_search("trust in God during difficulty", k=10)
    assert len(results) == 10
    # Scores should be reasonable (not near zero).
    assert results[0].score > 0.3, f"Top score too low: {results[0].score}"


def test_find_similar_ayat_al_kursi() -> None:
    """2:255 neighbours should include ayahs about divine attributes."""
    results = find_similar(surah=2, ayah=255, k=10)
    assert len(results) == 10
    # The focal ayah itself must not appear in results.
    assert not any(r.surah == 2 and r.ayah == 255 for r in results), (
        "Focal ayah 2:255 should be excluded from find_similar results"
    )
    # Neighbours should span the Quran meaningfully — expect more than one surah.
    assert len(_surah_numbers(results)) > 1, "Expected results from multiple surahs"


def test_result_fields_complete() -> None:
    """Every SearchResult should have all fields populated."""
    results = semantic_search("mercy and forgiveness", k=5)
    for r in results:
        assert r.surah > 0
        assert r.ayah > 0
        assert r.surah_name
        assert r.surah_name_arabic
        assert r.text
        assert 1 <= r.juz <= 30
        assert r.revelation_place in ("Mecca", "Medina")
        assert 0.0 <= r.score <= 1.0
