"""SPEC §7.3 — the anchored `lookup_key` field validator."""

import pytest

from rag.ingest.lookup_key import normalize_lookup_key


@pytest.mark.parametrize(
    ("num", "expected"),
    [
        # Plain two-segment — ~1,853 of the corpus.
        ("L113-3", "L113-3"),
        # Three-or-more segment (18%) — dash segments must all survive.
        ("L113-15-2", "L113-15-2"),
        # The sharpest regression case named in SPEC §9.1: must not truncate to L132-9-3.
        ("L132-9-3-1", "L132-9-3-1"),
        # Asterisk (décret en Conseil d'État) — legally meaningful on citation_id, but
        # SPEC §7.3 says lookup_key strips it.
        ("R*113-4", "R113-4"),
        # A dot-and-space variant of the same citation as "L113-2" — both forms occur in
        # the corpus (SPEC §9.1) and must normalize identically.
        ("L. 113-2", "L113-2"),
        ("A.121-1", "A121-1"),
        # No dash at all — 4 of the corpus.
        ("L500", "L500"),
        ("A112", "A112"),
        # Already-uppercase, no-op.
        ("D52-1", "D52-1"),
    ],
)
def test_normalizes_every_measured_form(num: str, expected: str) -> None:
    assert normalize_lookup_key(num) == expected


@pytest.mark.parametrize(
    "num",
    [
        # The 21 prose annexe labels — must never collide with a real numbered article.
        "Annexe à l'article A121-1",
        "Annexe I art. R*322-58",
        "Annexe R344-7",
        # Not one of the four in-force letters.
        "X113-3",
        # Empty / whitespace-only.
        "",
        "   ",
    ],
)
def test_returns_none_for_anything_not_a_full_match(num: str) -> None:
    assert normalize_lookup_key(num) is None


def test_l113_15_does_not_swallow_l113_15_2() -> None:
    """The two are distinct in-force articles (SPEC §9.1) — normalization must not merge them."""
    assert normalize_lookup_key("L113-15") != normalize_lookup_key("L113-15-2")
