"""SPEC §9.1 — the unanchored query-side scanner and the membership check."""

import pytest

from rag.ingest.lookup_key import normalize_lookup_key
from rag.retrieval.short_circuit import (
    ShortCircuitPath,
    resolve_short_circuit,
    scan_article_reference,
)


@pytest.mark.parametrize(
    ("raw_turn", "expected"),
    [
        # Plain two-segment — ~1,853 of the corpus.
        ("Que dit L113-3 sur la résiliation ?", "L113-3"),
        # Three-or-more segment (18%).
        ("Et pour L113-15-2, la garantie change ?", "L113-15-2"),
        # Asterisk (décret en Conseil d'État) — legally meaningful, must survive the scan.
        ("Le décret R*113-4 s'applique-t-il ici ?", "R*113-4"),
        # No dash at all.
        ("L500 couvre-t-il ce cas ?", "L500"),
    ],
)
def test_scans_all_four_measured_forms(raw_turn: str, expected: str) -> None:
    assert scan_article_reference(raw_turn) == expected


def test_l132_9_3_1_is_returned_whole_not_truncated_to_a_real_but_shorter_prefix() -> None:
    """The sharpest regression case named in SPEC §9.1: unanchored, leftmost-first
    semantics without a maximal-munch guarantee could hand back `L132-9-3` — itself a
    plausible-looking prefix — instead of the full four-segment reference."""
    assert scan_article_reference("Voir L132-9-3-1 pour le rachat.") == "L132-9-3-1"


def test_l113_15_2_is_not_truncated_to_l113_15() -> None:
    """`L113-15` is itself a real in-force article distinct from `L113-15-2` — the exact
    silent-wrong-answer shape SPEC §9.1 calls out for the query stage."""
    assert scan_article_reference("Et le L113-15-2, c'est pareil pour l'auto ?") == "L113-15-2"


@pytest.mark.parametrize(
    "raw_turn",
    [
        "Quelle est la franchise pour un dégât des eaux ?",
        "",
        "   ",
    ],
)
def test_returns_none_when_no_reference_is_present(raw_turn: str) -> None:
    assert scan_article_reference(raw_turn) is None


def test_prose_annexe_label_matches_its_own_embedded_article_number() -> None:
    """The 21 prose annexe labels each embed the number of the real article they attach
    to — 'Annexe à l'article A121-1' contains 'A121-1', a genuine in-force article distinct
    from the annexe itself (verified against the corpus). The scanner extracting it is
    correct: SPEC §7.3's collision concern is about the *field validator* never normalizing
    the annexe's own `num` to that key, not about suppressing a real citation embedded in
    free text — and nothing about the shape of 'A121-1' distinguishes it from any other
    reference once it's inside a sentence."""
    assert scan_article_reference("Annexe à l'article A121-1") == "A121-1"


@pytest.mark.parametrize(
    ("raw_turn", "expected_key"),
    [
        ("Que dit L113-2 sur la résiliation ?", "L113-2"),
        ("Que dit L. 113-2 sur la résiliation ?", "L113-2"),
        ("Et A.121-1, il dit quoi ?", "A121-1"),
    ],
)
def test_both_citation_spellings_normalize_to_the_same_key(
    raw_turn: str, expected_key: str
) -> None:
    candidate = scan_article_reference(raw_turn)
    assert candidate is not None
    assert normalize_lookup_key(candidate) == expected_key


@pytest.mark.parametrize(
    "num",
    ["L113-3", "L113-15-2", "L132-9-3-1", "R*113-4", "L. 113-2", "L500", "D52-1"],
)
def test_scanner_and_field_validator_agree_on_every_measured_form(num: str) -> None:
    """'One normalizer, two patterns' (SPEC §9.1): the anchored field validator and the
    unanchored scanner must produce the identical `lookup_key` for the same citation,
    whether it's the whole `num` field value or embedded in a user's raw turn."""
    embedded = f"Bonjour, {num} s'applique-t-il ici ?"
    scanned = scan_article_reference(embedded)
    assert scanned is not None
    assert normalize_lookup_key(scanned) == normalize_lookup_key(num)


def test_path_1_reference_and_membership_passes_short_circuits() -> None:
    result = resolve_short_circuit("Que dit L113-2 sur la résiliation ?", {"L113-2"})
    assert result.path is ShortCircuitPath.RESOLVED
    assert result.lookup_key == "L113-2"


def test_path_1_resolves_the_dot_spaced_spelling_against_the_normalized_membership_set() -> None:
    """The membership set holds normalized keys; a dot-spaced raw-turn spelling must still
    resolve, exercising scan → normalize → membership as one composed path."""
    result = resolve_short_circuit("Que dit L. 113-2 sur la résiliation ?", {"L113-2"})
    assert result.path is ShortCircuitPath.RESOLVED
    assert result.lookup_key == "L113-2"


def test_path_1_uses_the_spec_accepted_cost_example() -> None:
    """SPEC §9.1's own worked example: the reference short-circuits and the auto framing
    in the same turn never reaches retrieval — that's the accepted cost, not a bug."""
    result = resolve_short_circuit(
        "et le L113-15-2, c'est pareil pour l'auto ?", {"L113-15-2", "L113-15"}
    )
    assert result.path is ShortCircuitPath.RESOLVED
    assert result.lookup_key == "L113-15-2"


def test_path_2_reference_and_membership_fails_falls_through_to_search() -> None:
    """A regex match that isn't in the loaded `lookup_key` set is normal retrieval, never
    a confidently wrong article."""
    result = resolve_short_circuit("Que dit L999-9 sur la résiliation ?", {"L113-2"})
    assert result.path is ShortCircuitPath.MEMBERSHIP_FAILED
    assert result.lookup_key is None


def test_path_3_no_reference_falls_through_to_search() -> None:
    result = resolve_short_circuit("Quelle est la franchise pour un dégât des eaux ?", {"L113-2"})
    assert result.path is ShortCircuitPath.NO_REFERENCE
    assert result.lookup_key is None


def test_path_2_and_path_3_are_distinguishable_even_though_both_fall_through() -> None:
    """SPEC §9.1's paths 2 and 3 both fall through to hybrid search, but they are not the
    same outcome: only path 2 saw a reference. A caller deciding whether the raw turn
    should still feed a later fabricated-reference check needs to tell them apart."""
    membership_failed = resolve_short_circuit("L999-9 ?", {"L113-2"})
    no_reference = resolve_short_circuit("La franchise ?", {"L113-2"})
    assert membership_failed.path is not no_reference.path
