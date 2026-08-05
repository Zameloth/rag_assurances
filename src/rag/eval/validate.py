"""Golden-set validator (SPEC §12.1-§12.4, ADR-0010, #33).

Same shape as `ingest.assertions`: a set of small, independent checks, each returning the
violations it found rather than raising on the first one, aggregated by one entry point
that raises everything together — a golden-set item that is wrong for three unrelated
reasons should say so once, not on three separate runs.

**The empty-cell check is the one SPEC calls out by name** (§12.4): `reponse_sans_article`
and `refus:hors_corpus` both leave `gold_articles` empty, and they mean opposite things —
"no article clears the floor" vs. "nothing exists to retrieve". This validator never infers
`expected_state` from which fields are empty; it takes the annotator's declared state as
given and checks *that state's* required emptiness pattern, so a mislabeled item (the wrong
state for what was actually annotated) is exactly what it catches.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from re import match as re_match

from rag.eval.corpus import article_cids as corpus_article_cids
from rag.eval.corpus import fiche_chunk_texts as corpus_fiche_chunk_texts
from rag.eval.corpus import fiche_ids as corpus_fiche_ids
from rag.eval.corpus import load_articles
from rag.eval.schema import EXPECTED_STATES, GoldenItem

__all__ = ["GoldenSetValidationError", "validate_golden_set", "validate_golden_set_against_corpus"]

_ID_PATTERN = r"^gs-\d+$"
_MIN_EXPECTED_POINTS = 1
_MAX_EXPECTED_POINTS = 3


class GoldenSetValidationError(Exception):
    """One or more golden-set items fail SPEC §12's acceptance rules. Raised with every
    violation found, not just the first."""


def validate_golden_set(
    items: Iterable[GoldenItem],
    *,
    fiche_ids: set[str],
    article_cids: set[str],
    fiche_chunk_texts: Mapping[str, list[str]],
) -> None:
    """Run every check over `items`, raising with every violation found.

    `items` is consumed once — callers passing a generator should list() it first if they
    need it again afterwards. `fiche_chunk_texts` need only cover the fiche ids `items`
    actually reference; a missing id is treated as "no chunks", which the gold-fiches-
    resolve check already flags on its own.
    """
    materialized = list(items)
    violations: list[str] = [
        *_assert_unique_ids(materialized),
        *_assert_id_shape(materialized),
        *_assert_expected_state_known(materialized),
        *_assert_gold_fiches_resolve(materialized, fiche_ids),
        *_assert_gold_articles_resolve(materialized, article_cids),
        *_assert_gold_spans_verbatim(materialized, fiche_chunk_texts),
        *_assert_empty_cell_semantics(materialized),
        *_assert_expected_points_bounds(materialized),
    ]
    if violations:
        raise GoldenSetValidationError(f"{len(violations)} golden-set violation(s):\n" + "\n".join(violations))


def validate_golden_set_against_corpus(
    items: Iterable[GoldenItem],
    *,
    fiches_dir: Path,
    articles_path: Path,
) -> None:
    """Convenience wrapper: resolve the committed corpus (SPEC §12.1's "resolve in the
    committed corpus" requirement) and run `validate_golden_set` against it. Used by the
    CLI (`scripts/validate_golden_set.py`) and by the annotation helper's save path, which
    re-runs this before writing (#33's acceptance criterion) — one implementation, so the
    two can never disagree on what "valid" means.
    """
    materialized = list(items)
    referenced_fiches = {fid for item in materialized for fid in item.gold_fiches}
    chunk_texts = {
        fid: corpus_fiche_chunk_texts(fid, fiches_dir) for fid in referenced_fiches if (fiches_dir / f"{fid}.xml").exists()
    }
    articles = load_articles(articles_path)
    validate_golden_set(
        materialized,
        fiche_ids=corpus_fiche_ids(fiches_dir),
        article_cids=corpus_article_cids(articles),
        fiche_chunk_texts=chunk_texts,
    )


def _assert_unique_ids(items: list[GoldenItem]) -> list[str]:
    ids = [item.id for item in items]
    duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if not duplicates:
        return []
    return [f"{len(duplicates)} id(s) used more than once: {duplicates}"]


def _assert_id_shape(items: list[GoldenItem]) -> list[str]:
    """Stable, hand-assigned ids (SPEC §12.1) — `gs-<digits>`, checked for shape here;
    "never renumbered" is a process guarantee `ids.allocate_id` upholds, not something a
    stateless shape check on one snapshot of the file can see."""
    offenders = sorted({item.id for item in items if not re_match(_ID_PATTERN, item.id)})
    if not offenders:
        return []
    return [f"{len(offenders)} id(s) not shaped like gs-<digits>: {offenders}"]


def _assert_expected_state_known(items: list[GoldenItem]) -> list[str]:
    offenders = sorted({item.id for item in items if item.expected_state not in EXPECTED_STATES})
    if not offenders:
        return []
    return [f"{len(offenders)} item(s) with an expected_state outside {sorted(EXPECTED_STATES)}: {offenders}"]


def _assert_gold_fiches_resolve(items: list[GoldenItem], fiche_ids: set[str]) -> list[str]:
    offenders = sorted(
        {f"{item.id}:{fiche_id}" for item in items for fiche_id in item.gold_fiches if fiche_id not in fiche_ids}
    )
    if not offenders:
        return []
    return [f"{len(offenders)} gold_fiches entrie(s) not in the committed corpus: {offenders}"]


def _assert_gold_articles_resolve(items: list[GoldenItem], article_cids: set[str]) -> list[str]:
    offenders = sorted(
        {f"{item.id}:{cid}" for item in items for cid in item.gold_articles if cid not in article_cids}
    )
    if not offenders:
        return []
    return [f"{len(offenders)} gold_articles entrie(s) not in the committed corpus: {offenders}"]


def _assert_gold_spans_verbatim(items: list[GoldenItem], fiche_chunk_texts: Mapping[str, list[str]]) -> list[str]:
    """SPEC §12.2 — a `gold_span` must appear verbatim in some chunk of *this item's own*
    `gold_fiches` (not just anywhere in the corpus), so a span can never quietly point at a
    different fiche than the one it was annotated against."""
    offenders = []
    for item in items:
        candidate_texts = [text for fiche_id in item.gold_fiches for text in fiche_chunk_texts.get(fiche_id, [])]
        for span in item.gold_spans:
            if not any(span in text for text in candidate_texts):
                offenders.append(f"{item.id}:{span!r}")
    offenders.sort()
    if not offenders:
        return []
    return [f"{len(offenders)} gold_spans entrie(s) not verbatim in any gold_fiches chunk: {offenders}"]


def _assert_empty_cell_semantics(items: list[GoldenItem]) -> list[str]:
    violations = []
    for item in items:
        if item.expected_state == "refus:hors_corpus":
            if item.gold_fiches or item.gold_articles or item.gold_spans:
                violations.append(
                    f"{item.id}: refus:hors_corpus means nothing exists — gold_fiches, "
                    "gold_articles and gold_spans must all be empty"
                )
        elif item.expected_state == "reponse_sans_article":
            if item.gold_articles:
                violations.append(
                    f"{item.id}: reponse_sans_article has empty gold_articles by design — "
                    "the floor was not met, not that no fiche was found"
                )
            if not item.gold_fiches:
                violations.append(f"{item.id}: reponse_sans_article still needs a nonempty gold_fiches")
        else:
            if not item.gold_fiches:
                violations.append(f"{item.id}: {item.expected_state} needs a nonempty gold_fiches")
            if not item.gold_articles:
                violations.append(f"{item.id}: {item.expected_state} needs a nonempty gold_articles")
    return violations


def _assert_expected_points_bounds(items: list[GoldenItem]) -> list[str]:
    """SPEC §12.9 — 1-3 terse assertions per item, **except** `refus:hors_corpus`: "refusal
    items carry points too, ... `hors_corpus` items carry none" — there is nothing to assert
    a point about when nothing in the corpus answers the question."""
    offenders = sorted(item.id for item in items if not _points_within_bounds(item))
    if not offenders:
        return []
    return [f"{len(offenders)} item(s) with expected_points outside the SPEC §12.9 bounds: {offenders}"]


def _points_within_bounds(item: GoldenItem) -> bool:
    if item.expected_state == "refus:hors_corpus":
        return len(item.expected_points) == 0
    return _MIN_EXPECTED_POINTS <= len(item.expected_points) <= _MAX_EXPECTED_POINTS
