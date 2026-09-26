"""Generation eval's three deterministic metrics — SPEC §12.9, ADR-0009, #46.

Pure functions over an already-materialized `GenerationResult` (`rag.generation.pipeline`):
no Qdrant, no Langfuse, no OpenRouter. Mirrors `rag.eval.retrieval_metrics`'s own split
between plain scoring math and the (Langfuse-specific) harness that calls into it —
`score_item` is the seam `rag.eval.run_generation_experiment`'s evaluator calls into.

**Three of five cost nothing and cannot drift** (SPEC §12.9): the refusal contract, the four
terminal states and the fabrication guardrail are all measured by `==` and set containment,
so judge unreliability (the two judged metrics, a later ticket) threatens two numbers rather
than the whole eval. `citation_valid` is read straight off `GenerationResult.citation_outcome`
— `rag.generation.pipeline.generate` already ran `rag.generation.citation.check_citations`
against the typed `fondement_juridique` field, so this module never regexes prose and never
recomputes a containment check `generate()` already did.

**Citation correctness needs one join `citation_valid` never does.** `fondement_juridique.
article_id` — and so `CitationOutcome.cited_ids` — is a `citation_id` (SPEC §10.6: "copy its
identifier exactly as it appears... never construct one"), while `gold_articles` is keyed by
`cid` (CONTEXT.md: "`cid` is identity everywhere — point ids, gold labels, joins"). Two
different identifier spaces for the same article, 52% of which diverge (CONTEXT.md). `
resolve_cited_cids` closes that gap through the retrieved contexts — the only place, at eval
time, that carries both ids for the same candidate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag.eval.schema import GoldenItem
from rag.generation.pipeline import GenerationResult
from rag.generation.schema import Refus
from rag.retrieval.candidates import Candidate, Register
from rag.retrieval.quota import NO_ARTICLE_MARKER_ID

__all__ = ["ItemGenerationScore", "citation_correctness", "resolve_cited_cids", "score_item", "state_label"]


def state_label(result: GenerationResult) -> str:
    """The envelope's own terminal state, field for field — `type` + `motif` +
    `aucun_fondement` collapsed into one of `rag.eval.schema.EXPECTED_STATES`'s five
    spellings (SPEC §12.1: "expected_state mirrors the generation envelope field for field",
    the acceptance criterion this function exists to satisfy: no mapping through an
    intermediate vocabulary, just the three envelope fields read directly).
    """
    envelope = result.envelope
    if isinstance(envelope, Refus):
        return f"refus:{envelope.motif.value}"
    return "reponse_sans_article" if envelope.aucun_fondement is not None else "reponse"


def resolve_cited_cids(cited_ids: frozenset[str], contexts: Sequence[Candidate]) -> frozenset[str]:
    """`cited_ids` (`citation_id`-space) -> the `cid`s of the retrieved articles they name.

    An article cited by an id that names nothing in `contexts` resolves to nothing — either
    it was fabricated (already flagged by `citation_valid`, SPEC §10.5's guardrail) or it is
    the no-article marker, which carries no `citation_id` to match in the first place
    (`rag.generation.citation.retrieved_citation_ids` excludes it the same way).
    """
    return frozenset(
        str(candidate.payload["legiarti_cid"])
        for candidate in contexts
        if candidate.register is Register.ARTICLE
        and candidate.id != NO_ARTICLE_MARKER_ID
        and str(candidate.payload.get("citation_id")) in cited_ids
    )


def citation_correctness(cited_cids: frozenset[str], gold_articles: frozenset[str]) -> float | None:
    """Cited `cid`s vs `gold_articles` (SPEC §12.9) — the same recall shape
    `rag.eval.retrieval_metrics._recall` uses, kept a distinct number on purpose: article
    recall@4 asks whether the gold article *reached the prompt*, this asks whether the model
    *cited it*. Same "undefined, not zero" posture recall itself takes: `None` on an empty
    gold set — a `hors_corpus`/`reponse_sans_article` item has nothing here to score, not a
    vacuous 0 or 1. Takes `cited_cids` already resolved by `resolve_cited_cids` — this
    function is pure set arithmetic and does not know about `citation_id`s at all.
    """
    if not gold_articles:
        return None
    return len(cited_cids & gold_articles) / len(gold_articles)


@dataclass(frozen=True)
class ItemGenerationScore:
    """One golden-set item's row of SPEC §12.9's table, plus the golden id and the actual
    state reached — everything `eval/runs/<run-id>.json` needs per item for the generation
    regime (SPEC §12.11).

    `score_item` fills the three deterministic metrics; `faithfulness`/`point_coverage` are
    the judged two (#47), filled by `rag.eval.run_generation_experiment` from the judge's
    evaluations and `None` when no judge ran — or, for `point_coverage`, when the item has
    no `expected_points` to cover (`hors_corpus`, SPEC §12.9).
    """

    item_id: str
    expected_state: str
    actual_state: str
    state_correct: bool
    citation_valid: bool
    fabricated_ids: tuple[str, ...]
    citation_correctness: float | None
    faithfulness: float | None = None
    point_coverage: float | None = None


def score_item(item: GoldenItem, result: GenerationResult) -> ItemGenerationScore:
    """The three deterministic metrics SPEC §12.9 names, computed for one item against one
    `GenerationResult`. Always `generate()`'s unrepaired output (SPEC §10.5: "eval always
    sees the model's own, unrepaired output") — never
    `rag.generation.citation.repair_for_display`'s demo-path copy, or a fabricated citation
    that should fail `citation_valid` would instead disappear before scoring.
    """
    actual_state = state_label(result)
    outcome = result.citation_outcome
    cited_cids = resolve_cited_cids(outcome.cited_ids, result.contexts)
    return ItemGenerationScore(
        item_id=item.id,
        expected_state=item.expected_state,
        actual_state=actual_state,
        state_correct=actual_state == item.expected_state,
        citation_valid=outcome.valid,
        fabricated_ids=tuple(sorted(outcome.fabricated_ids)),
        citation_correctness=citation_correctness(cited_cids, frozenset(item.gold_articles)),
    )
