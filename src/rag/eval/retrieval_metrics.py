"""The retrieval eval harness's own math — SPEC §12.6, ADR-0011, #35.

Pure functions over already-materialized `Candidate`/`RetrievalResult` objects (`rag.retrieval`):
no Qdrant, no Langfuse, nothing async. `score_item` is the seam the (Langfuse-specific)
experiment harness calls into — its evaluators translate one `ItemRetrievalScore` into
whatever shape `dataset.run_experiment()` wants, but the scoring itself owes nothing to how
retrieval was invoked.

**Nine numbers, not eight** — SPEC §12.6's table itemizes fiche/article recall@4,
zero-article rate, floor correctness (the four decision metrics, matching ADR-0011's "four
of which decide"), fiche/article recall@10, fiche/article recall@candidate, and span
containment@4: nine named cells. The ADR's own prose calls this "eight retrieval numbers"
by counting the fiche/article recall *pairs* as one line each; every cell in the table is
still computed here, so nothing named in SPEC §12.6 is missing.

**Depth-then-dedupe, not dedupe-then-depth.** `ranked_ids` drops duplicate chunks of the
same document *before* a `k` cut is applied, so "recall@4" means "4 distinct documents",
matching the quota's own unit (SPEC §7.6: 4 fiche + 4 article slots) rather than 4 chunks
that might collapse to fewer documents.

**`recall_at_candidate` reads the raw pools, not `contexts`** — "never found" (absent from
the pool the leg/expansion ever produced) is a different failure from "found then dropped"
(present in the pool, cut before the final contexts), and `RetrievalResult.candidate_pools`
is exactly the fat-object field SPEC §2/§10.7 keeps around to answer that question. It
returns `None` on the short-circuit path (`candidate_pools == {}`, SPEC §9.1): there is no
pool to diagnose when search never ran, which is a different fact from having searched and
found nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from rag.eval.schema import GoldenItem
from rag.retrieval.candidates import Candidate, Register, merge_candidates
from rag.retrieval.pipeline import ARTICLE_LEG, EXPANSION_POOL, FICHE_LEG, RetrievalResult
from rag.retrieval.quota import NO_ARTICLE_MARKER_ID

__all__ = [
    "ARTICLE_DIAGNOSTIC_DEPTH",
    "ARTICLE_FINAL_DEPTH",
    "FICHE_DIAGNOSTIC_DEPTH",
    "FICHE_FINAL_DEPTH",
    "SPAN_CONTAINMENT_DEPTH",
    "ItemRetrievalScore",
    "article_recall_working_set",
    "floor_correctness_working_set",
    "ranked_ids",
    "score_item",
    "working_set",
]

# SPEC §12.6's own depths — named here rather than left as bare literals in `score_item`,
# since the pre-registered table (ADR-0011) fixes these before any rung runs, the same
# reason `rag.retrieval.pipeline.TOP_K` is a constant rather than an inline `8`.
FICHE_FINAL_DEPTH = 4
ARTICLE_FINAL_DEPTH = 4
FICHE_DIAGNOSTIC_DEPTH = 10
ARTICLE_DIAGNOSTIC_DEPTH = 10
SPAN_CONTAINMENT_DEPTH = 4

_NATURAL_ID_FIELD = {Register.FICHE: "fiche_id", Register.ARTICLE: "legiarti_cid"}


def working_set(items: Sequence[GoldenItem]) -> list[GoldenItem]:
    """CONTEXT.md's working set: single-turn items carrying gold contexts.

    Multi-turn items are excluded — the ladder runs the `history == []` subset only (SPEC
    §12.1: a condenser call in front of six rungs would tax every run with a component not
    under test). `hors_corpus` items are excluded too: empty `gold_fiches` *and* empty
    `gold_articles` means "nothing exists" (SPEC §12.4), not a retrieval target — recall of
    an empty gold set is undefined, not zero, so there is nothing here to score.
    """
    return [item for item in items if not item.history and (item.gold_fiches or item.gold_articles)]


def article_recall_working_set(items: Sequence[GoldenItem]) -> list[GoldenItem]:
    """SPEC §12.6: article-recall metrics run on the working-set items with non-empty
    `gold_articles` — `reponse_sans_article` items are excluded here on purpose, since
    their empty `gold_articles` *is* the correct answer (scored instead by
    `floor_correctness_working_set`), not a recall target."""
    return [item for item in working_set(items) if item.gold_articles]


def floor_correctness_working_set(items: Sequence[GoldenItem]) -> list[GoldenItem]:
    """The `reponse_sans_article` items (SPEC §12.6's floor-correctness row), restricted to
    the working set — the ladder's retrieval dataset is single-turn only (SPEC §12.5), so a
    multi-turn `reponse_sans_article` item belongs to the generation dataset, not here."""
    return [item for item in working_set(items) if item.expected_state == "reponse_sans_article"]


def ranked_ids(candidates: Sequence[Candidate], register: Register) -> list[str]:
    """The distinct natural ids (`fiche_id` / `legiarti_cid`) of `register`'s candidates, in
    first-occurrence rank order — recall is measured in documents, not chunks (ADR-0010:
    gold labels are document-level), so a document repeated across several chunks must not
    inflate a `k`-deep cut.

    Skips `rag.retrieval.quota`'s no-article marker (#41): it carries `register is
    Register.ARTICLE` so the quota's floor-not-met outcome still occupies the article slot
    shape `RetrievalResult.contexts` expects, but it names no real document — its payload
    has no `legiarti_cid` to key on, and counting it here would both crash and, if patched
    over some other way, silently turn "zero articles cleared the floor" into "one article
    found."""
    field = _NATURAL_ID_FIELD[register]
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        if candidate.register is not register or candidate.id == NO_ARTICLE_MARKER_ID:
            continue
        natural_id = str(candidate.payload[field])
        if natural_id not in seen:
            seen.add(natural_id)
            ordered.append(natural_id)
    return ordered


def _recall(present: Iterable[str], gold: frozenset[str]) -> float | None:
    """`None` on an empty gold set — recall is undefined there, not vacuously 1 or 0 (SPEC
    §12.6), which is exactly why `reponse_sans_article` items are scored by floor
    correctness instead of by this function."""
    if not gold:
        return None
    return len(set(present) & gold) / len(gold)


def _recall_at_k(candidates: Sequence[Candidate], register: Register, gold: frozenset[str], k: int) -> float | None:
    return _recall(ranked_ids(candidates, register)[:k], gold)


def _recall_at_candidate(
    candidate_pools: Mapping[str, list[Candidate]], register: Register, gold: frozenset[str]
) -> float | None:
    if not candidate_pools:
        return None
    if register is Register.FICHE:
        pool: list[Candidate] = candidate_pools.get(FICHE_LEG, [])
    else:
        pool = merge_candidates(
            candidate_pools.get(ARTICLE_LEG, []), candidate_pools.get(EXPANSION_POOL, [])
        )
    return _recall(ranked_ids(pool, register), gold)


def _zero_articles(contexts: Sequence[Candidate]) -> bool:
    """True iff no *real* article is present — the no-article marker (#41, rung 5's own
    floor-not-met outcome) is itself `register is Register.ARTICLE`, so it has to be
    excluded explicitly here or this would read "zero articles" as false on the exact rung
    whose primary metric (SPEC §12.7's zero-article rate) this function exists to feed."""
    return not any(
        candidate.register is Register.ARTICLE and candidate.id != NO_ARTICLE_MARKER_ID
        for candidate in contexts
    )


def _floor_correct(contexts: Sequence[Candidate], expected_state: str) -> bool | None:
    if expected_state != "reponse_sans_article":
        return None
    return _zero_articles(contexts)


def _span_containment_at_k(contexts: Sequence[Candidate], gold_spans: Sequence[str], k: int) -> float | None:
    if not gold_spans:
        return None
    fiche_texts = [str(c.payload.get("text", "")) for c in contexts if c.register is Register.FICHE][:k]
    hits = sum(1 for span in gold_spans if any(span in text for text in fiche_texts))
    return hits / len(gold_spans)


@dataclass(frozen=True)
class ItemRetrievalScore:
    """One golden-set item's row of SPEC §12.6's table, plus the golden id and the
    short-circuit path it took — everything `eval/runs/<run-id>.json` needs per item
    (SPEC §12.11)."""

    item_id: str
    short_circuit_path: str
    fiche_recall_at_4: float | None
    fiche_recall_at_10: float | None
    fiche_recall_at_candidate: float | None
    article_recall_at_4: float | None
    article_recall_at_10: float | None
    article_recall_at_candidate: float | None
    zero_articles: bool | None
    floor_correct: bool | None
    span_containment_at_4: float | None


def score_item(item: GoldenItem, result: RetrievalResult) -> ItemRetrievalScore:
    """Every metric SPEC §12.6 names, computed for one item against one `RetrievalResult`.

    A metric reads `None` wherever it is not applicable to this item (empty gold, off the
    `reponse_sans_article` subset, or the short-circuit's empty candidate pool) rather than
    a placeholder zero — `_recall`'s docstring is the reason this matters for the recall
    family, and the same "undefined, not zero" posture is kept for `zero_articles` and
    `floor_correct` so an aggregate over the wrong subset fails loudly instead of quietly
    diluting toward zero.
    """
    fiche_gold = frozenset(item.gold_fiches)
    article_gold = frozenset(item.gold_articles)
    contexts = result.contexts
    return ItemRetrievalScore(
        item_id=item.id,
        short_circuit_path=result.short_circuit_path.value,
        fiche_recall_at_4=_recall_at_k(contexts, Register.FICHE, fiche_gold, FICHE_FINAL_DEPTH),
        fiche_recall_at_10=_recall_at_k(contexts, Register.FICHE, fiche_gold, FICHE_DIAGNOSTIC_DEPTH),
        fiche_recall_at_candidate=_recall_at_candidate(result.candidate_pools, Register.FICHE, fiche_gold),
        article_recall_at_4=_recall_at_k(contexts, Register.ARTICLE, article_gold, ARTICLE_FINAL_DEPTH),
        article_recall_at_10=_recall_at_k(contexts, Register.ARTICLE, article_gold, ARTICLE_DIAGNOSTIC_DEPTH),
        article_recall_at_candidate=_recall_at_candidate(result.candidate_pools, Register.ARTICLE, article_gold),
        zero_articles=_zero_articles(contexts) if article_gold else None,
        floor_correct=_floor_correct(contexts, item.expected_state),
        span_containment_at_4=_span_containment_at_k(contexts, item.gold_spans, SPAN_CONTAINMENT_DEPTH),
    )
