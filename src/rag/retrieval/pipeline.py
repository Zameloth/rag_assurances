"""Question -> `RetrievalResult`: path selection, the rung-1 arm, and the arm registry (SPEC §2, §9, §12.7, ADR-0015, #28).

This is the plain-Python orchestration `rag.query` and, later, the `BaseRetriever`
subclass call into — native `qdrant-client` reads and dataclasses throughout, no LangChain.
SPEC §6.1 reserves "retrieval stays a LangChain component" for the retriever wrapper itself;
what it wraps is exactly this module.

**The fat object.** SPEC §2 and §10.7 require the chain to return final contexts *and* the
per-leg candidate pools, because Langfuse evaluators see only the task's return value and
recall is measured at both candidate depth and final depth. `RetrievalResult` carries both
today, even though rung 1 has nothing to compute recall *against* yet (SPEC §12.7: rung 1 is
"reference floor, not a comparison") — the shape needs to exist before the eval harness does,
not be retrofitted once it does.

**Rung 1, precisely** (SPEC §12.7's ladder row, read the way ADR-0015 reconciles it against
ADR-0005's two collections): dense-only, no hybrid weighting (#29), no `<dc:source>`
expansion (#30), no rerank, no register quota (#33-ish, "quota vs free-for-all" is rung 5)
— a flat top-8 by raw dense score across both legs merged. The short-circuit (#27, SPEC
§9.1) sits in front of every rung, rung 1 included: it is not part of the ladder, it is
what decides whether the ladder's search path runs at all.

**Rung 4** (SPEC §9.4, #31) adds `rag.retrieval.rerank`'s cross-encoder over rung 3's exact
fused pool — still no LangChain, no Langfuse: the reranker is not a LangChain component, so
hand-wrapping its call in a Langfuse span belongs at the retriever boundary
(`langchain_retriever.py`), paired work with @Zameloth per the #31 issue comment, not this
module's job.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field

from qdrant_client import QdrantClient

from rag.ingest.upsert import EmbedFn
from rag.retrieval.candidates import Candidate, Register, merge_candidates
from rag.retrieval.expansion import EXPANSION_CAP, EXPANSION_FICHE_DEPTH, expand
from rag.retrieval.fusion import ARTICLE_LEG_WEIGHTS, FICHE_LEG_WEIGHTS, LegWeights, hybrid_leg
from rag.retrieval.legs import search_leg
from rag.retrieval.lookup import lookup_article_chunks_by_key
from rag.retrieval.rerank import DEFAULT_RERANKER_MODEL, RerankerBackend, RerankFn, rerank
from rag.retrieval.short_circuit import ShortCircuitPath, resolve_short_circuit

__all__ = [
    "ARTICLE_LEG",
    "DEFAULT_RETRIEVAL_ARM",
    "EXPANSION_POOL",
    "FICHE_LEG",
    "LEG_CANDIDATE_LIMIT",
    "RETRIEVAL_ARMS",
    "TOP_K",
    "RetrievalResult",
    "rank_candidates",
    "retrieve",
    "retrieve_rung1",
    "retrieve_rung2",
    "retrieve_rung3",
    "retrieve_rung4",
]

# SPEC §9.2 fixes both search legs at top-20 independently of which rung is active; rung 1
# doesn't hybridize or weight them (#29), but fetching at the architecture's own depth keeps
# the per-leg pools comparable across rungs instead of rung 1 measuring a smaller window.
LEG_CANDIDATE_LIMIT = 20

# SPEC §12.7's rung-1 row: naive top-8, pre-quota, pre-rerank. This is also rung 5's
# "free-for-all" incumbent-to-beat (SPEC §12.7 row 5), so the name is deliberately not
# "rung1-only" even though only rung 1 exists today.
TOP_K = 8

FICHE_LEG = "fiche_leg"
ARTICLE_LEG = "article_leg"
EXPANSION_POOL = "expansion"


@dataclass(frozen=True)
class RetrievalResult:
    """The fat object (SPEC §2, §10.7): `contexts` is what generation would see, the
    per-leg pools are what the eval harness needs at candidate depth. `candidate_pools` is
    empty on the short-circuit path — SPEC §9.1: it skips search entirely, so there is no
    pool to report."""

    short_circuit_path: ShortCircuitPath
    contexts: list[Candidate]
    candidate_pools: dict[str, list[Candidate]] = field(default_factory=dict)


def rank_candidates(candidates: list[Candidate], *, top_k: int = TOP_K) -> list[Candidate]:
    """SPEC §12.7's rung-1 arm: sort by raw score descending, cap at `top_k`.

    Pure and Qdrant-free on purpose — the merged pool's *ordering* is this function's own
    logic, not the store's, so it is exercised directly against synthetic scores rather
    than only indirectly through a live query's cosine numbers (`QdrantClient(":memory:")`
    is for plumbing assertions, never ranking ones — SPEC §6.3, `tests/conftest.py`).
    """
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[:top_k]


def retrieve_rung1(
    client: QdrantClient,
    embed: EmbedFn,
    raw_turn: str,
    lookup_keys: AbstractSet[str],
) -> RetrievalResult:
    """SPEC §9.1's three paths, then SPEC §12.7's rung-1 arm on the fall-through paths.

    `embed` is the ingest-side `EmbedFn` shape (`Sequence[str] -> list[Embedding]`); only
    the dense half of its single BGE-M3 forward pass is used here — the sparse half is
    computed and discarded, which is the accepted cost of reusing one embed call rather
    than adding a dense-only code path that would diverge from ingest's.
    """
    result = resolve_short_circuit(raw_turn, frozenset(lookup_keys))
    if result.path is ShortCircuitPath.RESOLVED:
        assert result.lookup_key is not None  # RESOLVED always carries its key
        contexts = lookup_article_chunks_by_key(client, result.lookup_key)
        return RetrievalResult(
            short_circuit_path=result.path, contexts=contexts, candidate_pools={}
        )

    dense_vector, _sparse = embed([raw_turn])[0]
    fiche_pool = search_leg(client, Register.FICHE, dense_vector, LEG_CANDIDATE_LIMIT)
    article_pool = search_leg(client, Register.ARTICLE, dense_vector, LEG_CANDIDATE_LIMIT)

    merged = merge_candidates(fiche_pool, article_pool)
    contexts = rank_candidates(merged)

    return RetrievalResult(
        short_circuit_path=result.path,
        contexts=contexts,
        candidate_pools={FICHE_LEG: fiche_pool, ARTICLE_LEG: article_pool},
    )


def retrieve_rung2(
    client: QdrantClient,
    embed: EmbedFn,
    raw_turn: str,
    lookup_keys: AbstractSet[str],
    *,
    fiche_weights: LegWeights = FICHE_LEG_WEIGHTS,
    article_weights: LegWeights = ARTICLE_LEG_WEIGHTS,
) -> RetrievalResult:
    """SPEC §9.1's three paths, then SPEC §9.3's rung-2 arm on the fall-through paths: both
    legs go hybrid, fused client-side with per-leg weights (#29) — dense-leaning for the
    fiche leg, sparse-leaning for the article leg, per the M3 paper's MIRACL/MLDR regime
    inversion. Short-circuit, merge and rank are `retrieve_rung1`'s own calls, unchanged;
    only how each leg's pool gets its score differs.

    `fiche_weights`/`article_weights` default to the named constants but are real
    parameters, not just internal ones `hybrid_leg` happens to take — the eval ladder
    tunes these (ADR-0016), so the override has to reach this far up, the same way
    `RETRIEVAL_ARMS`/`arm` lets a caller override which rung runs at all.
    """
    result = resolve_short_circuit(raw_turn, frozenset(lookup_keys))
    if result.path is ShortCircuitPath.RESOLVED:
        assert result.lookup_key is not None  # RESOLVED always carries its key
        contexts = lookup_article_chunks_by_key(client, result.lookup_key)
        return RetrievalResult(
            short_circuit_path=result.path, contexts=contexts, candidate_pools={}
        )

    dense_vector, sparse_vector = embed([raw_turn])[0]
    fiche_pool = hybrid_leg(
        client,
        Register.FICHE,
        dense_vector,
        sparse_vector,
        fiche_weights,
        limit=LEG_CANDIDATE_LIMIT,
    )
    article_pool = hybrid_leg(
        client,
        Register.ARTICLE,
        dense_vector,
        sparse_vector,
        article_weights,
        limit=LEG_CANDIDATE_LIMIT,
    )

    merged = merge_candidates(fiche_pool, article_pool)
    contexts = rank_candidates(merged)

    return RetrievalResult(
        short_circuit_path=result.path,
        contexts=contexts,
        candidate_pools={FICHE_LEG: fiche_pool, ARTICLE_LEG: article_pool},
    )


def retrieve_rung3(
    client: QdrantClient,
    embed: EmbedFn,
    raw_turn: str,
    lookup_keys: AbstractSet[str],
    *,
    fiche_weights: LegWeights = FICHE_LEG_WEIGHTS,
    article_weights: LegWeights = ARTICLE_LEG_WEIGHTS,
    expansion_depth: int = EXPANSION_FICHE_DEPTH,
    expansion_cap: int = EXPANSION_CAP,
) -> RetrievalResult:
    """SPEC §9.1's three paths, then SPEC §9.2's rung-3 arm on the fall-through paths: the
    top `expansion_depth` fiches from the (already-hybrid) fiche pool expand into the
    `LEGISCTA` sections they cite, and a filtered vector search within those sections joins
    the fused pool (#30, the headline experiment — ADR-0006). Short-circuit, both legs, merge
    and rank are `retrieve_rung2`'s own calls, unchanged; expansion is a third, additional
    pool alongside them, not a replacement for either leg.

    `expansion_depth`/`expansion_cap` default to the named SPEC constants but are real
    parameters, the same shape `fiche_weights`/`article_weights` already take (#29) — this
    ticket's own acceptance criterion is that the arm stays ablatable through them.
    """
    result = resolve_short_circuit(raw_turn, frozenset(lookup_keys))
    if result.path is ShortCircuitPath.RESOLVED:
        assert result.lookup_key is not None  # RESOLVED always carries its key
        contexts = lookup_article_chunks_by_key(client, result.lookup_key)
        return RetrievalResult(
            short_circuit_path=result.path, contexts=contexts, candidate_pools={}
        )

    dense_vector, sparse_vector = embed([raw_turn])[0]
    fiche_pool = hybrid_leg(
        client,
        Register.FICHE,
        dense_vector,
        sparse_vector,
        fiche_weights,
        limit=LEG_CANDIDATE_LIMIT,
    )
    article_pool = hybrid_leg(
        client,
        Register.ARTICLE,
        dense_vector,
        sparse_vector,
        article_weights,
        limit=LEG_CANDIDATE_LIMIT,
    )
    expansion_pool = expand(
        client, fiche_pool, dense_vector, depth=expansion_depth, cap=expansion_cap
    )

    merged = merge_candidates(fiche_pool, article_pool, expansion_pool)
    contexts = rank_candidates(merged)

    return RetrievalResult(
        short_circuit_path=result.path,
        contexts=contexts,
        candidate_pools={
            FICHE_LEG: fiche_pool,
            ARTICLE_LEG: article_pool,
            EXPANSION_POOL: expansion_pool,
        },
    )


def retrieve_rung4(
    client: QdrantClient,
    embed: EmbedFn,
    raw_turn: str,
    lookup_keys: AbstractSet[str],
    *,
    fiche_weights: LegWeights = FICHE_LEG_WEIGHTS,
    article_weights: LegWeights = ARTICLE_LEG_WEIGHTS,
    expansion_depth: int = EXPANSION_FICHE_DEPTH,
    expansion_cap: int = EXPANSION_CAP,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    reranker_backend: RerankerBackend = RerankerBackend.FP32,
    rerank_fn: RerankFn | None = None,
) -> RetrievalResult:
    """SPEC §9.1's three paths, then SPEC §9.4's rung-4 arm on the fall-through paths: the
    same fused pool `retrieve_rung3` builds (both hybrid legs + `<dc:source>` expansion) is
    rescored by a cross-encoder before the top-K cap, every candidate's score replaced
    outright (ADR-0017 already named unifying the merged pool's scale across three
    incomparable sources "rung 4's problem"). Short-circuit, both legs, expansion and merge
    are `retrieve_rung3`'s own calls, unchanged; rerank is a fourth, final step over the
    already-merged pool, not a fifth candidate pool — `RetrievalResult.candidate_pools`
    keeps rung 3's exact shape, since reranking has nothing to do with which leg or
    expansion found a candidate, only how the whole set is ordered.

    `reranker_model`/`reranker_backend` default to the SPEC §9.4 primary arm but are real
    parameters — this ticket's own acceptance criterion is that the cheap arm and int8 ONNX
    quantisation stay reachable without a code change (SPEC §14.4's deploy-time RAM rule),
    the same shape `expansion_depth`/`fiche_weights` already take.

    `rerank_fn` defaults to `None`, resolved below to the real `rerank()` bound to
    `reranker_model`/`reranker_backend` — tests inject a fake instead (mirrors the `EmbedFn`
    seam), in which case `reranker_model`/`reranker_backend` go unused, since the injected
    function owns scoring entirely.
    """
    result = resolve_short_circuit(raw_turn, frozenset(lookup_keys))
    if result.path is ShortCircuitPath.RESOLVED:
        assert result.lookup_key is not None  # RESOLVED always carries its key
        contexts = lookup_article_chunks_by_key(client, result.lookup_key)
        return RetrievalResult(
            short_circuit_path=result.path, contexts=contexts, candidate_pools={}
        )

    dense_vector, sparse_vector = embed([raw_turn])[0]
    fiche_pool = hybrid_leg(
        client,
        Register.FICHE,
        dense_vector,
        sparse_vector,
        fiche_weights,
        limit=LEG_CANDIDATE_LIMIT,
    )
    article_pool = hybrid_leg(
        client,
        Register.ARTICLE,
        dense_vector,
        sparse_vector,
        article_weights,
        limit=LEG_CANDIDATE_LIMIT,
    )
    expansion_pool = expand(
        client, fiche_pool, dense_vector, depth=expansion_depth, cap=expansion_cap
    )

    merged = merge_candidates(fiche_pool, article_pool, expansion_pool)
    scorer = rerank_fn or (
        lambda q, c: rerank(q, c, model_id=reranker_model, backend=reranker_backend)
    )
    contexts = rank_candidates(scorer(raw_turn, merged))

    return RetrievalResult(
        short_circuit_path=result.path,
        contexts=contexts,
        candidate_pools={
            FICHE_LEG: fiche_pool,
            ARTICLE_LEG: article_pool,
            EXPANSION_POOL: expansion_pool,
        },
    )


RetrieveFn = Callable[[QdrantClient, EmbedFn, str, AbstractSet[str]], RetrievalResult]

# The registry `rag.query` (and later the retriever wrapper) select from — this ticket's
# own acceptance criterion: "the arm is selectable by config, so rung 1 stays runnable
# after later rungs land." ADR-0015 records why this is a named constant plus an
# injectable parameter (mirroring `rag.ingest.pipeline`'s `ARTICLES_ARM`/`FICHES_ARM`)
# rather than a `RETRIEVAL_ARM` environment variable. #29 adds "rung2" as the same kind of
# addition ADR-0015 anticipated: "no `Settings`/`.env.example` change is implied by that
# addition." `DEFAULT_RETRIEVAL_ARM` stays "rung1" — landing an arm is not the same as the
# ladder adopting it (SPEC §12.7's pre-registered rule, "inconclusive keeps the incumbent").
RETRIEVAL_ARMS: dict[str, RetrieveFn] = {
    "rung1": retrieve_rung1,
    "rung2": retrieve_rung2,
    "rung3": retrieve_rung3,
    "rung4": retrieve_rung4,
}
DEFAULT_RETRIEVAL_ARM = "rung1"


def retrieve(
    client: QdrantClient,
    embed: EmbedFn,
    raw_turn: str,
    lookup_keys: AbstractSet[str],
    *,
    arm: str = DEFAULT_RETRIEVAL_ARM,
) -> RetrievalResult:
    """Dispatch to the named arm in `RETRIEVAL_ARMS`. `KeyError` on an unknown name — the
    same "fail rather than silently resolve to a default" posture `rag.config` takes on a
    misspelled model id."""
    try:
        arm_fn = RETRIEVAL_ARMS[arm]
    except KeyError:
        known = ", ".join(sorted(RETRIEVAL_ARMS))
        raise KeyError(f"unknown retrieval arm {arm!r}; known arms: {known}") from None
    return arm_fn(client, embed, raw_turn, lookup_keys)
