"""Register quota with a relevance floor — the rung-5 arm (SPEC §9.5, ADR-0019, #32).

The failure this designs out: a cross-encoder ranks by relevance to the query alone, and the
query is consumer French. Naive top-8 can return eight fiche chunks and zero articles —
leaving the model nothing to cite, intermittently, on the system's headline feature (SPEC
§9.5). `assemble_quota` fills two slots — 4 fiches, 4 articles — separately, over the
already-reranked merged pool (rung 4's own output, SPEC §9.4), rather than one shared top-8.

Two rules apply to article slots only, never to fiche slots (SPEC §9.5 names neither a
preference nor a floor for fiches):

- **Expansion-sourced articles are preferred over search-sourced ones, within a margin** — an
  expansion-sourced candidate outranks a search-sourced one scoring up to
  `expansion_preference_margin` higher, because DILA's editorial join carries different
  relevance semantics than a cosine score (SPEC §9.5): "not equivalent," so a close call
  between the two isn't decided by the cosine number alone. A search-sourced candidate whose
  lead exceeds the margin still wins outright — the preference breaks ties among comparably
  relevant candidates, it does not let a barely-floor-clearing expansion hit bump a clearly
  stronger search hit out of a slot entirely.
- **A relevance floor rejects rather than pads.** An article scoring under `relevance_floor`
  never fills a slot; fewer than 4 articles is a valid outcome (SPEC §9.5, this ticket's own
  acceptance criterion). Only when *zero* articles clear the floor does the explicit
  no-article marker enter the context — partial fill (1-3 articles) is not the floor-not-met
  case, it is the quota simply returning what cleared it.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.retrieval.candidates import Candidate, Provenance, Register

__all__ = [
    "ARTICLE_QUOTA",
    "ARTICLE_RELEVANCE_FLOOR",
    "EXPANSION_PREFERENCE_MARGIN",
    "FICHE_QUOTA",
    "NO_ARTICLE_MARKER_ID",
    "NO_ARTICLE_MARKER_TEXT",
    "QuotaResult",
    "assemble_quota",
]

# SPEC §9.5: "4 fiche chunks + 4 articles."
FICHE_QUOTA = 4
ARTICLE_QUOTA = 4

# The cross-encoder's `compute_score(..., normalize=True)` (`rag.retrieval.rerank`) is a
# sigmoid, so every candidate arrives scored as P(relevant) in [0, 1] — 0.5 is that
# classifier's own decision boundary, not a value tuned against this corpus. SPEC §12.7
# pre-registers "floor correctness" (the 8 `reponse_sans_article` golden items) as a
# first-class decision metric precisely because this number is a starting point for the
# ladder to calibrate, the same posture ADR-0016 already took for the fusion weights.
ARTICLE_RELEVANCE_FLOOR = 0.5

# How much of a cross-encoder-score lead a search-sourced article needs over an
# expansion-sourced one before the search-sourced one wins outright (SPEC §9.5's provenance
# preference, ADR-0019): 0.1 of the same [0, 1] sigmoid scale as the floor — a starting
# point for the ladder to calibrate, not a measured value, the same posture as the floor
# itself. Within this margin, the two scores are treated as not meaningfully distinguishable
# and the editorial (expansion) provenance breaks the tie.
EXPANSION_PREFERENCE_MARGIN = 0.1

# Sentinel id/text for the no-article marker (SPEC §9.5, §10.2's `aucun_fondement`, CONTEXT.md's
# "the no-article marker" entry): synthesized here, never a real point id, so it can never
# collide with one and never carries a citable `citation_id` in its payload.
NO_ARTICLE_MARKER_ID = "no-article-marker"
NO_ARTICLE_MARKER_TEXT = "Pas de fondement juridique direct dans le corpus."


@dataclass(frozen=True)
class QuotaResult:
    """`contexts` is fiche slots followed by article slots (or the marker) — SPEC §10.6's
    prompt grouping reads this order directly. `floor_met` is the fat object's own field
    (SPEC §9.5, §10.7's acceptance criterion): `False` exactly when the marker replaced the
    article slots, `True` otherwise (including the 1-3-articles partial-fill case)."""

    contexts: list[Candidate]
    floor_met: bool


def assemble_quota(
    candidates: list[Candidate],
    *,
    fiche_quota: int = FICHE_QUOTA,
    article_quota: int = ARTICLE_QUOTA,
    relevance_floor: float = ARTICLE_RELEVANCE_FLOOR,
    expansion_preference_margin: float = EXPANSION_PREFERENCE_MARGIN,
) -> QuotaResult:
    """Fill `fiche_quota` fiche slots and `article_quota` article slots separately from the
    reranked merged pool (SPEC §9.5).

    Sorts by score descending itself rather than trusting incoming order — the same
    defensive posture `rank_candidates` already takes (`rag.retrieval.pipeline`), so this
    function is exercised directly against synthetic scores, not only indirectly through a
    live rerank call.

    Fiche slots are a plain top-`fiche_quota` cap, no floor, no preference — SPEC §9.5 names
    both rules for article slots only. Article slots are filled from whichever candidates
    clear `relevance_floor`, ordered by score with an `expansion_preference_margin` bonus
    added to expansion-sourced candidates before that ordering — so an expansion-sourced
    candidate only overtakes a search-sourced one scoring within the margin, never one with
    a bigger lead (SPEC §9.5's provenance preference, read as a tiebreak, not a hard
    partition — ADR-0019). **Rejected, not padded**, so the article side can return fewer
    than `article_quota` candidates. Only when none clear the floor does the no-article
    marker take the article side's place.
    """
    ranked = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

    fiches = [c for c in ranked if c.register is Register.FICHE][:fiche_quota]

    article_pool = [c for c in ranked if c.register is Register.ARTICLE]
    above_floor = [c for c in article_pool if c.score >= relevance_floor]

    def _preference_score(candidate: Candidate) -> float:
        bonus = expansion_preference_margin if Provenance.EXPANSION in candidate.provenance else 0.0
        return candidate.score + bonus

    ordered = sorted(above_floor, key=_preference_score, reverse=True)
    articles = ordered[:article_quota]

    floor_met = len(articles) > 0
    if not floor_met:
        articles = [_no_article_marker()]

    return QuotaResult(contexts=fiches + articles, floor_met=floor_met)


def _no_article_marker() -> Candidate:
    """The explicit no-article marker (SPEC §9.5): a synthetic `Candidate` so it flows
    through `RetrievalResult.contexts`'s existing shape unchanged, rather than growing a
    second, parallel return field the prompt-assembly step (§10.6, a later ticket) would
    have to check in addition to the contexts list. `provenance` is the empty set — it was
    reached by no path, unlike `Provenance.LOOKUP`'s own "found, just not by search"."""
    return Candidate(
        id=NO_ARTICLE_MARKER_ID,
        score=0.0,
        register=Register.ARTICLE,
        payload={"text": NO_ARTICLE_MARKER_TEXT},
        provenance=frozenset(),
    )
