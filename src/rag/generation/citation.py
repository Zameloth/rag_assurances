"""The citation guardrail — `cited ⊆ retrieved_context`, never `⊆ corpus` (SPEC §10.5,
ADR-0009, #42).

An article that exists in the corpus and was never retrieved is still fabrication — the
model produced it from parametric memory, not from the pipeline — and a corpus-wide lookup
would wave it through. `check_citations` reads the envelope's typed `fondement_juridique`
field (SPEC §10.2's whole point: a guardrail that loops over a typed field instead of
regexing prose), and never mutates it — recording the outcome and repairing the display copy
are two different jobs, so eval always sees what the model actually returned.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.generation.schema import Envelope, Reponse
from rag.retrieval.candidates import Candidate, Register
from rag.retrieval.quota import NO_ARTICLE_MARKER_ID, NO_ARTICLE_MARKER_TEXT

__all__ = ["CitationOutcome", "check_citations", "repair_for_display", "retrieved_citation_ids"]


@dataclass(frozen=True)
class CitationOutcome:
    """The per-turn guardrail verdict (SPEC §10.5, §10.7's "citation-check outcome").
    `fabricated_ids` is `cited - retrieved` — real-but-unretrieved citations, SPEC §10.5's
    "worse failure" precisely because they survive a corpus-wide check that only asks
    whether the id exists at all."""

    cited_ids: frozenset[str]
    retrieved_ids: frozenset[str]
    fabricated_ids: frozenset[str]

    @property
    def valid(self) -> bool:
        return not self.fabricated_ids


def retrieved_citation_ids(contexts: list[Candidate]) -> frozenset[str]:
    """The guardrail's right-hand side (SPEC §10.5's `retrieved_context`): every real
    article `citation_id` among `contexts`. Excludes the no-article marker
    (`rag.retrieval.quota`) — it carries no `citation_id` and was reached by no retrieval
    path, so it can never itself be a legitimate citation."""
    return frozenset(
        str(c.payload["citation_id"])
        for c in contexts
        if c.register is Register.ARTICLE and c.id != NO_ARTICLE_MARKER_ID
    )


def check_citations(envelope: Envelope, contexts: list[Candidate]) -> CitationOutcome:
    """`cited ⊆ retrieved_context` (SPEC §10.5). Pure containment over sets of ids — no
    corpus lookup, so an article that is real but was never retrieved still fails."""
    cited = frozenset(f.article_id for f in envelope.fondement_juridique)
    retrieved = retrieved_citation_ids(contexts)
    return CitationOutcome(cited_ids=cited, retrieved_ids=retrieved, fabricated_ids=cited - retrieved)


def repair_for_display(envelope: Envelope, outcome: CitationOutcome) -> Envelope:
    """The demo path only (SPEC §10.5): drops fabricated citations, and — if that leaves an
    answerable turn with none left — surfaces the no-article marker, since the model's own
    stated grounding turned out to be worth nothing. **Never called on the eval path**:
    `outcome` itself, unrepaired, is what eval records, or the failure it exists to catch
    disappears (SPEC §10.5, ADR-0009).
    """
    if outcome.valid:
        return envelope

    kept = [f for f in envelope.fondement_juridique if f.article_id not in outcome.fabricated_ids]

    if isinstance(envelope, Reponse):
        aucun_fondement = envelope.aucun_fondement
        if not kept:
            aucun_fondement = NO_ARTICLE_MARKER_TEXT
        return envelope.model_copy(
            update={"fondement_juridique": kept, "aucun_fondement": aucun_fondement}
        )

    return envelope.model_copy(update={"fondement_juridique": kept})
