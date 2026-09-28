"""`GenerationResult` -> what the app renders (SPEC §13.5, §10.5, #50).

Presentation only — no retrieval or generation logic lives here (CONTEXT.md: "the pipeline
is a library"). The one decision taken on the way is SPEC §10.5's demo repair
(`rag.generation.citation.repair_for_display`): a fabricated citation is dropped from the
display copy, never from the result eval sees.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import date

from rag.generation.citation import repair_for_display
from rag.generation.pipeline import GenerationResult
from rag.generation.schema import Envelope, Motif, Refus
from rag.retrieval.candidates import Register
from rag.retrieval.quota import NO_ARTICLE_MARKER_ID

__all__ = [
    "AnswerState",
    "AnswerView",
    "ArticleCitation",
    "FicheSource",
    "build_view",
    "legifrance_url",
]

# SPEC §13.5: the article link is derived, never stored — and from `legiarti_version_id`,
# never `cid`, which points at the article's first version.
LEGIFRANCE_ARTICLE_URL = "https://www.legifrance.gouv.fr/codes/article_lc/{legiarti_version_id}"


_MOTIF_BADGES = {
    Motif.RECOMMANDATION_PRODUIT: "Recommandation de produit",
    Motif.CONSEIL_ACTION: "Conseil personnalisé",
    Motif.HORS_CORPUS: "Hors corpus",
}


class AnswerState(enum.Enum):
    """SPEC §10.3's four terminal states, one partial each (SPEC §13.1). Coarser than
    `rag.eval.schema.EXPECTED_STATES` on purpose: the two regulated-act motifs are one
    state to a reader — the badge says which act was declined, the partial is the same."""

    REPONSE = "reponse"
    REPONSE_SANS_ARTICLE = "reponse_sans_article"
    REFUS_REGULE = "refus_regule"
    HORS_CORPUS = "hors_corpus"


@dataclass(frozen=True)
class ArticleCitation:
    citation_id: str
    gloss: str
    url: str


@dataclass(frozen=True)
class FicheSource:
    title: str
    url: str
    date_modified: date


@dataclass(frozen=True)
class AnswerView:
    """`envelope` is the display copy — repaired when the guardrail fired, so `state` and
    `fondement_juridique` are read off it, not off the model's unrepaired output. `badge`
    is the state's label, with the declined act named on a regulated-act refusal."""

    state: AnswerState
    badge: str
    envelope: Envelope
    fondement_juridique: tuple[ArticleCitation, ...]
    fiches: tuple[FicheSource, ...]


def legifrance_url(legiarti_version_id: str) -> str:
    return LEGIFRANCE_ARTICLE_URL.format(legiarti_version_id=legiarti_version_id)


def build_view(result: GenerationResult) -> AnswerView:
    envelope = repair_for_display(result.envelope, result.citation_outcome)

    # After repair every cited id is in the retrieved context, and every article payload
    # carries `legiarti_version_id` (`rag.ingest.payload.build_article_payload` writes it
    # from the source row's required `id`) — so this lookup cannot miss.
    version_ids = {
        str(c.payload["citation_id"]): str(c.payload["legiarti_version_id"])
        for c in result.contexts
        if c.register is Register.ARTICLE and c.id != NO_ARTICLE_MARKER_ID
    }
    fondement_juridique = tuple(
        ArticleCitation(
            citation_id=f.article_id,
            gloss=f.gloss,
            url=legifrance_url(version_ids[f.article_id]),
        )
        for f in envelope.fondement_juridique
    )

    fiches: dict[str, FicheSource] = {}
    for c in result.contexts:
        if c.register is Register.FICHE and c.payload["fiche_id"] not in fiches:
            fiches[c.payload["fiche_id"]] = FicheSource(
                title=str(c.payload["title"]),
                url=str(c.payload["sp_url"]),
                date_modified=date.fromisoformat(str(c.payload["date_modified"])),
            )

    state = _state(envelope)
    return AnswerView(
        state=state,
        badge=_badge(state, envelope),
        envelope=envelope,
        fondement_juridique=fondement_juridique,
        fiches=tuple(fiches.values()),
    )


def _state(envelope: Envelope) -> AnswerState:
    if isinstance(envelope, Refus):
        if envelope.motif is Motif.HORS_CORPUS:
            return AnswerState.HORS_CORPUS
        return AnswerState.REFUS_REGULE
    if envelope.aucun_fondement is not None:
        return AnswerState.REPONSE_SANS_ARTICLE
    return AnswerState.REPONSE


def _badge(state: AnswerState, envelope: Envelope) -> str:
    if isinstance(envelope, Refus):
        return _MOTIF_BADGES[envelope.motif]
    return "Sans article" if state is AnswerState.REPONSE_SANS_ARTICLE else "Réponse"
