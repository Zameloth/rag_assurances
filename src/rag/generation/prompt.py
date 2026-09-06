"""Prompt assembly — context grouped by register, history stripped of citations (SPEC
§10.6, ADR-0009, #42).

Context is grouped by register, never interleaved by score: a labelled fiche block and a
labelled article block, mirroring the register quota's own 4+4 split (SPEC §9.5) — the
two-register split *is* the architecture, so presenting it as one ranked blob would ask the
model to re-derive a distinction the pipeline already made.

`citation_id` **is the label** — no `[A1]`/`[F1]` handle, so the guardrail's
`cited ⊆ retrieved_context` check (`rag.generation.citation`) stays a plain set
comparison. Excluded, each for a reason SPEC §10.6 names: URLs (the model writes URLs into
prose, which an id-based check cannot validate), `provenance` (the quota already encodes it
structurally), `fiche_id` (`fondement_juridique` is articles-only), and dates
(app-rendered). This module only ever reads `title`/`chapitre_titre`/`cas_label`/`text` off
a fiche payload and `citation_id`/`full_sections_titre`/`text` off an article payload, so
those exclusions hold by construction rather than by a filter someone has to remember.

Building the actual LangChain messages (`SystemMessage`/`HumanMessage`/`AIMessage`) and the
`with_structured_output()` call around them is paired with @Zameloth (#42 issue comment);
`build_messages` returns plain `(role, content)` pairs so that wiring is a thin conversion,
not a rewrite of this module's logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from rag.retrieval.candidates import Candidate, Register
from rag.retrieval.quota import NO_ARTICLE_MARKER_ID, NO_ARTICLE_MARKER_TEXT

__all__ = [
    "ARTICLE_HEADING",
    "FICHE_HEADING",
    "SYSTEM_PROMPT",
    "HistoryTurn",
    "Message",
    "build_context_section",
    "build_messages",
    "strip_breadcrumb_root",
]

FICHE_HEADING = "### Fiches service-public.fr"
ARTICLE_HEADING = "### Code des assurances"

Message = tuple[Literal["system", "user", "assistant"], str]

# SPEC §10.3, §10.4, §10.5, §10.6 — the policy this prompt has to encode: the schema
# constrains the envelope, this text constrains the judgement calls the schema itself
# can't (which of the four terminal states, when the refusal line is crossed, what counts
# as a citable id). Deliberately the most likely piece of this module to be rewritten by
# hand during iteration (SPEC §10.3: "the most iteration-prone component in the system") —
# nothing downstream depends on its exact wording, only on `build_messages` keeping it as
# the first message.
SYSTEM_PROMPT = """\
Tu es l'assistant assurance de service-public.fr. Tu réponds toujours en français, \
quelle que soit la langue de la question.

Le contexte ci-dessous est composé de deux blocs distincts : des fiches conseil en \
français courant, et des articles du Code des assurances en français juridique. Utilise \
les deux registres pour répondre, mais ne mélange jamais leurs identifiants.

Distingue quatre situations, jamais plus :
1. La question trouve une fiche et un article pertinents : réponds (`type: "reponse"`), \
`aucun_fondement` reste vide.
2. La question trouve une fiche mais aucun article du contexte ne dépasse le seuil de \
pertinence : réponds quand même (`type: "reponse"`), et remplis `aucun_fondement` pour \
dire explicitement qu'il n'y a pas de fondement juridique direct dans le corpus. Ce n'est \
pas un refus.
3. La question demande une recommandation de produit ou un conseil d'action personnalisé \
(« quelle assurance choisir ? », « dois-je résilier ? ») : refuse cette partie \
(`type: "refus"`, `motif: "recommandation_produit"` ou `"conseil_action"`), mais explique \
quand même la règle applicable si le contexte en contient une — refuser de recommander \
n'est pas refuser d'expliquer.
4. La question sort du champ de l'assurance couvert par le contexte fourni : refuse \
(`type: "refus"`, `motif: "hors_corpus"`). Ne déduis jamais cela d'un contexte vide ou \
non pertinent en apparence — seulement du sujet de la question lui-même.

Pour chaque article que tu cites dans `fondement_juridique`, copie son identifiant \
exactement comme il apparaît en tête du bloc article dans le contexte (par exemple \
"L113-15-2") : ne construis jamais un identifiant, n'utilise jamais de repère comme \
"[A1]" ou "[F1]". Ne cite un article que s'il apparaît réellement dans le contexte fourni \
pour ce tour — jamais un article que tu connais par ailleurs.
"""


@dataclass(frozen=True)
class HistoryTurn:
    """One prior turn (SPEC §10.6's history stripping, ADR-0009). `content` is the full
    message for a `user` turn, and **`explanation` only** for an `assistant` turn — there
    is no `fondement_juridique` field on this type at all, so a prior citation list has
    nothing to be carried in. Stripping happens by construction, not by a runtime filter
    over a richer type, which is what keeps it load-bearing rather than best-effort (SPEC
    §10.6: it also leaves the condenser no honest source for an unasked-for reference)."""

    role: Literal["user", "assistant"]
    content: str


def strip_breadcrumb_root(full_sections_titre: str | None) -> str:
    """The article breadcrumb **minus its first segment** (SPEC §10.6): "Partie
    législative" is already carried by the article number's own `L`/`R`/`A`/`D` prefix
    (SPEC §7.3), so repeating it in every breadcrumb is pure noise. `None`/empty input
    (a prose annexe with no breadcrumb) returns `""`, not a crash."""
    if not full_sections_titre:
        return ""
    segments = [segment.strip() for segment in full_sections_titre.split(">")]
    return " > ".join(segments[1:])


def _fiche_block(candidate: Candidate) -> str:
    payload = candidate.payload
    label = " · ".join(
        str(payload[key]) for key in ("title", "chapitre_titre", "cas_label") if payload.get(key)
    )
    text = str(payload.get("text", ""))
    return f"— {label}\n{text}" if label else text


def _article_block(candidate: Candidate) -> str:
    if candidate.id == NO_ARTICLE_MARKER_ID:
        return NO_ARTICLE_MARKER_TEXT
    payload = candidate.payload
    citation_id = str(payload["citation_id"])
    full_sections_titre = payload.get("full_sections_titre")
    breadcrumb = strip_breadcrumb_root(
        full_sections_titre if isinstance(full_sections_titre, str) else None
    )
    label = f"{citation_id} · {breadcrumb}" if breadcrumb else citation_id
    text = str(payload.get("text", ""))
    return f"— {label}\n{text}"


def build_context_section(contexts: list[Candidate]) -> str:
    """SPEC §10.6: a labelled fiche block, then a labelled article block — never one
    ranked blob. An empty register is simply omitted (no empty heading printed)."""
    fiches = [c for c in contexts if c.register is Register.FICHE]
    articles = [c for c in contexts if c.register is Register.ARTICLE]

    sections = []
    if fiches:
        body = "\n\n".join(_fiche_block(c) for c in fiches)
        sections.append(f"{FICHE_HEADING}\n\n{body}")
    if articles:
        body = "\n\n".join(_article_block(c) for c in articles)
        sections.append(f"{ARTICLE_HEADING}\n\n{body}")
    return "\n\n".join(sections)


def build_messages(
    raw_turn: str,
    contexts: list[Candidate],
    history: Sequence[HistoryTurn] = (),
) -> list[Message]:
    """The full message list for the generation call (SPEC §10.6, §10.7): system
    instructions, then `history` verbatim (already stripped by construction — see
    `HistoryTurn`), then the current turn with its context section attached.

    `raw_turn` is the current user turn as typed — condensation's output never reaches
    generation (ADR-0008), so this function never sees a condensed query, only ever the
    turn to answer and the contexts already retrieved for it.
    """
    messages: list[Message] = [("system", SYSTEM_PROMPT)]
    messages.extend((turn.role, turn.content) for turn in history)

    context_section = build_context_section(contexts)
    user_content = (
        f"{context_section}\n\n### Question\n\n{raw_turn}" if context_section else raw_turn
    )
    messages.append(("user", user_content))
    return messages
