"""The typed answer envelope (SPEC §10.2, ADR-0009, #42).

A Pydantic discriminated union, reached through LangChain's `with_structured_output()` (the
LangChain wiring itself is paired with @Zameloth, per the #42 issue comment — this module is
the schema it is built against, not the chain). The schema constrains the **envelope, not
the writing**: `explanation` stays unconstrained prose in both branches, and the citation
guardrail (`rag.generation.citation`) loops over the typed `fondement_juridique` field
instead of regexing that prose.

`Reponse` carries both of SPEC §10.3's non-refusal terminal states — "answerable" and "no
article" are the same shape, distinguished only by whether `aucun_fondement` is filled, not
by a second discriminator value. `Refus` carries the other two — regulated-act and
out-of-corpus — distinguished from each other by `motif`, never inferred from retrieval
(SPEC §10.3: a scope failure and a retrieval failure have completely different fixes).
"""

from __future__ import annotations

import enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

__all__ = [
    "Envelope",
    "FondementJuridique",
    "Motif",
    "Reponse",
    "Refus",
]


class Motif(enum.Enum):
    """The refusal-class enum (SPEC §10.2, §10.3) — `Refus` only, never null on that
    branch. `recommandation_produit`/`conseil_action` are the regulated-act refusal (SPEC
    §10.4: the refusal line is *recommending*, not *explaining*); `hors_corpus` is the
    scope refusal (SPEC §10.3: never inferred from a retrieval failure)."""

    RECOMMANDATION_PRODUIT = "recommandation_produit"
    CONSEIL_ACTION = "conseil_action"
    HORS_CORPUS = "hors_corpus"


class FondementJuridique(BaseModel):
    """One cited article (SPEC §10.2). `article_id` is `citation_id` copied **verbatim**
    from the prompt's article label (SPEC §10.6) — never composed, never a `[A1]`-style
    handle — so `cited ⊆ retrieved_context` (`rag.generation.citation`) is a plain
    string-set comparison, not a parse."""

    article_id: str
    gloss: str


class Reponse(BaseModel):
    """The non-refusal branch (SPEC §10.3): "answerable" (`aucun_fondement: None`) and "no
    article" (`aucun_fondement` filled) are one shape, not two — the floor-not-met case is
    a stated outcome, not a failure (SPEC §9.5), so it does not earn its own discriminator
    value."""

    type: Literal["reponse"] = "reponse"
    explanation: str
    fondement_juridique: list[FondementJuridique] = Field(default_factory=list)
    aucun_fondement: str | None = None


class Refus(BaseModel):
    """The refusal branch (SPEC §10.3, §10.4). Still carries `explanation` and
    `fondement_juridique` — SPEC §10.4's "refusals still answer the informational part":
    the regulated act is *recommending*, not *explaining*, so a bare refusal with nothing
    else would decline something service-public.fr does under state mandate."""

    type: Literal["refus"] = "refus"
    explanation: str
    fondement_juridique: list[FondementJuridique] = Field(default_factory=list)
    motif: Motif


Envelope = Annotated[Reponse | Refus, Field(discriminator="type")]
