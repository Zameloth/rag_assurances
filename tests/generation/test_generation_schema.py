"""SPEC §10.2, ADR-0009, #42 — the typed envelope validates as a discriminated union, and
each branch carries exactly the fields its terminal state needs."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from rag.generation.schema import Envelope, FondementJuridique, Motif, Refus, Reponse

_ENVELOPE: TypeAdapter[Reponse | Refus] = TypeAdapter(Envelope)


def test_reponse_answerable_has_no_marker_and_no_motif() -> None:
    envelope = _ENVELOPE.validate_python(
        {
            "type": "reponse",
            "explanation": "Vous devez prévenir l'assureur dans les quinze jours.",
            "fondement_juridique": [{"article_id": "L113-15-2", "gloss": "résiliation"}],
        }
    )

    assert isinstance(envelope, Reponse)
    assert envelope.aucun_fondement is None
    assert envelope.fondement_juridique == [
        FondementJuridique(article_id="L113-15-2", gloss="résiliation")
    ]


def test_reponse_no_article_fills_the_marker_and_is_not_a_refusal() -> None:
    """SPEC §10.3: "no article" is `reponse`, not `refus` — the floor-not-met case is a
    stated outcome the model fills in, not a failure."""
    envelope = _ENVELOPE.validate_python(
        {
            "type": "reponse",
            "explanation": "Cette situation n'est pas couverte par un article précis.",
            "fondement_juridique": [],
            "aucun_fondement": "Pas de fondement juridique direct dans le corpus.",
        }
    )

    assert isinstance(envelope, Reponse)
    assert envelope.aucun_fondement == "Pas de fondement juridique direct dans le corpus."


def test_refus_carries_motif_and_still_answers_the_informational_part() -> None:
    """SPEC §10.4: a refusal still carries `explanation` + `fondement_juridique` where one
    exists — the regulated act is recommending, not explaining."""
    envelope = _ENVELOPE.validate_python(
        {
            "type": "refus",
            "explanation": "Je ne peux pas recommander un contrat, mais l'assurance "
            "responsabilité civile automobile est obligatoire.",
            "fondement_juridique": [{"article_id": "L211-1", "gloss": "obligation d'assurance"}],
            "motif": "recommandation_produit",
        }
    )

    assert isinstance(envelope, Refus)
    assert envelope.motif is Motif.RECOMMANDATION_PRODUIT
    assert envelope.fondement_juridique[0].article_id == "L211-1"


def test_refus_requires_a_motif() -> None:
    with pytest.raises(ValidationError):
        _ENVELOPE.validate_python(
            {"type": "refus", "explanation": "Non.", "fondement_juridique": []}
        )


def test_unknown_type_is_rejected_by_the_discriminator() -> None:
    with pytest.raises(ValidationError):
        _ENVELOPE.validate_python(
            {"type": "autre", "explanation": "x", "fondement_juridique": []}
        )
