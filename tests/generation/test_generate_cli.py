"""`python -m rag.generate` — CLI wiring, injected client/embed/generate_fn (SPEC §10, #42).

Never loads BGE-M3 or calls OpenRouter: `main`'s `embed`/`generate_fn` parameters are
always supplied here, the same injection seam `rag.query.main` already takes for
`client`/`embed` (`tests/retrieval/test_query_cli.py`).
"""

from __future__ import annotations

import pytest
from conftest import CreateCollection, raw_point, stub_embed
from qdrant_client import QdrantClient

from rag.generate import main
from rag.generation.pipeline import GenerateFn
from rag.generation.prompt import Message
from rag.generation.schema import Envelope, FondementJuridique, Motif, Refus, Reponse
from rag.retrieval.legs import ARTICLES_ALIAS, FICHES_ALIAS


def _fake_generate_fn(envelope: Envelope) -> GenerateFn:
    def fn(messages: list[Message]) -> Envelope:
        return envelope

    return fn


def test_main_prints_reponse_with_fondement_juridique(
    qdrant: QdrantClient, create_collection: CreateCollection, capsys: pytest.CaptureFixture[str]
) -> None:
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"citation_id": "L113-2", "text": "Le texte."})],
    )
    envelope = Reponse(
        explanation="Voici la réponse.",
        fondement_juridique=[FondementJuridique(article_id="L113-2", gloss="Le principe.")],
    )

    main(
        ["Quelle franchise pour un dégât des eaux ?"],
        client=qdrant,
        embed=stub_embed([1.0, 0.0, 0.0, 0.0]),
        generate_fn=_fake_generate_fn(envelope),
    )

    out = capsys.readouterr().out
    assert "[REPONSE]" in out
    assert "Voici la réponse." in out
    assert "L113-2: Le principe." in out
    assert "GUARDRAIL" not in out


def test_main_prints_refus_with_motif(
    qdrant: QdrantClient, create_collection: CreateCollection, capsys: pytest.CaptureFixture[str]
) -> None:
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    envelope = Refus(explanation="Je ne peux pas recommander.", motif=Motif.RECOMMANDATION_PRODUIT)

    main(
        ["Quelle assurance auto choisir ?"],
        client=qdrant,
        embed=stub_embed([1.0, 0.0, 0.0, 0.0]),
        generate_fn=_fake_generate_fn(envelope),
    )

    out = capsys.readouterr().out
    assert "[REFUS] motif=recommandation_produit" in out
    assert "Je ne peux pas recommander." in out


def test_main_prints_no_fondement_marker(
    qdrant: QdrantClient, create_collection: CreateCollection, capsys: pytest.CaptureFixture[str]
) -> None:
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    envelope = Reponse(explanation="Réponse sans article.", aucun_fondement="Aucun texte trouvé.")

    main(
        ["une question"],
        client=qdrant,
        embed=stub_embed([1.0, 0.0, 0.0, 0.0]),
        generate_fn=_fake_generate_fn(envelope),
    )

    out = capsys.readouterr().out
    assert "[REPONSE — aucun fondement juridique dans le corpus]" in out
    assert "Aucun texte trouvé." in out


def test_main_flags_fabricated_citations_without_repairing(
    qdrant: QdrantClient, create_collection: CreateCollection, capsys: pytest.CaptureFixture[str]
) -> None:
    """SPEC §10.5 — never auto-repaired: a citation absent from `retrieved_context` is
    named as fabricated but stays printed in `fondement_juridique`, exactly what the model
    returned."""
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    envelope = Reponse(
        explanation="Réponse.",
        fondement_juridique=[FondementJuridique(article_id="L999-9", gloss="Inventé.")],
    )

    main(
        ["une question"],
        client=qdrant,
        embed=stub_embed([1.0, 0.0, 0.0, 0.0]),
        generate_fn=_fake_generate_fn(envelope),
    )

    out = capsys.readouterr().out
    assert "L999-9: Inventé." in out
    assert "[GUARDRAIL] fabricated citations (not in retrieved context): L999-9" in out


def test_main_does_not_close_an_injected_client(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    """Same contract as `rag.query.main`/`rag.ingest.pipeline.main`: only a client this
    CLI created itself gets closed."""
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    envelope = Reponse(explanation="Réponse.")

    main(
        ["une question"],
        client=qdrant,
        embed=stub_embed([1.0, 0.0, 0.0, 0.0]),
        generate_fn=_fake_generate_fn(envelope),
    )

    # Still usable — closing it would make this raise.
    qdrant.get_collections()
