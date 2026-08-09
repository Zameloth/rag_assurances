"""SPEC §6.4 / §7 / #25 — orchestration: chunk -> payload -> point id -> vector -> upsert.

Deterministic stub vectors throughout (SPEC #25's own acceptance criterion): this ticket
needs no model, only something shaped like one. `embed` here returns a distinct dense
vector per input text (`[index, 0, 0, 0]`) precisely so a test can catch the two arrays
(texts in, points out) silently getting out of order.
"""

from collections.abc import Sequence
from typing import Any

import pytest
from conftest import CreateCollection
from qdrant_client import QdrantClient, models

import rag.ingest.upsert as upsert_module
from rag.ingest.payload import article_point_id, fiche_point_id
from rag.ingest.upsert import Embedding, upsert_articles, upsert_fiches


def _payload(point: models.Record) -> dict[str, Any]:
    assert point.payload is not None
    return point.payload


def stub_embed(texts: Sequence[str]) -> list[Embedding]:
    return [
        ([float(i), 0.0, 0.0, 0.0], models.SparseVector(indices=[i], values=[1.0]))
        for i in range(len(texts))
    ]


def _article_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "cid": "LEGIARTI000000000001",
        "id": "LEGIARTI000000000002",
        "citation_id": "L113-1",
        "texte": "placeholder",
        "texteHtml": "<p>placeholder</p>",
        "etat": "VIGUEUR",
        "dateDebut": "2018-04-01",
        "sectionParentId": "LEGISCTA000000000099",
        "sectionParentTitre": "Chapitre I : Placeholder",
        "fullSectionsTitre": "Partie réglementaire > Livre Ier",
    }
    base.update(overrides)
    return base


_SHORT_TEXT = "Le contrat d'assurance est régi par les dispositions du présent titre et les stipulations particulières."

_FICHE_XML = (
    b'<?xml version="1.0" encoding="UTF-8"?>'
    b'<Publication xmlns:dc="http://purl.org/dc/elements/1.1/" ID="F1" '
    b'type="Fiche d\'information conditionn\xc3\xa9e" '
    b'spUrl="https://www.service-public.gouv.fr/particuliers/vosdroits/F1">'
    b"<dc:title>Titre de test</dc:title>"
    b"<dc:date>modified 2025-04-28</dc:date>"
    b"<dc:source>https://www.legifrance.gouv.fr/codes/id/LEGISCTA000000000099</dc:source>"
    b'<FilDAriane><Niveau ID="Particuliers">Accueil particuliers</Niveau></FilDAriane>'
    b"<Texte><Paragraphe>La modification du contrat peut \xc3\xaatre demand\xc3\xa9e par l'assur\xc3\xa9 "
    b"\xc3\xa0 tout moment de la vie du contrat d'assurance habitation.</Paragraphe></Texte>"
    b"</Publication>"
)


def test_upsert_articles_writes_one_point_per_chunk(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, "articles")
    rows = [_article_row(texteHtml=f"<p>{_SHORT_TEXT}</p>")]

    written = upsert_articles(qdrant, "articles", rows, stub_embed)

    assert written == 1
    assert qdrant.count("articles").count == 1


def test_upsert_articles_is_idempotent(qdrant: QdrantClient, create_collection: CreateCollection) -> None:
    """SPEC §6.4 — re-running is an overwrite in place, not a second copy of the corpus."""
    create_collection(qdrant, "articles")
    rows = [_article_row(texteHtml=f"<p>{_SHORT_TEXT}</p>")]

    upsert_articles(qdrant, "articles", rows, stub_embed)
    upsert_articles(qdrant, "articles", rows, stub_embed)

    assert qdrant.count("articles").count == 1


def test_upsert_articles_keeps_texts_and_vectors_in_order_across_rows(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    """A row that chunks to 2+ points, followed by another row, must not scramble which
    vector lands on which point — the exact failure a flattened zip could introduce."""
    create_collection(qdrant, "articles")
    long_html = "".join(f"<p>{_SHORT_TEXT} Bloc numéro {i}.</p>" for i in range(40))
    rows = [
        _article_row(cid="LEGIARTI_A", id="LEGIARTI_A_v1", citation_id="L100-1", texteHtml=long_html),
        _article_row(cid="LEGIARTI_B", id="LEGIARTI_B_v1", citation_id="L100-2", texteHtml=f"<p>{_SHORT_TEXT}</p>"),
    ]

    upsert_articles(qdrant, "articles", rows, stub_embed)

    point_id = article_point_id("LEGIARTI_B", 0)
    [point] = qdrant.retrieve("articles", ids=[point_id], with_payload=True, with_vectors=True)
    assert _payload(point)["citation_id"] == "L100-2"


def test_upsert_articles_batches_large_point_sets_across_multiple_upsert_calls(
    monkeypatch: pytest.MonkeyPatch, qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    """SPEC #26 — one request over Qdrant's REST payload cap must never happen (measured
    failure against the real corpus: a single 2,801-point call serialized to 70 MB against
    a 32 MB limit). Regression-tested at a monkeypatched batch size of 2 rather than by
    generating tens of megabytes of fixture data to reproduce the real threshold."""
    monkeypatch.setattr(upsert_module, "_UPSERT_BATCH_SIZE", 2)
    create_collection(qdrant, "articles")
    rows = [
        _article_row(cid=f"LEGIARTI_{i}", id=f"LEGIARTI_{i}_v1", citation_id=f"L100-{i}")
        for i in range(5)
    ]
    call_sizes: list[int] = []
    original_upsert = qdrant.upsert

    def _spying_upsert(**kwargs: Any) -> Any:
        call_sizes.append(len(kwargs["points"]))
        return original_upsert(**kwargs)

    monkeypatch.setattr(qdrant, "upsert", _spying_upsert)

    written = upsert_articles(qdrant, "articles", rows, stub_embed)

    assert written == 5
    assert qdrant.count("articles").count == 5
    assert call_sizes == [2, 2, 1]


def test_upsert_fiches_writes_one_point_per_chunk(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, "fiches")

    written = upsert_fiches(qdrant, "fiches", [_FICHE_XML], stub_embed)

    assert written == 1
    assert qdrant.count("fiches").count == 1


def test_upsert_fiches_is_idempotent(qdrant: QdrantClient, create_collection: CreateCollection) -> None:
    create_collection(qdrant, "fiches")

    upsert_fiches(qdrant, "fiches", [_FICHE_XML], stub_embed)
    upsert_fiches(qdrant, "fiches", [_FICHE_XML], stub_embed)

    assert qdrant.count("fiches").count == 1


def test_upsert_fiches_payload_round_trips(qdrant: QdrantClient, create_collection: CreateCollection) -> None:
    create_collection(qdrant, "fiches")

    upsert_fiches(qdrant, "fiches", [_FICHE_XML], stub_embed)

    point_id = fiche_point_id("F1", 0)
    [point] = qdrant.retrieve("fiches", ids=[point_id], with_payload=True)
    payload = _payload(point)
    assert payload["fiche_id"] == "F1"
    assert payload["title"] == "Titre de test"
    assert payload["section_ids"] == ["LEGISCTA000000000099"]
