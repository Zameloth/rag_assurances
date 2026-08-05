"""SPEC §6-§7 / #25 — end to end in `QdrantClient(":memory:")` with deterministic stub
vectors: arm collections, aliases, payload, idempotent upsert, all wired together.

This is the acceptance criterion from the ticket read literally — no model, only something
shaped like one — and it is deliberately separate from `test_upsert.py` (which drives
`upsert_articles`/`upsert_fiches` directly against a bare collection) and
`test_arms.py` (collection shape alone): this file is the one place asserting that
an arm created by `arms.py`, addressed only through its alias, accepts points built
by `payload.py` via `upsert.py` — the whole path #26 will later swap a real embedder into.
"""

from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from rag.ingest.arms import ensure_articles_collection, ensure_fiches_collection, flip_alias
from rag.ingest.upsert import Embedding, upsert_articles, upsert_fiches

DENSE_DIM = 4


def _payload(point: models.Record) -> dict[str, Any]:
    assert point.payload is not None
    return point.payload

ARTICLES_ARM = "articles__test_arm__v1"
FICHES_ARM = "fiches__test_arm__v1"

_ARTICLE_ROW = {
    "cid": "LEGIARTI000006785773",
    "id": "LEGIARTI000036754138",
    "citation_id": "L113-3",
    "texte": "placeholder",
    "texteHtml": (
        "<p>Le contrat d'assurance est régi par les dispositions du présent titre "
        "ainsi que par les stipulations particulières convenues entre les parties.</p>"
    ),
    "etat": "VIGUEUR",
    "dateDebut": "2018-04-01",
    "sectionParentId": "LEGISCTA000006156957",
    "fullSectionsTitre": "Partie réglementaire > Livre Ier",
}

_FICHE_XML = (
    b'<?xml version="1.0" encoding="UTF-8"?>'
    b'<Publication xmlns:dc="http://purl.org/dc/elements/1.1/" ID="F2594" '
    b'type="Fiche d\'information conditionn\xc3\xa9e" '
    b'spUrl="https://www.service-public.gouv.fr/particuliers/vosdroits/F2594">'
    b"<dc:title>Modification du contrat d'assurance habitation</dc:title>"
    b"<dc:date>modified 2025-04-28</dc:date>"
    b"<dc:source>https://www.legifrance.gouv.fr/codes/id/LEGISCTA000006156957</dc:source>"
    b'<FilDAriane><Niveau ID="Particuliers">Accueil particuliers</Niveau></FilDAriane>'
    b"<Texte><Paragraphe>La modification du contrat peut \xc3\xaatre demand\xc3\xa9e par "
    b"l'assur\xc3\xa9, d\xc3\xa9cid\xc3\xa9e par l'assureur, ou impos\xc3\xa9e par la loi.</Paragraphe></Texte>"
    b"</Publication>"
)


def stub_embed(texts: Sequence[str]) -> list[Embedding]:
    return [
        ([float(i), 0.0, 0.0, 0.0], models.SparseVector(indices=[i], values=[1.0]))
        for i in range(len(texts))
    ]


def test_both_registers_land_behind_their_own_alias_with_no_cross_leak(qdrant: QdrantClient) -> None:
    try:
        ensure_articles_collection(qdrant, ARTICLES_ARM, dense_dim=DENSE_DIM)
        ensure_fiches_collection(qdrant, FICHES_ARM, dense_dim=DENSE_DIM)

        upsert_articles(qdrant, ARTICLES_ARM, [_ARTICLE_ROW], stub_embed)
        upsert_fiches(qdrant, FICHES_ARM, [_FICHE_XML], stub_embed)

        flip_alias(qdrant, "articles", ARTICLES_ARM)
        flip_alias(qdrant, "fiches", FICHES_ARM)

        assert qdrant.count("articles").count == 1
        assert qdrant.count("fiches").count == 1
        [article_point] = qdrant.scroll("articles", with_payload=True)[0]
        [fiche_point] = qdrant.scroll("fiches", with_payload=True)[0]
        assert _payload(article_point)["legiarti_cid"] == "LEGIARTI000006785773"
        assert _payload(fiche_point)["fiche_id"] == "F2594"
    finally:
        qdrant.delete_collection(ARTICLES_ARM)
        qdrant.delete_collection(FICHES_ARM)


def test_rerunning_the_whole_pipeline_through_the_alias_is_a_no_op_upsert(qdrant: QdrantClient) -> None:
    """SPEC §6.4 — the acceptance criterion in plain terms: re-running is an overwrite in
    place, addressed the same way `make ingest` would, through the alias."""
    try:
        ensure_articles_collection(qdrant, ARTICLES_ARM, dense_dim=DENSE_DIM)
        flip_alias(qdrant, "articles", ARTICLES_ARM)

        upsert_articles(qdrant, "articles", [_ARTICLE_ROW], stub_embed)
        upsert_articles(qdrant, "articles", [_ARTICLE_ROW], stub_embed)

        assert qdrant.count("articles").count == 1
    finally:
        qdrant.delete_collection(ARTICLES_ARM)
