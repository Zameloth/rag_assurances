"""SPEC §12.8 / #38 — challenger-arm construction for the two pre-ladder A/Bs.

Small fixture corpora, not the real committed one (`test_ingest_pipeline.py` already pays
that cost for the shared chunker/upsert path) — this file is only about the enrichment
seam actually reaching `upsert_articles`/`upsert_fiches`, which a two-row fixture proves as
well as 2,377 rows would.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from qdrant_client import QdrantClient, models

from rag.ingest.ab_arms import (
    ARTICLE_BREADCRUMB_ARM,
    FICHE_HEADER_ARM,
    build_article_breadcrumb_arm,
    build_fiche_header_arm,
)
from rag.ingest.arms import DENSE_DIM
from rag.ingest.payload import article_point_id, fiche_point_id
from rag.ingest.upsert import Embedding, EmbedFn

_ARTICLE_ROW: dict[str, object] = {
    "cid": "LEGIARTI000000000001",
    "id": "LEGIARTI000000000002",
    "citation_id": "L113-3",
    "texte": "placeholder",
    "texteHtml": (
        "<p>La résiliation prend effet un mois après réception de la lettre par "
        "l'assureur, sauf stipulation contraire du contrat conclu entre les parties.</p>"
    ),
    "etat": "VIGUEUR",
    "dateDebut": "2018-04-01",
    "sectionParentId": "LEGISCTA000000000099",
    "fullSectionsTitre": "Partie législative > Livre Ier : Le contrat > Titre Ier : Résiliation",
}

_FICHE_XML = (
    b'<?xml version="1.0" encoding="UTF-8"?>'
    b'<Publication xmlns:dc="http://purl.org/dc/elements/1.1/" ID="F1" '
    b'type="Fiche d\'information conditionn\xc3\xa9e" '
    b'spUrl="https://www.service-public.gouv.fr/particuliers/vosdroits/F1">'
    b"<dc:title>Modification du contrat d'assurance habitation</dc:title>"
    b"<dc:date>modified 2025-04-28</dc:date>"
    b"<dc:source>https://www.legifrance.gouv.fr/codes/id/LEGISCTA000000000099</dc:source>"
    b'<FilDAriane><Niveau ID="Particuliers">Accueil particuliers</Niveau></FilDAriane>'
    b"<Texte><Paragraphe>La modification du contrat peut \xc3\xaatre demand\xc3\xa9e "
    b"par l'assur\xc3\xa9 \xc3\xa0 tout moment de la vie du contrat d'assurance "
    b"habitation, notamment en cas de changement de situation personnelle ou "
    b"familiale.</Paragraphe></Texte>"
    b"</Publication>"
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    import json

    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _spying_embed(calls: list[list[str]]) -> EmbedFn:
    def embed(texts: Sequence[str]) -> list[Embedding]:
        calls.append(list(texts))
        return [
            ([1.0] * DENSE_DIM, models.SparseVector(indices=[i], values=[1.0]))
            for i in range(len(texts))
        ]

    return embed


class TestBuildArticleBreadcrumbArm:
    def test_creates_the_named_arm_and_writes_one_point_per_chunk(
        self, qdrant: QdrantClient, tmp_path: Path
    ) -> None:
        articles_path = tmp_path / "articles.jsonl"
        _write_jsonl(articles_path, [_ARTICLE_ROW])

        written = build_article_breadcrumb_arm(
            qdrant, _spying_embed([]), articles_path=articles_path, arm="articles__test"
        )

        assert written == 1
        assert qdrant.count("articles__test").count == 1

    def test_dense_half_comes_from_the_breadcrumb_and_sparse_half_from_the_raw_chunk(
        self, qdrant: QdrantClient, tmp_path: Path
    ) -> None:
        articles_path = tmp_path / "articles.jsonl"
        _write_jsonl(articles_path, [_ARTICLE_ROW])
        calls: list[list[str]] = []

        build_article_breadcrumb_arm(
            qdrant, _spying_embed(calls), articles_path=articles_path, arm="articles__test"
        )

        assert len(calls) == 2  # dense_text's call, then the raw-text call — upsert.py's seam
        [dense_call], [sparse_call] = calls
        assert str(_ARTICLE_ROW["fullSectionsTitre"]) in dense_call
        assert str(_ARTICLE_ROW["fullSectionsTitre"]) not in sparse_call

    def test_stored_text_payload_stays_the_raw_chunk(self, qdrant: QdrantClient, tmp_path: Path) -> None:
        articles_path = tmp_path / "articles.jsonl"
        _write_jsonl(articles_path, [_ARTICLE_ROW])

        build_article_breadcrumb_arm(
            qdrant, _spying_embed([]), articles_path=articles_path, arm="articles__test"
        )

        point_id = article_point_id(str(_ARTICLE_ROW["cid"]), 0)
        [point] = qdrant.retrieve("articles__test", ids=[point_id], with_payload=True)
        assert point.payload is not None
        assert str(_ARTICLE_ROW["fullSectionsTitre"]) not in str(point.payload["text"])

    def test_default_arm_name_matches_the_named_constant(self, qdrant: QdrantClient, tmp_path: Path) -> None:
        articles_path = tmp_path / "articles.jsonl"
        _write_jsonl(articles_path, [_ARTICLE_ROW])

        build_article_breadcrumb_arm(qdrant, _spying_embed([]), articles_path=articles_path)

        assert qdrant.collection_exists(ARTICLE_BREADCRUMB_ARM)


class TestBuildFicheHeaderArm:
    def test_creates_the_named_arm_and_writes_one_point_per_chunk(
        self, qdrant: QdrantClient, tmp_path: Path
    ) -> None:
        fiches_dir = tmp_path / "fiches"
        fiches_dir.mkdir()
        (fiches_dir / "F1.xml").write_bytes(_FICHE_XML)

        written = build_fiche_header_arm(qdrant, _spying_embed([]), fiches_dir=fiches_dir, arm="fiches__test")

        assert written == 1
        assert qdrant.count("fiches__test").count == 1

    def test_dense_half_comes_from_the_title_and_sparse_half_from_the_raw_chunk(
        self, qdrant: QdrantClient, tmp_path: Path
    ) -> None:
        fiches_dir = tmp_path / "fiches"
        fiches_dir.mkdir()
        (fiches_dir / "F1.xml").write_bytes(_FICHE_XML)
        calls: list[list[str]] = []

        build_fiche_header_arm(qdrant, _spying_embed(calls), fiches_dir=fiches_dir, arm="fiches__test")

        assert len(calls) == 2
        [dense_call], [sparse_call] = calls
        assert "Modification du contrat d'assurance habitation" in dense_call
        assert "Modification du contrat d'assurance habitation" not in sparse_call

    def test_stored_text_payload_stays_the_raw_chunk(self, qdrant: QdrantClient, tmp_path: Path) -> None:
        fiches_dir = tmp_path / "fiches"
        fiches_dir.mkdir()
        (fiches_dir / "F1.xml").write_bytes(_FICHE_XML)

        build_fiche_header_arm(qdrant, _spying_embed([]), fiches_dir=fiches_dir, arm="fiches__test")

        point_id = fiche_point_id("F1", 0)
        [point] = qdrant.retrieve("fiches__test", ids=[point_id], with_payload=True)
        assert point.payload is not None
        assert not str(point.payload["text"]).startswith("Modification du contrat")

    def test_default_arm_name_matches_the_named_constant(self, qdrant: QdrantClient, tmp_path: Path) -> None:
        fiches_dir = tmp_path / "fiches"
        fiches_dir.mkdir()
        (fiches_dir / "F1.xml").write_bytes(_FICHE_XML)

        build_fiche_header_arm(qdrant, _spying_embed([]), fiches_dir=fiches_dir)

        assert qdrant.collection_exists(FICHE_HEADER_ARM)
