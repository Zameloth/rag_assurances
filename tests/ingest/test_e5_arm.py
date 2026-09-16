"""SPEC §5/§6.4, ADR-0004, #39 — rung 6's e5-dense + M3-sparse arm builders.

Small fixture corpora, not the real committed one — `test_ingest_pipeline.py` already pays
that cost for the shared chunker/upsert path. This file is only about the same chunk
population `run_ingest` builds reaching `ensure_articles_collection`/`ensure_fiches_collection`
and `upsert_articles`/`upsert_fiches` under the rung-6 arm names, with no re-chunking and no
`dense_text` seam (that seam is the pre-ladder A/Bs' — rung 6 changes the embedder, not the
embedded text).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from qdrant_client import QdrantClient, models

from rag.ingest.e5_arm import (
    ARTICLES_E5_ARM,
    FICHES_E5_ARM,
    build_articles_e5_arm,
    build_fiches_e5_arm,
)
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

_DENSE_DIM = 1024


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _spying_embed(calls: list[list[str]]) -> EmbedFn:
    def embed(texts: Sequence[str]) -> list[Embedding]:
        calls.append(list(texts))
        return [
            ([1.0] * _DENSE_DIM, models.SparseVector(indices=[i], values=[1.0]))
            for i in range(len(texts))
        ]

    return embed


class TestBuildArticlesE5Arm:
    def test_creates_the_named_arm_and_writes_one_point_per_chunk(
        self, qdrant: QdrantClient, tmp_path: Path
    ) -> None:
        articles_path = tmp_path / "articles.jsonl"
        _write_jsonl(articles_path, [_ARTICLE_ROW])

        written = build_articles_e5_arm(
            qdrant, _spying_embed([]), articles_path=articles_path, arm="articles__test"
        )

        assert written == 1
        assert qdrant.count("articles__test").count == 1

    def test_embed_is_called_once_on_the_raw_chunk_no_dense_text_seam(
        self, qdrant: QdrantClient, tmp_path: Path
    ) -> None:
        articles_path = tmp_path / "articles.jsonl"
        _write_jsonl(articles_path, [_ARTICLE_ROW])
        calls: list[list[str]] = []

        build_articles_e5_arm(
            qdrant, _spying_embed(calls), articles_path=articles_path, arm="articles__test"
        )

        assert len(calls) == 1  # one embed call, unlike the pre-ladder A/Bs' dense_text seam

    def test_stored_text_payload_stays_the_raw_chunk(self, qdrant: QdrantClient, tmp_path: Path) -> None:
        articles_path = tmp_path / "articles.jsonl"
        _write_jsonl(articles_path, [_ARTICLE_ROW])

        build_articles_e5_arm(
            qdrant, _spying_embed([]), articles_path=articles_path, arm="articles__test"
        )

        point_id = article_point_id(str(_ARTICLE_ROW["cid"]), 0)
        [point] = qdrant.retrieve("articles__test", ids=[point_id], with_payload=True)
        assert point.payload is not None
        assert "résiliation" in str(point.payload["text"]).lower()

    def test_default_arm_name_matches_the_named_constant(self, qdrant: QdrantClient, tmp_path: Path) -> None:
        articles_path = tmp_path / "articles.jsonl"
        _write_jsonl(articles_path, [_ARTICLE_ROW])

        build_articles_e5_arm(qdrant, _spying_embed([]), articles_path=articles_path)

        assert qdrant.collection_exists(ARTICLES_E5_ARM)

    def test_idempotent_rerun_does_not_duplicate_points(self, qdrant: QdrantClient, tmp_path: Path) -> None:
        articles_path = tmp_path / "articles.jsonl"
        _write_jsonl(articles_path, [_ARTICLE_ROW])

        build_articles_e5_arm(qdrant, _spying_embed([]), articles_path=articles_path, arm="articles__test")
        build_articles_e5_arm(qdrant, _spying_embed([]), articles_path=articles_path, arm="articles__test")

        assert qdrant.count("articles__test").count == 1


class TestBuildFichesE5Arm:
    def test_creates_the_named_arm_and_writes_one_point_per_chunk(
        self, qdrant: QdrantClient, tmp_path: Path
    ) -> None:
        fiches_dir = tmp_path / "fiches"
        fiches_dir.mkdir()
        (fiches_dir / "F1.xml").write_bytes(_FICHE_XML)

        written = build_fiches_e5_arm(qdrant, _spying_embed([]), fiches_dir=fiches_dir, arm="fiches__test")

        assert written == 1
        assert qdrant.count("fiches__test").count == 1

    def test_embed_is_called_once_on_the_raw_chunk_no_dense_text_seam(
        self, qdrant: QdrantClient, tmp_path: Path
    ) -> None:
        fiches_dir = tmp_path / "fiches"
        fiches_dir.mkdir()
        (fiches_dir / "F1.xml").write_bytes(_FICHE_XML)
        calls: list[list[str]] = []

        build_fiches_e5_arm(qdrant, _spying_embed(calls), fiches_dir=fiches_dir, arm="fiches__test")

        assert len(calls) == 1

    def test_stored_text_payload_stays_the_raw_chunk(self, qdrant: QdrantClient, tmp_path: Path) -> None:
        fiches_dir = tmp_path / "fiches"
        fiches_dir.mkdir()
        (fiches_dir / "F1.xml").write_bytes(_FICHE_XML)

        build_fiches_e5_arm(qdrant, _spying_embed([]), fiches_dir=fiches_dir, arm="fiches__test")

        point_id = fiche_point_id("F1", 0)
        [point] = qdrant.retrieve("fiches__test", ids=[point_id], with_payload=True)
        assert point.payload is not None
        assert "modification du contrat" in str(point.payload["text"]).lower()

    def test_default_arm_name_matches_the_named_constant(self, qdrant: QdrantClient, tmp_path: Path) -> None:
        fiches_dir = tmp_path / "fiches"
        fiches_dir.mkdir()
        (fiches_dir / "F1.xml").write_bytes(_FICHE_XML)

        build_fiches_e5_arm(qdrant, _spying_embed([]), fiches_dir=fiches_dir)

        assert qdrant.collection_exists(FICHES_E5_ARM)
