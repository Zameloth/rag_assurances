"""SPEC §7.1 / §7.2 / §6.4 / #25 — payload assembly and point ids."""

import uuid

from qdrant_client import models

from rag.ingest.articles import ArticleChunk
from rag.ingest.fiches import FicheChunk, FicheMetadata
from rag.ingest.payload import (
    NAMESPACE,
    article_point_id,
    build_article_payload,
    build_article_point,
    build_fiche_payload,
    build_fiche_point,
    fiche_point_id,
)

ARTICLE_ROW = {
    "cid": "LEGIARTI000006785773",
    "id": "LEGIARTI000036754138",
    "citation_id": "L113-3",
    "etat": "VIGUEUR",
    "dateDebut": "2018-04-01",
    "sectionParentId": "LEGISCTA000006156957",
    "fullSectionsTitre": "Partie réglementaire > Livre Ier : Le contrat",
}

ARTICLE_CHUNK = ArticleChunk(text="Le contrat est régi par les présentes dispositions.", chunk_index=0, tokens=12, is_stub=False)

FICHE_META = FicheMetadata(
    fiche_id="F2594",
    section_ids=["LEGISCTA000006157200", "LEGISCTA000006158221"],
    title="Modification du contrat d'assurance habitation",
    sp_url="https://www.service-public.gouv.fr/particuliers/vosdroits/F2594",
    date_modified="2025-04-28",
    fil_ariane="Accueil particuliers > Assurance habitation > Modification",
    fiche_type="Fiche d'information conditionnée",
)

FICHE_CHUNK = FicheChunk(
    text="La modification du contrat peut être demandée par l'assuré.",
    chunk_index=0,
    tokens=13,
    chapitre_titre="Modification à l'initiative de l'assuré",
    cas_label=None,
)


def test_article_point_id_is_uuid5_over_cid_and_chunk_index() -> None:
    expected = str(uuid.uuid5(NAMESPACE, "LEGIARTI000006785773#0"))
    assert article_point_id("LEGIARTI000006785773", 0) == expected


def test_article_point_id_changes_with_chunk_index() -> None:
    """SPEC §6.4 — chunk_index is load-bearing: 289 articles produce 2+ points."""
    assert article_point_id("LEGIARTI000006785773", 0) != article_point_id("LEGIARTI000006785773", 1)


def test_article_point_id_is_stable_across_calls() -> None:
    """SPEC §6.4 — idempotent ingestion depends on this being deterministic, not random."""
    assert article_point_id("LEGIARTI000006785773", 0) == article_point_id("LEGIARTI000006785773", 0)


def test_fiche_point_id_is_uuid5_over_fiche_id_and_chunk_index() -> None:
    expected = str(uuid.uuid5(NAMESPACE, "F2594#0"))
    assert fiche_point_id("F2594", 0) == expected


def test_build_article_payload_matches_spec_field_for_field() -> None:
    payload = build_article_payload(ARTICLE_ROW, ARTICLE_CHUNK)
    assert payload == {
        "legiarti_cid": "LEGIARTI000006785773",
        "chunk_index": 0,
        "text": "Le contrat est régi par les présentes dispositions.",
        "citation_id": "L113-3",
        "lookup_key": "L113-3",
        "section_id": "LEGISCTA000006156957",
        "full_sections_titre": "Partie réglementaire > Livre Ier : Le contrat",
        "legiarti_version_id": "LEGIARTI000036754138",
        "date_debut": "2018-04-01",
        "etat": "VIGUEUR",
    }


def test_build_article_payload_lookup_key_is_null_for_a_prose_annexe_label() -> None:
    """SPEC §7.3 — the 21 prose annexe labels must never collide with a real article number."""
    row = {**ARTICLE_ROW, "citation_id": "Annexe à l'article A121-1"}
    payload = build_article_payload(row, ARTICLE_CHUNK)
    assert payload["citation_id"] == "Annexe à l'article A121-1"
    assert payload["lookup_key"] is None


def test_build_fiche_payload_matches_spec_field_for_field() -> None:
    payload = build_fiche_payload(FICHE_META, FICHE_CHUNK)
    assert payload == {
        "fiche_id": "F2594",
        "chunk_index": 0,
        "text": "La modification du contrat peut être demandée par l'assuré.",
        "title": "Modification du contrat d'assurance habitation",
        "chapitre_titre": "Modification à l'initiative de l'assuré",
        "cas_label": None,
        "section_ids": ["LEGISCTA000006157200", "LEGISCTA000006158221"],
        "sp_url": "https://www.service-public.gouv.fr/particuliers/vosdroits/F2594",
        "date_modified": "2025-04-28",
        "fil_ariane": "Accueil particuliers > Assurance habitation > Modification",
        "type": "Fiche d'information conditionnée",
    }


def test_build_article_point_carries_dense_and_sparse_named_vectors() -> None:
    sparse = models.SparseVector(indices=[1, 7], values=[0.9, 0.4])
    point = build_article_point(ARTICLE_ROW, ARTICLE_CHUNK, [1.0, 0.0], sparse)
    assert point.id == article_point_id("LEGIARTI000006785773", 0)
    assert point.vector == {"dense": [1.0, 0.0], "sparse": sparse}
    assert point.payload == build_article_payload(ARTICLE_ROW, ARTICLE_CHUNK)


def test_build_fiche_point_carries_dense_and_sparse_named_vectors() -> None:
    sparse = models.SparseVector(indices=[2], values=[0.5])
    point = build_fiche_point(FICHE_META, FICHE_CHUNK, [0.0, 1.0], sparse)
    assert point.id == fiche_point_id("F2594", 0)
    assert point.vector == {"dense": [0.0, 1.0], "sparse": sparse}
    assert point.payload == build_fiche_payload(FICHE_META, FICHE_CHUNK)
