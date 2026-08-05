"""Payload assembly and point ids (SPEC §7.1, §7.2, §6.4, ADR-0005, ADR-0007, #25).

Every field here earns its place by serving one of SPEC §7's four named consumers —
retrieval, the prompt, the app, eval — and nothing else: `register` and `provenance` are
retrieval annotations attached by the retriever (SPEC §7.5), never stored, so they are not
built here.

Point ids are **UUIDv5** over a fixed namespace plus the natural key, computable from the
source document without querying the store — the same natural key a re-run produces, which
is what makes ingestion an idempotent overwrite rather than a second copy of the corpus
(SPEC §6.4). `NAMESPACE` is fixed once, here, for the whole project; `tests/conftest.py`
imports it rather than carrying its own placeholder.
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import models

from rag.ingest.articles import ArticleChunk, ArticleRow
from rag.ingest.fiches import FicheChunk, FicheMetadata
from rag.ingest.lookup_key import normalize_lookup_key

__all__ = [
    "NAMESPACE",
    "article_point_id",
    "fiche_point_id",
    "build_article_payload",
    "build_fiche_payload",
    "build_article_point",
    "build_fiche_point",
]

# Derived rather than a bare random UUID, so the namespace is reproducible from source
# alone — anyone re-reading this file gets the exact same constant, with no separate value
# to keep in sync anywhere else.
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "rag_assurances.qdrant")


def article_point_id(legiarti_cid: str, chunk_index: int) -> str:
    """`uuid5(NS, f"{legiarti_cid}#{chunk_index}")` — SPEC §6.4. `cid`, never the version id:
    an amended article's version id would hash to a different point on every amendment and
    leave the stale version behind as a retrievable orphan (ADR-0005)."""
    return str(uuid.uuid5(NAMESPACE, f"{legiarti_cid}#{chunk_index}"))


def fiche_point_id(fiche_id: str, chunk_index: int) -> str:
    """`uuid5(NS, f"{fiche_id}#{chunk_index}")` — SPEC §6.4."""
    return str(uuid.uuid5(NAMESPACE, f"{fiche_id}#{chunk_index}"))


def build_article_payload(row: ArticleRow, chunk: ArticleChunk) -> dict[str, Any]:
    """SPEC §7.2, field for field. `lookup_key` is computed here, not carried by `row` —
    it is an ingest-time transform of `citation_id`, null for the 21 prose annexe labels."""
    return {
        "legiarti_cid": row["cid"],
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "citation_id": row["citation_id"],
        "lookup_key": normalize_lookup_key(row["citation_id"]),
        "section_id": row.get("sectionParentId"),
        "full_sections_titre": row.get("fullSectionsTitre"),
        "legiarti_version_id": row["id"],
        "date_debut": row.get("dateDebut"),
        "etat": row["etat"],
    }


def build_fiche_payload(meta: FicheMetadata, chunk: FicheChunk) -> dict[str, Any]:
    """SPEC §7.1, field for field. `section_ids` is read by expansion, never filtered on —
    hence no payload index on it, unlike the article side's `section_id`."""
    return {
        "fiche_id": meta.fiche_id,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "title": meta.title,
        "chapitre_titre": chunk.chapitre_titre,
        "cas_label": chunk.cas_label,
        "section_ids": meta.section_ids,
        "sp_url": meta.sp_url,
        "date_modified": meta.date_modified,
        "fil_ariane": meta.fil_ariane,
        "type": meta.fiche_type,
    }


def build_article_point(
    row: ArticleRow, chunk: ArticleChunk, dense: list[float], sparse: models.SparseVector
) -> models.PointStruct:
    """One `articles` point: UUIDv5 id, dense + sparse as two named vectors kept in lockstep
    by construction (SPEC §6.4), flat payload."""
    return models.PointStruct(
        id=article_point_id(row["cid"], chunk.chunk_index),
        vector={"dense": dense, "sparse": sparse},
        payload=build_article_payload(row, chunk),
    )


def build_fiche_point(
    meta: FicheMetadata, chunk: FicheChunk, dense: list[float], sparse: models.SparseVector
) -> models.PointStruct:
    """One `fiches` point — the fiche analogue of `build_article_point`."""
    return models.PointStruct(
        id=fiche_point_id(meta.fiche_id, chunk.chunk_index),
        vector={"dense": dense, "sparse": sparse},
        payload=build_fiche_payload(meta, chunk),
    )
