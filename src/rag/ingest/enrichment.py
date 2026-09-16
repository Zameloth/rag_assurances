"""Embedding-time enrichment for the two pre-ladder A/Bs (SPEC §12.8, #38).

Enrichment prepends document-level context to the text handed to the **dense** half of the
embedder, so a bare chunk that names neither the contract nor the article it sits under
still carries that context into the vector it is found by:

- `enrich_article_dense_text` — the article breadcrumb (`fullSectionsTitre`, verbatim,
  never truncated the way the prompt's display copy is — §10.6's first-segment drop is a
  rendering rule, not an embedding one).
- `enrich_fiche_dense_text` — `title`, `chapitre_titre`, `cas_label`, whichever of the two
  nullable fields this chunk actually carries.

Neither touches the **sparse** half, and neither is ever stored: `rag.ingest.payload`
always writes the raw chunk into `text` (SPEC §7), and `rag.ingest.upsert`'s `dense_text`
seam is the only place either function's output is used. SPEC §12.8's "dilution" argument
is exactly why the split is enforced at that seam rather than here — a naive challenger
would hand this same text to the sparse half too, confounding "does context help dense"
with "does breadcrumb noise hurt sparse".
"""

from __future__ import annotations

from rag.ingest.articles import ArticleChunk, ArticleRow
from rag.ingest.fiches import FicheChunk, FicheMetadata

__all__ = ["enrich_article_dense_text", "enrich_fiche_dense_text"]


def enrich_article_dense_text(row: ArticleRow, chunk: ArticleChunk) -> str:
    """SPEC §12.8's article-breadcrumb challenger. Falls back to the raw chunk text
    unchanged when `fullSectionsTitre` is absent, rather than prepending an empty line —
    the same "nullable field, no-op when null" posture `build_article_payload` already
    takes on this field."""
    breadcrumb = row.get("fullSectionsTitre")
    if not breadcrumb:
        return chunk.text
    return f"{breadcrumb}\n{chunk.text}"


def enrich_fiche_dense_text(meta: FicheMetadata, chunk: FicheChunk) -> str:
    """SPEC §12.8's fiche-header challenger: `title` · `chapitre_titre` · `cas_label`,
    whichever of the latter two this chunk carries — `chapitre_titre`/`cas_label` are
    nullable per chunk (SPEC §7.1), unlike `title`, which every fiche has."""
    header = "\n".join(part for part in (meta.title, chunk.chapitre_titre, chunk.cas_label) if part)
    if not header:
        return chunk.text
    return f"{header}\n{chunk.text}"
