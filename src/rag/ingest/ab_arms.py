"""Challenger-arm construction for the two pre-ladder A/Bs (SPEC §12.8, ADR-0005, #38).

`rag.ingest.pipeline.run_ingest` builds the two default, unenriched arms every ladder rung
and the app read through the stable aliases (`ARTICLES_ARM`/`FICHES_ARM`). Each pre-ladder
A/B needs one more, non-default arm alongside that: a **challenger** collection embedded
with `rag.ingest.enrichment`'s dense-text transform, built from the exact same chunk
population as the incumbent — SPEC §12.8's own acceptance criterion ("no re-chunking — the
point ids and therefore the gold labels are untouched"). Same chunker, same
`row`/`chunk`-derived UUIDv5 ids, same corpus commit; only which text the *dense* half of
each embedding call sees differs.

Neither builder here flips a stable alias — that is the eval harness's job
(`scripts/run_ab_pilot.py`), done deliberately once a verdict is in, never as a side effect
of building a collection nobody has judged yet.
"""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient

from rag.ingest.arms import ensure_articles_collection, ensure_fiches_collection
from rag.ingest.enrichment import enrich_article_dense_text, enrich_fiche_dense_text
from rag.ingest.pipeline import DEFAULT_ARTICLES_PATH, DEFAULT_FICHES_DIR
from rag.ingest.refresh_diff import load_jsonl
from rag.ingest.upsert import EmbedFn, upsert_articles, upsert_fiches

__all__ = [
    "ARTICLE_BREADCRUMB_ARM",
    "FICHE_HEADER_ARM",
    "build_article_breadcrumb_arm",
    "build_fiche_header_arm",
]

# SPEC §6.4's arm-naming convention (`<register>__<embedder>__<chunk config>__<version>`),
# with the enrichment flag as an extra segment `rag.ingest.pipeline.ARTICLES_ARM`/
# `FICHES_ARM` don't carry — these are distinct arms, not a replacement for either.
ARTICLE_BREADCRUMB_ARM = "articles__m3__c512__breadcrumb-dense__v1"
FICHE_HEADER_ARM = "fiches__m3__c512__header-dense__v1"


def build_article_breadcrumb_arm(
    client: QdrantClient,
    embed: EmbedFn,
    *,
    articles_path: Path = DEFAULT_ARTICLES_PATH,
    arm: str = ARTICLE_BREADCRUMB_ARM,
) -> int:
    """SPEC §12.8's article-breadcrumb challenger: dense vectors embed `fullSectionsTitre`
    ahead of the chunk, sparse vectors embed the raw chunk (`enrich_article_dense_text`,
    `upsert_articles`'s `dense_text` seam). Idempotent the same way `run_ingest` is —
    `ensure_articles_collection` is a no-op on an arm that already exists, `upsert_articles`
    overwrites by UUIDv5 point id — so re-running this against an unchanged corpus commit
    writes the same points twice rather than accumulating a second copy of the arm.

    Assumes the corpus already cleared `rag.ingest.pipeline.run_ingest`'s ingest-assertion
    gate for this commit (both challenger builders read the same committed
    `articles_path`/`fiches_dir` the incumbent arm was built from) — SPEC §12.8 does not ask
    for a second gate on data already gated once for the same commit.
    """
    rows = load_jsonl(articles_path)
    ensure_articles_collection(client, arm)
    return upsert_articles(client, arm, rows, embed, dense_text=enrich_article_dense_text)


def build_fiche_header_arm(
    client: QdrantClient,
    embed: EmbedFn,
    *,
    fiches_dir: Path = DEFAULT_FICHES_DIR,
    arm: str = FICHE_HEADER_ARM,
) -> int:
    """The fiche analogue of `build_article_breadcrumb_arm` — SPEC §12.8's fiche-header
    challenger (`title` · `chapitre_titre` · `cas_label` ahead of the chunk, dense-only)."""
    fiche_paths = sorted(fiches_dir.glob("*.xml"))
    fiche_bytes = [path.read_bytes() for path in fiche_paths]
    ensure_fiches_collection(client, arm)
    return upsert_fiches(client, arm, fiche_bytes, embed, dense_text=enrich_fiche_dense_text)
