"""Rung 6's index-bearing arm: e5-dense + M3-sparse, on the incumbent's own chunk
population (SPEC §5, §6.4, §12.7, ADR-0004, #39).

**No re-chunking.** `build_articles_e5_arm`/`build_fiches_e5_arm` read the exact same
`articles_path`/`fiches_dir` `rag.ingest.pipeline.run_ingest` builds `ARTICLES_ARM`/
`FICHES_ARM` from, and chunk them with the same `chunk_article`/`chunk_fiche` calls
`upsert_articles`/`upsert_fiches` already make. Per-arm chunking would un-share the point
ids the gold labels depend on (SPEC §4) — sizing to e5's own 512-token limit rather than
BGE-M3's 8192 window is exactly why the band is 512 for every arm, not just this one.

**No `dense_text` seam.** The two pre-ladder A/Bs (`rag.ingest.ab_arms`, #38) hold the
embedder fixed and change what text the dense half sees; rung 6 holds the text fixed and
changes the embedder. `upsert_articles`/`upsert_fiches` are called with `dense_text=None`
(their default) — one `embed` call per collection, over the raw chunk text, same as the
default arms. Which model actually answers that call is entirely `embed`'s business:
`rag.ingest.e5_embedder.index_embed_batch` (dense from e5's passage side, sparse from
BGE-M3) at build time, `query_embed_batch` (dense from e5's query side) at retrieval time —
this module never imports either, so a caller free to inject any `EmbedFn` here for tests.

**Reachable only by alias flip.** Like `rag.ingest.ab_arms`, no builder here touches
`ARTICLES_ALIAS`/`FICHES_ALIAS` — that is the eval harness's job (SPEC §12.7's own
"rung 6 wins" is a verdict read after both incumbent and challenger arms have run, never a
side effect of building the challenger).
"""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient

from rag.ingest.arms import ensure_articles_collection, ensure_fiches_collection
from rag.ingest.pipeline import DEFAULT_ARTICLES_PATH, DEFAULT_FICHES_DIR
from rag.ingest.refresh_diff import load_jsonl
from rag.ingest.upsert import EmbedFn, upsert_articles, upsert_fiches

__all__ = [
    "ARTICLES_E5_ARM",
    "FICHES_E5_ARM",
    "build_articles_e5_arm",
    "build_fiches_e5_arm",
]

# SPEC §6.4's arm-naming convention (`<register>__<embedder>__<chunk config>__<version>`) —
# `e5-m3` names the hybrid pair itself (e5-dense + M3-sparse, ADR-0004), the same way
# `ARTICLES_ARM`'s `m3` names BGE-M3 supplying both halves.
ARTICLES_E5_ARM = "articles__e5-m3__c512__v1"
FICHES_E5_ARM = "fiches__e5-m3__c512__v1"


def build_articles_e5_arm(
    client: QdrantClient,
    embed: EmbedFn,
    *,
    articles_path: Path = DEFAULT_ARTICLES_PATH,
    arm: str = ARTICLES_E5_ARM,
) -> int:
    """Build (or idempotently rebuild) the rung-6 `articles` arm. `embed` is expected to be
    `rag.ingest.e5_embedder.index_embed_batch` in production — a real `EmbedFn`, not a
    special-cased signature, so the fixture embedders `tests/ingest/test_e5_arm.py` and
    `rag.ingest.ab_arms`'s own tests already use work here unchanged.

    Assumes the corpus already cleared `run_ingest`'s ingest-assertion gate for this commit,
    the same precondition `rag.ingest.ab_arms`'s builders state for the same reason: both
    read the committed `articles_path` the incumbent arm was built from, gated once per
    commit, not once per arm.
    """
    rows = load_jsonl(articles_path)
    ensure_articles_collection(client, arm)
    return upsert_articles(client, arm, rows, embed)


def build_fiches_e5_arm(
    client: QdrantClient,
    embed: EmbedFn,
    *,
    fiches_dir: Path = DEFAULT_FICHES_DIR,
    arm: str = FICHES_E5_ARM,
) -> int:
    """The fiche analogue of `build_articles_e5_arm`."""
    fiche_paths = sorted(fiches_dir.glob("*.xml"))
    fiche_bytes = [path.read_bytes() for path in fiche_paths]
    ensure_fiches_collection(client, arm)
    return upsert_fiches(client, arm, fiche_bytes, embed)
