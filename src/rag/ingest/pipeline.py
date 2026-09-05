"""`make ingest`: corpus -> chunk -> payload -> BGE-M3 -> Qdrant, end to end (SPEC §4-§7, #26).

`run_ingest` is the orchestration `upsert.py` (#25) left as a seam: read the committed
corpus, gate it with ingest assertions 1-9 (SPEC §4.5, §7.4), embed every chunk in one
batch per collection, upsert into a versioned arm, and flip the stable alias onto it
(ADR-0005). `main` wires that against the real corpus, the real Qdrant and the real
BGE-M3 embedder; every argument is injectable so `run_ingest` and `main` are both tested
against a stub embedder — the point counts below are a property of chunking and upsert,
not of what the embedder returns, so a stub is exactly as informative here as BGE-M3 would
be and several minutes cheaper.

**SPEC.md §4.4 documents 3,687 points (2,805 articles + 882 fiches).** The committed
chunkers (#23/#24) measure **2,801 + 849 = 3,650** against the real corpus — a known,
already-accepted discrepancy (see the module docstrings of `rag.ingest.fiches` and
`tests/test_articles_chunking_corpus.py`/`tests/test_fiches_chunking_corpus.py`), not
something this ticket re-opens. `main` reports whatever the chunkers actually produce.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from rag.config import load_settings
from rag.ingest.arms import (
    ARTICLES_ALIAS,
    FICHES_ALIAS,
    ensure_articles_collection,
    ensure_fiches_collection,
    flip_alias,
)
from rag.ingest.articles import ArticleRow
from rag.ingest.assertions import (
    run_article_assertions,
    run_article_chunk_assertions,
    run_fiche_assertions,
    run_fiche_chunk_assertions,
)
from rag.ingest.fiches import FicheMetadata, parse_fiche_metadata
from rag.ingest.refresh_diff import load_jsonl
from rag.ingest.upsert import EmbedFn, upsert_articles, upsert_fiches

__all__ = ["ARTICLES_ARM", "FICHES_ARM", "IngestReport", "main", "run_ingest"]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTICLES_PATH = REPO_ROOT / "data" / "corpus" / "articles.jsonl"
DEFAULT_FICHES_DIR = REPO_ROOT / "data" / "corpus" / "fiches"

# SPEC §6.4's own example, verbatim for articles: `<register>__<embedder>__<chunk
# config>__<version>`. Bumping either the embedder or the chunk band earns a new arm name,
# never an in-place rewrite of this one — that is what keeps a stable alias meaningful.
ARTICLES_ARM = "articles__m3__c512__v1"
FICHES_ARM = "fiches__m3__c512__v1"


@dataclass(frozen=True)
class IngestReport:
    articles_arm: str
    fiches_arm: str
    articles_points: int
    fiches_points: int

    @property
    def total_points(self) -> int:
        return self.articles_points + self.fiches_points


def run_ingest(
    client: QdrantClient,
    embed: EmbedFn,
    *,
    articles_path: Path = DEFAULT_ARTICLES_PATH,
    fiches_dir: Path = DEFAULT_FICHES_DIR,
    articles_arm: str = ARTICLES_ARM,
    fiches_arm: str = FICHES_ARM,
) -> IngestReport:
    """Read the committed corpus, gate it, embed it and upsert it behind its aliases.

    Idempotent by construction, not by special-casing here: `ensure_*_collection` is a
    no-op on an arm that already exists, `upsert_*` overwrites by UUIDv5 point id
    (ADR-0005), and `flip_alias` is a no-op once the alias already points at `articles_arm`
    / `fiches_arm` — so calling this twice with the same arm names writes the same points
    twice rather than accumulating a second copy of the corpus.
    """
    rows = load_jsonl(articles_path)
    fiche_paths = sorted(fiches_dir.glob("*.xml"))
    fiche_bytes = [path.read_bytes() for path in fiche_paths]
    fiche_meta = [parse_fiche_metadata(xml) for xml in fiche_bytes]

    _gate(rows, fiche_paths, fiche_bytes, fiche_meta)

    ensure_articles_collection(client, articles_arm)
    ensure_fiches_collection(client, fiches_arm)

    articles_written = upsert_articles(client, articles_arm, rows, embed)
    fiches_written = upsert_fiches(client, fiches_arm, fiche_bytes, embed)

    flip_alias(client, ARTICLES_ALIAS, articles_arm)
    flip_alias(client, FICHES_ALIAS, fiches_arm)

    return IngestReport(
        articles_arm=articles_arm,
        fiches_arm=fiches_arm,
        articles_points=articles_written,
        fiches_points=fiches_written,
    )


def _gate(
    rows: Sequence[ArticleRow],
    fiche_paths: Sequence[Path],
    fiche_bytes: Sequence[bytes],
    fiche_meta: Iterable[FicheMetadata],
) -> None:
    """Ingest assertions 1-9 (SPEC §4.5, §7.4), raising `CorpusAssertionError` rather than
    upserting a corpus that fails its own checks — the same posture the fetch scripts take
    before they write (SPEC §3.3)."""
    run_article_assertions(rows)
    run_fiche_assertions(
        ({"fiche_id": meta.fiche_id, "section_ids": meta.section_ids} for meta in fiche_meta),
        rows,
    )
    run_article_chunk_assertions(rows)
    run_fiche_chunk_assertions(zip((path.stem for path in fiche_paths), fiche_bytes, strict=True))


def main(
    *,
    client: QdrantClient | None = None,
    embed: EmbedFn | None = None,
    articles_path: Path = DEFAULT_ARTICLES_PATH,
    fiches_dir: Path = DEFAULT_FICHES_DIR,
) -> IngestReport:
    """The `python -m rag.ingest` entry point behind `make ingest`.

    `client`/`embed` default to the real Qdrant and the real BGE-M3 embedder — never
    exercised by the test suite, which always injects both, the same way `run_ingest` is
    tested against the real committed corpus without a real embedder.
    """
    owns_client = client is None
    if client is None:
        client = QdrantClient(load_settings().qdrant_url)
    if embed is None:
        from rag.ingest.embedder import embed_batch  # deferred: pulls in torch

        embed = embed_batch
    try:
        report = run_ingest(client, embed, articles_path=articles_path, fiches_dir=fiches_dir)
        _print_report(report)
        return report
    finally:
        if owns_client:
            client.close()


def _print_report(report: IngestReport) -> None:
    print(f"articles: {report.articles_points} point(s) -> {report.articles_arm} (alias: articles)")
    print(f"fiches:   {report.fiches_points} point(s) -> {report.fiches_arm} (alias: fiches)")
    print(f"total:    {report.total_points} point(s)")
