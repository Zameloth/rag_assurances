"""`python -m rag.query` — ask a question, print ranked contexts (SPEC §2, §9, #28).

The command-line half of the rung-1 baseline arm: load `Settings`, connect to Qdrant, load
the `lookup_key` membership set, run `rag.retrieval.pipeline.retrieve`, and print each
returned context with its register, score and provenance — the same fat object the app and
`compare.py` will eventually share (SPEC §2's "importable library, no UI-only logic").

Deliberately not the LangChain-wrapped path: this CLI calls the plain-Python
`rag.retrieval.pipeline.retrieve` directly, the same seam a future `BaseRetriever` subclass
would call from `_get_relevant_documents`. Nothing here changes once that wrapper exists.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from qdrant_client import QdrantClient

from rag.config import load_settings
from rag.ingest.upsert import EmbedFn
from rag.retrieval.lookup import load_lookup_keys
from rag.retrieval.pipeline import DEFAULT_RETRIEVAL_ARM, RetrievalResult, retrieve
from rag.retrieval.short_circuit import ShortCircuitPath

__all__ = ["main", "print_result"]


def main(
    argv: Sequence[str] | None = None,
    *,
    client: QdrantClient | None = None,
    embed: EmbedFn | None = None,
) -> RetrievalResult:
    """The `python -m rag.query` entry point.

    `client`/`embed` default to the real Qdrant and the real BGE-M3 embedder — the same
    injection seam `rag.ingest.pipeline.main` uses, so the test suite never has to load
    BGE-M3 to exercise this CLI's wiring.
    """
    args = _parse_args(argv)
    settings = load_settings()

    owns_client = client is None
    if client is None:
        client = QdrantClient(settings.qdrant_url)
    if embed is None:
        from rag.ingest.embedder import embed_batch  # deferred: pulls in torch

        embed = embed_batch

    try:
        lookup_keys = load_lookup_keys(client)
        result = retrieve(client, embed, args.question, lookup_keys, arm=args.arm)
        print_result(result)
        return result
    finally:
        if owns_client:
            client.close()


def print_result(result: RetrievalResult) -> None:
    """Ranked contexts, one per line block: rank, register, score, provenance, then the
    citation/title identifier and a text preview — everything the acceptance criterion
    ("prints the ranked contexts with their register, score and provenance") asks for."""
    if result.short_circuit_path is ShortCircuitPath.RESOLVED:
        path = result.short_circuit_path.value
        print(f"short-circuit path: {path} (no search, no candidate pools)")
    else:
        pool_sizes = ", ".join(f"{leg}={len(pool)}" for leg, pool in result.candidate_pools.items())
        print(f"short-circuit path: {result.short_circuit_path.value} ({pool_sizes})")

    if not result.contexts:
        print("no contexts returned")
        return

    for rank, candidate in enumerate(result.contexts, start=1):
        provenance = "+".join(sorted(p.value for p in candidate.provenance))
        identifier = candidate.payload.get("citation_id") or candidate.payload.get("fiche_id", "?")
        text = str(candidate.payload.get("text", ""))
        preview = text if len(text) <= 160 else text[:157] + "..."
        print(
            f"[{rank}] register={candidate.register.value} score={candidate.score:.4f} "
            f"provenance={provenance} id={identifier}"
        )
        print(f"    {preview}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m rag.query",
        description="Ask a question, get ranked contexts back (SPEC §9 rung-1 baseline).",
    )
    parser.add_argument("question", help="the raw user turn")
    parser.add_argument(
        "--arm",
        default=DEFAULT_RETRIEVAL_ARM,
        help=f"retrieval arm to run (default: {DEFAULT_RETRIEVAL_ARM})",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
