"""`python -m rag.generate` — ask a question, get the typed answer envelope back
(SPEC §2, §10, #42).

Retrieval then generation in one CLI: `rag.retrieval.pipeline.retrieve` builds the
contexts, `rag.generation.pipeline.generate` calls the real `GenerateFn`
(`rag.generation.chain.make_generate_fn` — `ChatOpenAI` over OpenRouter, pinned routing,
`with_structured_output`) against them and checks citations. Never repairs: SPEC §10.5's
guardrail outcome prints as-is, exactly what an eval run would see, not the demo-repaired
copy (`rag.generation.citation.repair_for_display`).

Deliberately not the app (SPEC §13, still "empty until the app ticket") — this is
`rag.query`'s counterpart one stage further down the pipeline, same injection seam.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from qdrant_client import QdrantClient

from rag.config import load_settings
from rag.generation.chain import make_generate_fn
from rag.generation.pipeline import GenerateFn, GenerationResult, generate
from rag.generation.schema import Refus
from rag.ingest.upsert import EmbedFn
from rag.retrieval.lookup import load_lookup_keys
from rag.retrieval.pipeline import DEFAULT_RETRIEVAL_ARM, retrieve

__all__ = ["main", "print_generation_result"]


def main(
    argv: Sequence[str] | None = None,
    *,
    client: QdrantClient | None = None,
    embed: EmbedFn | None = None,
    generate_fn: GenerateFn | None = None,
) -> GenerationResult:
    """The `python -m rag.generate` entry point.

    `client`/`embed`/`generate_fn` default to the real Qdrant, the real BGE-M3 embedder and
    the real OpenRouter chain — the same injection seam `rag.query.main` already takes for
    `client`/`embed`, so the test suite never has to hit Qdrant, load BGE-M3 or call
    OpenRouter to exercise this CLI's wiring.
    """
    args = _parse_args(argv)
    settings = load_settings()

    owns_client = client is None
    if client is None:
        client = QdrantClient(settings.qdrant_url)
    if embed is None:
        from rag.ingest.embedder import embed_batch  # deferred: pulls in torch

        embed = embed_batch
    if generate_fn is None:
        generate_fn = make_generate_fn(settings)

    try:
        lookup_keys = load_lookup_keys(client)
        retrieval = retrieve(client, embed, args.question, lookup_keys, arm=args.arm)
        result = generate(args.question, retrieval, generate_fn)
        print_generation_result(result)
        return result
    finally:
        if owns_client:
            client.close()


def print_generation_result(result: GenerationResult) -> None:
    """The typed envelope (SPEC §10.3's four terminal states), then the citation-check
    outcome — unrepaired, exactly what eval sees (SPEC §10.5): fabricated citations are
    named, never silently dropped."""
    envelope = result.envelope

    if isinstance(envelope, Refus):
        print(f"[REFUS] motif={envelope.motif.value}")
    elif envelope.aucun_fondement is not None:
        print("[REPONSE — aucun fondement juridique dans le corpus]")
    else:
        print("[REPONSE]")

    print(envelope.explanation)

    if envelope.fondement_juridique:
        print("\nFondement juridique:")
        for f in envelope.fondement_juridique:
            print(f"  - {f.article_id}: {f.gloss}")

    if not isinstance(envelope, Refus) and envelope.aucun_fondement:
        print(f"\n{envelope.aucun_fondement}")

    if not result.citation_outcome.valid:
        fabricated = ", ".join(sorted(result.citation_outcome.fabricated_ids))
        print(f"\n[GUARDRAIL] fabricated citations (not in retrieved context): {fabricated}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m rag.generate",
        description="Ask a question, get the typed answer envelope back (SPEC §10).",
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
