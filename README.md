# rag_assurances

A French-language RAG assistant over public French insurance law — service-public.fr consumer
fiches plus the in-force Code des assurances. It answers a curious consumer's question in two
parts: an explanation in consumer French, and the *fondement juridique* citing the article behind
it.

**Status: design complete; the skeleton and dev harness are in place, no pipeline stage is built
yet.** Build order is [SPEC §20](SPEC.md).

| | |
|---|---|
| [`SPEC.md`](SPEC.md) | The full specification — corpus, chunking, embeddings, vector store, retrieval, generation, eval, interface, deployment. Detailed enough to build from. |
| [`CONTEXT.md`](CONTEXT.md) | The project's domain vocabulary. |
| [`docs/adr/`](docs/adr/) | Fourteen architecture decision records. |
| [`docs/research/`](docs/research/) | Primary-source research notes on corpora, French embedding models, and Langfuse. |
| [Map #1](https://github.com/Zameloth/rag_assurances/issues/1) | The wayfinding map and its sixteen decision tickets, where the full reasoning lives. |

Stack: LangChain · BGE-M3 · Qdrant · Mistral Large 3 via OpenRouter · Langfuse · FastAPI + HTMX.

## Getting started

```sh
cp .env.example .env     # then fill in OPENROUTER_API_KEY
make install             # uv venv + the package and its dev group
make up                  # dev Qdrant, reachable at QDRANT_URL
make check               # mypy, then the test suite
```

`make` on its own lists every target. The pipeline targets — `ingest`, `ladder`, `publish-index`,
`deploy` — are named now and stubbed until their ticket lands; each exits non-zero and prints the
command it will run.

The suite is green without a live store: anything needing the real engine takes the `qdrant_server`
fixture and skips when nothing answers at `QDRANT_URL`. Everything else uses `QdrantClient(":memory:")`,
which is for plumbing assertions only — never for recall or ranking numbers ([SPEC §6.3](SPEC.md)).

Configuration is read from `.env` and nowhere else ([SPEC §16.3](SPEC.md)); `.env.example` is the
committed template and documents every variable. `LANGFUSE_TRACING` is **off by default** — the free
tier fails on interactive debugging long before it fails on the ladder ([SPEC §11.2](SPEC.md)).
