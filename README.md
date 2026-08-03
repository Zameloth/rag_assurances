# rag_assurances

A French-language RAG assistant over public French insurance law — service-public.fr consumer
fiches plus the in-force Code des assurances. It answers a curious consumer's question in two
parts: an explanation in consumer French, and the *fondement juridique* citing the article behind
it.

**Status: design complete, build not started.**

| | |
|---|---|
| [`SPEC.md`](SPEC.md) | The full specification — corpus, chunking, embeddings, vector store, retrieval, generation, eval, interface, deployment. Detailed enough to build from. |
| [`CONTEXT.md`](CONTEXT.md) | The project's domain vocabulary. |
| [`docs/adr/`](docs/adr/) | Fourteen architecture decision records. |
| [`docs/research/`](docs/research/) | Primary-source research notes on corpora, French embedding models, and Langfuse. |
| [Map #1](https://github.com/Zameloth/rag_assurances/issues/1) | The wayfinding map and its sixteen decision tickets, where the full reasoning lives. |

Stack: LangChain · BGE-M3 · Qdrant · Mistral Large 3 via OpenRouter · Langfuse · FastAPI + HTMX.
