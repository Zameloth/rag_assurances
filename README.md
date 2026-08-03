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

## Corpus

**[`data/corpus/articles.jsonl`](data/corpus/articles.jsonl)** — the **2,375 in-force Code des
assurances articles** (L, R, A and D), one JSON object per line, sorted by `cid`, ~8.3 MB. Every
row carries both `texte` and `texteHtml` — `texte` has zero newlines corpus-wide, so `texteHtml`
is the only place paragraph structure survives, and chunking will need it (SPEC §3.2, §4.2). The
87 service-public.fr consumer fiches that pair with it are a separate, not-yet-landed ticket.

This is 2 short of the source snapshot's own 2,377-row total: 2 rows carry `etat ==
"ABROGE_DIFF"` (repealed with deferred effect — not yet lapsed today, but not the literal
`"VIGUEUR"` ingest assertion 2 requires) and are dropped by the fetch script rather than
loosening the assertion.

Producer, licence and download-provenance attribution live in
**[`data/corpus/corpus_manifest.json`](data/corpus/corpus_manifest.json)** — script-emitted, never
hand-maintained, so it cannot drift from what was actually fetched.

**Scope is a rule, not a list**: in scope = any service-public.fr insurance fiche whose
`<dc:source>` resolves into the Code des assurances, plus the in-force Code des assurances itself
([SPEC §1.2](SPEC.md)). It is mechanical and reproducible, and it is why the corpus is
**public-sector information only** — no insurer *conditions générales* or other contract documents
are in here, or ever will be; that was a deliberate scoping call ([SPEC §1.6](SPEC.md)), not an
oversight.

**Refresh** re-runs `scripts/fetch_articles.py` by hand — a dev tool, never part of the build path
— and commits the result as a **reviewed diff**, never a scheduled job:

```sh
uv run --group fetch python scripts/fetch_articles.py
```

The repo commits **sources and decisions, never derived binaries** ([SPEC §16.5](SPEC.md)). The two
named exceptions to that principle live under `eval/`: the hand-annotated golden set (not derived
at all) and the per-item eval scores (tiny, and their whole purpose is outliving Langfuse's 30-day
retention).

## Licence

Two licences, because one root file would be a false statement — you can license your code,
you cannot relicense the Code des assurances ([SPEC §16.2](SPEC.md)).

- [`LICENSE`](LICENSE) — MIT, and it covers **the code only**.
- The corpus under `data/corpus/` is public-sector information from **DILA**, redistributed under
  **[Licence Ouverte 2.0](data/corpus/LICENSE.md)** — always, regardless of what licence label a
  third-party mirror declares on its own dataset card. `corpus_manifest.json` is the authoritative
  machine-readable attribution record.

## Notes

Configuration is read from `.env` and nowhere else ([SPEC §16.3](SPEC.md)); `.env.example` is the
committed template and documents every variable. `LANGFUSE_TRACING` is **off by default** — the free
tier fails on interactive debugging long before it fails on the ladder ([SPEC §11.2](SPEC.md)).
