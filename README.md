# rag_assurances

A French-language RAG assistant over public French insurance law — service-public.fr consumer
fiches plus the in-force Code des assurances. It answers a curious consumer's question in two
parts: an explanation in consumer French, and the *fondement juridique* citing the article behind
it.

**Status: corpus and the full ingest pipeline are built — `make ingest` runs corpus → chunks →
BGE-M3 → Qdrant end to end. Retrieval is under way (the short-circuit has landed); generation, eval
and the app have not started.** Build order is [SPEC §20](SPEC.md); the section below says where it
stands.

| | |
|---|---|
| [`SPEC.md`](SPEC.md) | The full specification — corpus, chunking, embeddings, vector store, retrieval, generation, eval, interface, deployment. Detailed enough to build from. |
| [`CONTEXT.md`](CONTEXT.md) | The project's domain vocabulary. |
| [`CODING_STANDARDS.md`](CODING_STANDARDS.md) | The house rules a review checks against, consolidated from `pyproject.toml`, `CONTEXT.md` and the ADRs. |
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

`make` on its own lists every target. Of the four pipeline targets, `ingest` is real — it needs
`make up` first, and downloads the BGE-M3 weights into gitignored `data/raw/` on its first run.
The remaining three — `ladder`, `publish-index`, `deploy` — are named now and stubbed until their
ticket lands; each exits non-zero and prints the command it will run.

The suite is green without a live store: anything needing the real engine takes the `qdrant_server`
fixture and skips when nothing answers at `QDRANT_URL`. Everything else uses `QdrantClient(":memory:")`,
which is for plumbing assertions only — never for recall or ranking numbers ([SPEC §6.3](SPEC.md)).

## Where the build stands

Against the twelve steps of [SPEC §20](SPEC.md):

| Step | | |
|---|---|---|
| 1 | Corpus | **done** — 2,375 articles + 87 fiches committed, assertions 1–5 green |
| 2 | Ingest library | **done** — `make ingest` runs chunk → payload → BGE-M3 → Qdrant, assertions 6–9 green |
| 3 | Retrieval library | **in progress** — the article-reference short-circuit has landed; the retriever itself and the four ladder arms have not |
| 4 | Golden set | tooling built (schema, corpus validator, annotation helper); **not a single item annotated yet** |
| 5–12 | Eval harness, ladder, generation, condenser, app, deploy | not started |

What that means concretely:

- **Ingest is end-to-end.** `rag.ingest.pipeline` reads the committed corpus, gates it with ingest
  assertions 1–9, embeds every chunk with BGE-M3 (dense + learned sparse in one forward pass),
  upserts into a versioned arm collection and flips the stable alias onto it ([ADR-0005](docs/adr/0005-qdrant-two-collections-behind-aliases.md)).
- **It writes 3,650 points** — 2,801 article chunks + 849 fiche chunks. SPEC §4.4 estimated 3,687;
  the gap is a known, accepted difference between the estimate and what the committed chunkers
  actually measure, documented at `rag.ingest.fiches` and the two corpus-chunking test modules, not
  an open defect.
- **Retrieval has one piece.** `rag.retrieval.short_circuit` resolves a query carrying an article
  reference against `lookup_key` without searching at all ([SPEC §9.1](SPEC.md)) — one normalizer,
  two patterns. Nothing yet consumes it: there is no `BaseRetriever`, no hybrid legs, no fusion.
- **The golden set is empty.** `eval/golden/` holds only its `.gitkeep`. Annotation is
  [#34](https://github.com/Zameloth/rag_assurances/issues/34), a human task, and it blocks the whole
  eval track — #34 → #35 → #36 → #37 → the rungs. No rung can run without it.
- **One gate ahead is one-way.** The pre-registered rung/metric table
  ([#37](https://github.com/Zameloth/rag_assurances/issues/37), step 6) has to be committed *before
  a single rung runs*; written afterwards it is post-hoc metric selection, and no later commit
  repairs that ([SPEC §12.7](SPEC.md)). Nothing else in the build order is irrecoverable.

Next up, both unblocked: [#28](https://github.com/Zameloth/rag_assurances/issues/28) — the retriever
skeleton, the rung-1 baseline — and #34.

The suite is **254 passed, 7 skipped** (the skips are the ones needing a live Qdrant), at 96% line
coverage against a 90% floor.

## Corpus

Two committed pieces, ~10 MB together:

- **[`data/corpus/articles.jsonl`](data/corpus/articles.jsonl)** — the **2,375 in-force Code des
  assurances articles** (L, R, A and D), one JSON object per line, sorted by `cid`, ~8.3 MB. Every
  row carries both `texte` and `texteHtml` — `texte` has zero newlines corpus-wide, so `texteHtml`
  is the only place paragraph structure survives, and chunking will need it (SPEC §3.2, §4.2).
  This is 2 short of the source snapshot's own 2,377-row total: 2 rows carry `etat ==
  "ABROGE_DIFF"` (repealed with deferred effect — not yet lapsed today, but not the literal
  `"VIGUEUR"` ingest assertion 2 requires) and are dropped by the fetch script rather than
  loosening the assertion.
- **[`data/corpus/fiches/`](data/corpus/fiches/)** — **87 service-public.fr consumer fiches**
  (`F*.xml`), verbatim DILA XML under their original filenames, ~1.9 MB.

Producer, licence and download-provenance attribution for both live in
**[`data/corpus/corpus_manifest.json`](data/corpus/corpus_manifest.json)** — script-emitted, never
hand-maintained, so it cannot drift from what was actually fetched.

**Scope is a rule, not a list**: in scope = any service-public.fr insurance fiche whose
`<dc:source>` resolves into the Code des assurances, plus the in-force Code des assurances itself
([SPEC §1.2](SPEC.md)). Mechanically: each fiche's `<dc:source>` `LEGISCTA` ids intersected with
the distinct `sectionParentId` values across the in-force articles — code, not a hand-picked list,
so `scripts/fetch_fiches.py` reproduces the same 87 documents from the committed articles every
time it runs. It keeps auto, habitation and vie; it drops pure *assurance maladie*, whose fiches
rest on the Code de la sécurité sociale and would be structurally ungrounded.

The corpus is **public-sector information only** — no insurer *conditions générales* or other
contract documents are in here, or ever will be; that was a deliberate scoping call
([SPEC §1.6](SPEC.md)), not an oversight.

**Refresh** re-runs the fetch scripts by hand — dev tools, never part of the build path — and
commits the result as a **reviewed diff**, never a scheduled job. Each run reports three counts
before writing anything — added / removed / **changed-text-under-the-same-`cid`** (or `fiche_id`
for fiches) — and exits loudly rather than write if it would break the fiche→section join
([SPEC §3.3](SPEC.md)). Fiches must refresh *after* articles, since the scope rule reads whatever
`articles.jsonl` is on disk:

```sh
uv run --group fetch python scripts/fetch_articles.py
uv run --group fetch python scripts/fetch_fiches.py
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
