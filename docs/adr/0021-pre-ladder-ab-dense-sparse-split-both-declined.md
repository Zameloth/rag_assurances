# ADR-0021 — Embedding-time enrichment is a dense/sparse split behind two challenger arms; both pre-ladder A/Bs declined

- **Status**: Accepted
- **Ticket**: #38
- **Spec**: SPEC.md §7, §12.7, §12.8, ADR-0005, ADR-0011

## Context

SPEC §12.8 names two build-configuration choices to settle **before rung 1**, under the same
adoption rule the ladder itself uses: does prepending the article breadcrumb
(`fullSectionsTitre`) to the *dense* half of an article chunk's embedding help article recall@4,
and does prepending `title`/`chapitre_titre`/`cas_label` to the *dense* half of a fiche chunk's
embedding help fiche recall@4. Both exist because a bare chunk is frequently unretrievable on its
own text (a mid-fiche chunk under a `<Chapitre>` titled *"Résiliation"* names neither the contract
nor the line it governs); both are explicitly **not naive enrichment** — the *sparse* half must
stay on raw text, because the article leg is sparse-leaning and a shared breadcrumb across every
article in one section would tax the leg with generic tokens carrying no signal (the "dilution"
argument).

## Decision

**The enrichment transform never touches storage.** `rag.ingest.enrichment.enrich_article_dense_text`/
`enrich_fiche_dense_text` are pure functions from `(row/meta, chunk)` to a string; nothing in
`rag.ingest.payload` changes, so `text` stays the raw chunk exactly as SPEC §7 already requires.

**The dense/sparse split is enforced at the embed-call boundary, not inside the embedder.**
`rag.ingest.upsert.upsert_articles`/`upsert_fiches` gained one optional keyword, `dense_text`.
`None` (every default arm) keeps the one-forward-pass-per-batch shape SPEC §5 chose BGE-M3 for.
Given a function, `embed` is called **twice** — once on the enriched texts (dense half kept,
sparse half discarded) and once on the raw texts (sparse half kept, dense half discarded) — rather
than adding a second `EmbedFn` shape. This keeps `EmbedFn` (`Sequence[str] -> list[Embedding]`)
the single interface every other caller (retrieval's query-time embedding included) already
depends on; the cost of two passes over one is accepted only for the two non-default challenger
arms this ticket builds, never for the default ingest path.

**Each challenger is a real collection behind the naming convention SPEC §6.4 already
established**, built by `rag.ingest.ab_arms` from the exact same chunk population as the
incumbent — same chunker, same UUIDv5 point ids, no re-chunking, so the golden set's gold labels
stay valid unmodified:

- `articles__m3__c512__breadcrumb-dense__v1`
- `fiches__m3__c512__header-dense__v1`

**Both A/Bs are measured through the same fixed pipeline, rung 2's hybrid fusion**
(`rag.retrieval.pipeline.retrieve_rung2` — both hybrid legs, no expansion, no rerank, no quota),
not rung 1. Rung 1 is dense-only; since the sparse vector is identical between incumbent and
challenger by construction (always raw text), a dense-only pipeline would say nothing about
whether the split survives fusion under the article leg's real, sparse-leaning weights — which is
exactly the question §12.8's "dilution" argument raises. `rag.eval.run_experiment.run_retrieval_ladder`
gained a `pipeline_arm` keyword (default `None`, falling back to `arm`, so every existing rung 1-5
call is unaffected) so a run's header `arm` can record **which collection served the query**
(`"incumbent"`/`"challenger"`) independently of which `RETRIEVAL_ARMS` entry actually ran.

**The stable alias is flipped for the duration of one run and always restored in a `finally`**
(`scripts/run_ab_pilot.py`) — adopting a challenger is a separate, deliberate step taken after
reading a verdict, never a side effect of having measured it.

## Measured result

Both A/Bs ran on 2026-09-16 against the full committed corpus and the then-current 60-item golden
set's retrieval working set, judged by `rag.eval.compare.compare_registered_runs` against
`rag.eval.ladder_registry`'s pre-registered primaries:

| A/B | primary | net discordant | threshold | verdict |
|---|---|---|---|---|
| `ab_article_breadcrumb` | article_recall_at_4 (n=33 paired) | **+0** | ≥ 4 | **keep incumbent** |
| `ab_fiche_header` | fiche_recall_at_4 (n=40 paired) | **−1** | ≥ 4 | **keep incumbent** |

Both fall well short of the ≥4-net-item adoption bar, and neither trips a guard on the other
decision metrics. Per SPEC §12.7 rule 3, inconclusive resolves to *no change*: the incumbent
(raw-text dense and sparse, no enrichment) stays the config the ladder starts from — no default
in `rag.ingest.pipeline`/`rag.ingest.upsert` changes as a result of this ticket. Full per-item
scores and the verdict computation are committed at `eval/runs/ab_article_breadcrumb-*` and
`eval/runs/ab_fiche_header-*`.

The near-zero movement on `article_recall_at_4`/`article_recall_at_10` (0 discordant pairs on
either) while `article_recall_at_candidate` shows movement (3 improved / 3 regressed, net 0) is
consistent with the article leg's sparse-leaning fusion weights: the challenger's dense-only
perturbation reorders some borderline top-20 candidates but rarely survives into the harder top-4
cut once fused against an unchanged, dominant sparse score.

## Consequences

- Both challenger collections (`articles__m3__c512__breadcrumb-dense__v1`,
  `fiches__m3__c512__header-dense__v1`) are left built in the dev Qdrant instance for
  reproducibility/inspection; neither alias points at them, and no code names them outside
  `rag.ingest.ab_arms`/`scripts/run_ab_pilot.py`.
- This declines *this specific* challenger — a dense/sparse split of the named fields, measured
  through rung 2's fixed fusion weights. It does not close the door on a differently-shaped
  enrichment (different fields, different fusion weights, or measured after rung 4/5's rerank and
  quota are in place) — that would be a new, separately pre-registered A/B, not a re-run of this
  one.
- `rag.eval.run_experiment.run_retrieval_ladder`'s new `pipeline_arm` keyword is now the seam any
  future non-rung experiment (a fixed pipeline measuring something collection-shaped, not
  pipeline-shaped) reuses instead of inventing a new harness function.
