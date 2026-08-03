# ADR-0003 — Structural descent under a shared 512-token band, no overlap

- **Status**: Accepted — 2026-08-03
- **Ticket**: [#7](https://github.com/Zameloth/rag_assurances/issues/7)
- **Spec**: [`SPEC.md` §4](../../SPEC.md#4-ingestion-and-chunking)

## Context

The two embedder arms differ enormously in window — BGE-M3 takes 8192 tokens,
`multilingual-e5-large-instruct` only 512 — which looked like a hard constraint forcing per-arm
chunking. The corpus is also mixed-regime, and its XML carries native boundaries.

The corpus was measured directly (all 2,377 articles, all 87 fiches, BGE-M3 tokenizer) before
any decision was put.

## Decision

**A 512-token band, one shared chunk population across every rung and both embedder arms,
recursive structural descent, no overlap.**

- **Fiches** — body is four elements of 38 (`Introduction`, `Texte`, `Conclusion`,
  **`ListeSituations`**) → recursive descent → merge under a **100-token floor**, never crossing
  a `Chapitre`/`SousChapitre`/`Cas` boundary.
- **Articles** — one article per point where it fits; else `texteHtml` blocks packed to the
  band; `<table>` content stripped; stub point where prose falls under 32 tokens.

Output: **3,687 points** (2,805 articles + 882 fiches), 15.1 MB of dense vectors.

## Rationale

- **The window constraint is nearly inert**: article median 167 tokens, fiche `<Chapitre>`
  median 169, only 12.2% of articles over 512 and exactly one over 8192. M3's window is never
  used on this corpus.
- **The shared population is forced**, not convenient. Per-arm chunking breaks rung 6's
  one-variable rule and un-shares the point ids gold labels depend on; sizing to M3 instead would
  make e5 **silently truncate** 12% of articles, so rung 6 would measure "embedder + data loss".
- **Dropping navigation halves the fiche** (median 2,199 → 1,203 tokens).
- **`<ListeSituations>` reads as navigation and is not**: all 8 fiches lacking `<Texte>` carry
  one instead, it duplicates nothing, and it holds **26% of all indexed fiche text**. Excluding
  it would have indexed 8 of 87 fiches with **zero content**.
- **Descent over "pick a tag"** for its degenerate cases, not its typical one — both produce
  ~identical output, but descent handles the 18 Chapitre-less fiches and nesting that runs
  **both ways** (57 Chapitres contain a `<Cas>`, 11 `<Cas>` contain a `<Chapitre>`).
- **The floor is 100 because the quota is fixed at 4 fiche slots**, making chunk size the dial on
  how much consumer-French reaches the prompt. It lands the fiche median (137) near the article
  median (187).
- **`texte` has zero newlines**, so article structure exists only in `texteHtml`; extracted text
  is identical (difflib 1.000).
- **`<table>` stripping is structural** because the 21 annexes are two populations: 18,465 tokens
  of standard-form policy wordings against 29,902 tokens of mortality table.
- **No overlap, measured**: 0/882 fiche and 5/2,805 article chunks come from an arbitrary cut.

## Consequences

- `gold_spans` containment is **demoted from tuning signal to regression check** — it has nothing
  to detect here.
- Fiche header enrichment becomes a **second pre-ladder A/B**, judged on fiche recall@4, costing
  no statistical power because the two registers' recalls are already separate numbers.
- The 11.5 MB / 3.6 MB collection split **straddles Qdrant's 10 MB `indexing_threshold`**,
  confirming [ADR-0005](0005-qdrant-two-collections-behind-aliases.md)'s override catches a
  second trap.
- The corpus is **87 fiches, not 79** — resolving the scope rule against `sectionParentId`.
- `<Definition>` is excluded: 150 39-token tooltips would compete for 4 fixed slots.
