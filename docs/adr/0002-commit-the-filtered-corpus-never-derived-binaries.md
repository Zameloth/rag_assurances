# ADR-0002 — Commit the filtered corpus to git; never commit derived binaries

- **Status**: Accepted — 2026-08-03
- **Ticket**: [#16](https://github.com/Zameloth/rag_assurances/issues/16)
- **Spec**: [`SPEC.md` §3](../../SPEC.md#3-corpus), [§16](../../SPEC.md#16-repository-layout-configuration-and-licensing)

## Context

The repo is public and the corpus is Licence Ouverte 2.0, so committing it is *permitted*. The
question was whether it should be committed or fetched reproducibly at build time.

The obvious framing — "it's 5 MB and the licence allows it" — reaches the right answer for the
wrong reason and would not survive a size change.

## Decision

**Commit the *filtered* Tier-1 document set. The repo commits sources and decisions, never
derived binaries.**

| Path | Form |
|---|---|
| `data/corpus/fiches/F*.xml` | verbatim DILA XML, original filenames |
| `data/corpus/articles.jsonl` | `cid`-sorted JSONL, **carrying `texteHtml`** |
| `data/corpus/corpus_manifest.json` | script-emitted licence record **and** provenance pin |

Refresh is a **manual, reviewed commit** reporting added / removed /
**changed-text-under-the-same-`cid`**. `LICENSE` (MIT, code only) is split from
`data/corpus/LICENSE.md` (LO 2.0).

## Rationale

- **`vosdroits-latest.zip` has no version to pin.** Fetch-at-build is not merely inconvenient
  on the fiche side, it is structurally impossible. Git is the only available pin.
- **The eval design decides it, not the licence.** Gold labels are hand-annotated and the
  adoption rule is a paired per-item comparison across six rungs. A rung-4-vs-rung-3 comparison
  run a week apart would be a **cross-corpus comparison wearing the costume of an ablation**,
  invisible in the numbers.
- **Form is asymmetric on purpose.** Fiches stay XML because `<Chapitre>`/`<Cas>` are the
  chunking option space and `cas_label`'s only source — parsing here would decide chunking by
  accident. Articles are JSONL not parquet because **binary does not diff**, and the refresh
  review is the whole point.
- **`texteHtml` is mandatory**: `texte` has zero newlines corpus-wide, so a JSONL carrying only
  `texte` reproduces the corpus's *text* but not its *points*, which is the level the ladder
  compares at.

## Consequences

- Chunking is unblocked and its output is **not repo content**, so re-chunking is a config flip
  with no data commit.
- **52% of articles have been amended**, so refreshes are *expected* to invalidate some labels.
  The changed-text count is the alarm; a filename-level diff would hide it.
- Scheduled refresh is rejected — it would let the corpus move *between* two rungs.
- Two derived things ship anyway, named so the principle is not later cited against them: the
  per-item eval scores, and the hand-annotated golden set.
- The Qdrant snapshot stays out of git — which later forces the delivery decision in
  [ADR-0014](0014-parquet-points-dump-on-a-release-with-alias-flip.md).
