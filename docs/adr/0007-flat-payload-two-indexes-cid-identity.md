# ADR-0007 — Flat payload, exactly two filterable fields, article identity on `cid`

- **Status**: Accepted — 2026-08-03
- **Ticket**: [#15](https://github.com/Zameloth/rag_assurances/issues/15)
- **Spec**: [`SPEC.md` §7](../../SPEC.md#7-payload-manifest)

## Context

Three fields were required by retrieval (section ids for expansion, an exact-match article
number for the short-circuit, a register marker for quota-filling), and the corpus carries
topical breadcrumbs that look filterable.

Measuring the article `num` field first turned out to matter more than any of that.

## Decision

**A flat per-chunk payload with exactly two payload indexes, the article number split into two
fields, and article identity anchored on `cid`.**

| field | index | note |
|---|---|---|
| `citation_id` | — | DILA's raw `num`, **verbatim, never null** |
| `lookup_key` | **keyword** | strict full-match normalization, **null for the 21 annexes** |
| `section_id` | **keyword** | expansion (`MatchAny`) |

`register` and `provenance` are **retrieval annotations, never stored**. `provenance` is a
**set**, unioned across legs. **No topical filtering, ever.**

## Rationale

- **The `num` format space is wider than assumed**: 430 articles (18%) have three or more
  segments, 69 carry an asterisk (`R*113-4`), 21 are prose labels. A single field cannot serve
  both a lookup key (normalize hard) and a citation id (normalize nothing) — and normalizing
  `Annexe à l'article A121-1` to `A121-1` would **collide with the real `A121-1`**.
- **`lookup_key` is computed only on a full match**, which makes the key set **provably
  collision-free**. A **membership check** before short-circuiting converts any future regex gap
  into normal search rather than a confident wrong answer.
- **`cid`, not the version id.** 1,230 of 2,377 articles (52%) diverge. A version anchor
  silently invalidates hand-annotated gold labels on any refresh, and orphans points on
  re-ingest.
- **Flat over a document registry** on ablation-arm consistency: a registry sits *outside* the
  alias mechanism, so rungs would share one file with nothing detecting staleness after a
  re-chunk. Duplication costs ~150 KB, and `set_payload` rewrites payload without touching
  vectors.
- **`provenance` must be a set** — an article can arrive by both legs in one query, and a scalar
  would make the winner depend on merge order. `register` is not stored because a stored copy can
  only ever be *wrong*, disagreeing with the collection it lives in with nothing detecting it.
- **No topical filtering**: gates fail closed and Livre Ier is cross-line; it needs a classifier
  we don't have; **the article side has no line taxonomy at all**; 87 fiches is not a set worth
  narrowing; and no rung measures it.

## Consequences

- **Two corrections to closed decisions, both silent-wrong-answer bugs** — the short-circuit
  regex, and `cid` vs the version id (which also lands on the golden set).
- **`<dc:source>` expansion verified working natively**: `sectionParentId` resolves a real fiche
  value to 22 articles; across all 556 sections the distribution is median 3, p90 10, max 43.
- **The cap of 40 becomes a filtered vector search.** This does not reinstate the weak hop — ~25
  nDCG@10 is a needle in 2,377 articles, whereas this only *sorts within* a set DILA has already
  certified. Similarity is demoted from **gate** to **sort order**.
- **`cas_label` closes a gap that would have struck the one capability the use case calls
  distinctive** — a chunk cut from inside a `<Cas>` does not repeat "si vous êtes locataire".
- Metadata-as-embedding-input becomes a **pre-ladder A/B**, not a silent default and not a
  seventh rung; the challenger is dense-enriched / sparse-raw, because a naive challenger would
  confound two effects.
- **Assertion 5** — every fiche `section_ids` entry resolves to ≥ 1 article — is a build-time
  health check on rung 3, and gates the refresh as well as the ingest.
- **Recorded limitation**: a user meaning the Code de la consommation's `L121-1` is served the
  assurances article silently. No field fixes this; it is the model's `hors_corpus` judgement.
