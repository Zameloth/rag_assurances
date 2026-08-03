# ADR-0010 — A hand-annotated golden set, labelled at document level, with `<dc:source>` demoted to a reading aid

- **Status**: Accepted — 2026-08-03
- **Ticket**: [#11](https://github.com/Zameloth/rag_assurances/issues/11)
- **Spec**: [`SPEC.md` §12.1–12.4](../../SPEC.md#12-evaluation)

## Context

`<dc:source>` was offered as "close to free ground truth" for article recall — a ready-made
mapping from consumer questions to the legal text behind them. Retrieval metrics were also
expected to need **known-relevant chunk ids** per query.

Both turned out to be wrong, in ways that would have been invisible in the numbers.

## Decision

**60 hand-annotated items in one repo-canonical YAML file, labelled at document level, with gold
articles hand-picked at article granularity.**

| field | granularity |
|---|---|
| `gold_fiches` | fiche id |
| `gold_articles` | **`cid`** |
| `gold_spans` | verbatim text |
| `expected_points` | 1–3 assertions, scored as coverage |
| `expected_state` | mirrors the answer envelope field-for-field |

One schema with `history` usually empty; the ladder runs the `history == []` subset. Storage is
**repo-canonical, Langfuse a projection**.

## Rationale

- **`<dc:source>` as ground truth makes rung 3 unfalsifiable.** Expansion would follow exactly
  the link the labels came from and score near-perfect **by construction**, while the direct
  article leg is graded against a target it was never aiming at. "The architecture works" would
  mean only "it did the thing it does". Second, independent defect: `<dc:source>` holds *section*
  ids, and "the right section reached the pool" is a far weaker claim than "the right article
  reached the prompt".
- **The human pick must be free to land on an article the fiche never cited.** That freedom is
  what actually breaks the circle. It converts ground truth into a manual annotation job, which
  is **the price of a headline experiment that can fail**, accepted deliberately.
- **Chunk-id labels would die the moment chunking landed**, or the first time chunk size was
  tuned — precisely the iteration this eval exists to enable. Document-level labels plus verbatim
  spans are invariant to chunking, so chunking landed with **no re-annotation**.
- **Both sides of the single-turn/multi-turn fork are wrong.** Pure single-turn: in production
  retrieval never sees the user's question. Pure multi-turn: a condenser call in front of six
  rungs taxes every run with a component not under test. So the multi-turn subset carries **gold
  contexts, not gold condensed queries** — the condenser is scored by *downstream effect*.
- **Questions are fiche-anchored, LLM-drafted, human-rewritten.** Generating from article text
  would hand the sparse leg a score earned from **lexical leakage**, "proving" rung 2 when what
  was measured was the generator talking to itself. The rule is *the words a consumer knows, never
  the statute's phrasing*.
- **`expected_points`, not reference answers.** 60 paragraphs of authored French is the largest
  time sink available, and reference answers invite scoring similarity to one person's phrasing.
- **Repo-canonical for git, not for durability.** What is actually needed is **diff and blame on
  ground truth**: when rung 4 posts a different number, "did the pipeline change or did the labels
  change?" must be answerable.

## Consequences

- **Labels anchor on `cid`** — 52% of articles are amended, so a version anchor would let any
  refresh silently invalidate paid-for human work.
- **`gold_spans` is a regression check, not a tuning signal** — nothing in this corpus is cut
  arbitrarily, so containment cannot discriminate granularity.
- **Two limits recorded up front**: rungs 5 and 6 will likely land in the noise at ~44 items, and
  comparison must be **paired per-item**, never two independent proportions.
- **Validation is a disagreement detector, not a validator** — a solo annotator has no
  inter-annotator agreement, so the substitute catches attention slipping on item 34 of 38 rather
  than pretending to catch bias.
- **Explanation precision remains unmeasured**, handed forward deliberately.
- The 22 behavioural items are handwritten with no LLM — an LLM asked to generate refusable
  questions produces cartoons of them — and the **situational** items are the canary that the
  guardrail has not been over-tuned into uselessness.
