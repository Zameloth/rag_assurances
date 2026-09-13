# ADR-0020 — The annotation helper proposes gold-set candidates via a lexical shortlist, never the production retrieval pipeline

- **Status**: Accepted
- **Ticket**: #34 (annotation), new tooling ticket (helper changes)
- **Spec**: SPEC.md §12.1-§12.4, ADR-0010

## Context

Hand-annotating the 38 retrieval-bearing golden items (#34) was the bottleneck ADR-0010 accepted
as the price of a falsifiable rung 3: the annotator must be free to pick a `gold_article` the
fiche never cited, so `<dc:source>` stays a reading aid, never a label source. In practice, that
freedom means searching the full 2,377-article corpus by hand whenever the right article sits
outside the `<dc:source>` reading list — a search that requires legal-domain vocabulary the
annotator doesn't have, for every one of `gold_articles`, `gold_spans` and `expected_points`.

## Decision

Extend the annotation helper (`annotate_app.py`) with three independent, on-demand,
individually-rerollable "propose" actions — `gold_articles`, `gold_spans`, `expected_points` —
each an LLM call the annotator verifies, edits or rejects, never auto-accepted.

`gold_articles` candidates come from a **shortlist**: `<dc:source>` section articles ∪ the top-N
matches of a hand-rolled lexical search (no new dependency, no embedding model) scored against
the question and fiche body combined. The LLM only ever picks from real corpus text inside that
shortlist and justifies each pick — it never names an article from its own knowledge, which would
risk a hallucinated `citation_id`/`cid`. Full-corpus manual search remains available, preserving
the "free to pick outside the suggestion" property ADR-0010 requires.

`gold_spans` candidates are checked server-side as exact substrings of the real chunk text before
being offered — the verbatim invariant (SPEC §12.2) is enforced structurally, not by trusting the
model's quoting.

`expected_state` gets no LLM call: it defaults deterministically from whether `gold_articles`
ended up empty, and stays a normal editable field.

Items where any AI-proposed field was accepted are tagged `ai_assisted`.

**The lexical shortlist deliberately never touches the production embedder, reranker, or any arm
the ladder evaluates.** Deriving gold labels from the pipeline under test would make rung 3
unfalsifiable in aggregate — the same failure ADR-0010 already rejected once for `<dc:source>`,
one level removed.

## Consequences

- The disagreement-detector pass described in ADR-0010 and #34's acceptance criteria (a model
  independently picking articles, disagreements re-reviewed, no authority to change a label) is
  **not satisfied by this change** and stays a separate, later pass over the finished golden set.
  Proposal-first assistance introduces anchoring risk that an independent post-hoc pass doesn't
  have — the two mechanisms solve different problems and don't substitute for each other.
- Behavioural/refusal items (22 of the eventual ~60) are explicitly out of scope for all of the
  above and keep their existing fully hand-written path (ADR-0010's rationale for that carve-out
  is unchanged).
- The lexical shortlist's quality is a new maintenance surface with no test coverage precedent in
  this repo — it's the one genuinely new piece of retrieval-shaped code the eval tooling now
  contains, deliberately kept dumb (term matching, not embeddings) to keep that surface small.
