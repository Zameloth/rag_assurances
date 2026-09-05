# ADR-0019 — Rung 5's provenance preference is a margin-bounded tiebreak, its quota rejects rather than pads, the floor is the reranker's own sigmoid midpoint, and the no-article marker is a synthetic `Candidate`

- **Status**: Accepted — 2026-09-05
- **Ticket**: [#32](https://github.com/Zameloth/rag_assurances/issues/32)
- **Spec**: [`SPEC.md` §9.5](../../SPEC.md#95-context-assembly--register-quota-with-a-floor), [`SPEC.md` §10.7](../../SPEC.md#107-return-shape), [ADR-0018](0018-rung-4-rerank-is-a-scoring-replacement-with-two-config-only-ram-levers.md)

## Context

SPEC §9.5 already settles the shape — 4 fiche + 4 article slots, filled separately after reranking,
article slots preferring expansion-sourced candidates, a relevance floor that rejects rather than
pads, and an explicit no-article marker when the floor is not met. Four things it doesn't pin down:

1. **What "prefer expansion-sourced" means mechanically.** A hard ordering where any
   expansion-sourced candidate outranks any search-sourced one regardless of score, or a bounded
   tiebreak that only decides among comparably-scored candidates? A first pass implemented the hard
   ordering and a code review (`/code-review`'s Spec axis) flagged its consequence concretely: a
   0.51-scoring expansion-sourced article — barely clearing the floor — would bump three
   0.95/0.9/0.85-scoring search-sourced articles out of the 4 slots entirely, which "prefer" alone
   doesn't obviously authorize. Put to the ticket owner directly, the answer was the bounded
   tiebreak.
2. **What counts as "the floor not met."** Zero articles clearing the floor, or fewer than the full
   quota — the two read differently against the acceptance criterion "fewer than 4 articles is a
   valid outcome."
3. **What numeric value the floor takes.** SPEC names the rule, never a number.
4. **What shape the no-article marker takes**, concretely enough that "the fat return object ...
   whether the floor was met" (this ticket's own acceptance criterion) and "an explicit no-article
   marker enters the assembled context" are both satisfiable now, in `RetrievalResult`'s existing
   `contexts: list[Candidate]` field, without waiting on the prompt-assembly ticket (§10.6) that
   actually renders it.

## Decision

**`assemble_quota()` (`rag.retrieval.quota`) orders article slots by score, with an
`EXPANSION_PREFERENCE_MARGIN` (default `0.1`, same `[0, 1]` sigmoid scale as the floor) added to
expansion-sourced candidates before that ordering** — a search-sourced candidate only loses to an
expansion-sourced one scoring within the margin below it; a lead bigger than the margin still wins
outright. Mechanically: sort by `score + (margin if expansion-sourced else 0)`, descending. SPEC
§9.5's own rationale — "expansion-sourced articles carry DILA's editorial provenance; search-sourced
ones carry a cosine score. Not equivalent." — motivates *some* preference, but doesn't extend to
letting a barely-floor-clearing expansion hit erase a clearly-relevant search hit from the context
entirely; the margin is what keeps the preference a tiebreak among comparably-relevant candidates
rather than a category override.

**"The floor not met" means zero articles clear it, not "fewer than 4."** `floor_met` is `True` for
a 1-, 2-, or 3-article partial fill; it is `False` only when the article side is empty after the
floor rejects everything. This is the literal reading of this ticket's acceptance criterion ("When
no article clears the floor...") and of SPEC §9.5's rule against padding: a floor that fires on any
shortfall would have nothing left to distinguish "the pipeline honestly found 2 good articles" from
"the pipeline found none," which is exactly the intermittent zero-article failure this rung exists
to make legible (SPEC §12.7's "zero-article rate," rung 5's primary metric).

**`ARTICLE_RELEVANCE_FLOOR = 0.5`.** `rerank()`'s `compute_score(..., normalize=True)` is a sigmoid
(`rag.retrieval.rerank`), so every candidate arrives as P(relevant) in `[0, 1]` — 0.5 is that
classifier's own decision boundary, the same kind of "starting point, not a measured-for-this-corpus
value" ADR-0016 already used to justify the 2:1 fusion weights. SPEC §12.7 pre-registers "floor
correctness" against the 8 golden `reponse_sans_article` items specifically because this number is
meant to be calibrated by the ladder, not guessed correctly on the first try — landing an
uncalibrated-but-documented default is this ticket's job, not landing the final value.

**Fiche slots take neither the preference nor the floor rule** — a plain top-`fiche_quota` cap.
SPEC §9.5 states both rules for article slots only, and CONTEXT.md's "the relevance floor" entry
already reads "the threshold on article slots," singular.

**The no-article marker is a synthetic `Candidate`** (`register=Register.ARTICLE`, a reserved
sentinel id `no-article-marker`, the French sentence from SPEC §9.5/§10.2 as `payload["text"]`, and
`provenance=frozenset()`) rather than a new field on `RetrievalResult` or `QuotaResult`. It flows
through `RetrievalResult.contexts`'s existing `list[Candidate]` shape unchanged, so anything that
already knows how to render a `Candidate` (`langchain_retriever.py`'s `_to_document`, in particular)
needs no special case to carry it forward; only a downstream reader that cares (the prompt-assembly
step, §10.6, a later ticket) needs to recognise the sentinel id. Empty provenance is deliberate: the
marker was reached by no path, which is a different fact from `Provenance.LOOKUP`'s "found, just not
by search."

**`RetrievalResult.floor_met: bool | None = None`.** `None` on every rung that never runs quota
assembly (rungs 1-4) and on rung 5's own short-circuit path (a metadata lookup has no article slots
to fill — the same reason `candidate_pools` is `{}` there); `True`/`False` only where
`assemble_quota` actually ran. A plain `bool` defaulting to `True` was rejected — it would claim the
floor was checked and passed on rungs where the concept doesn't apply at all, silently answering a
question that was never asked.

**`retrieve_rung5` is `retrieve_rung4` plus one step**, mirroring ADR-0018's own "rung 4 is rung 3
plus one step": short-circuit, both hybrid legs, expansion, merge and rerank are byte-for-byte the
same calls; `assemble_quota` replaces `rank_candidates`' shared top-8 cap as the final step.
`candidate_pools` keeps rung 3's exact shape for the same reason ADR-0018 already gives — quota
assembly, like reranking, is an ordering/selection step over the merged pool, not a fifth pool.
`fiche_quota`/`article_quota`/`relevance_floor`/`expansion_preference_margin` are keyword parameters
with named-constant defaults, the same shape `fiche_weights`/`expansion_depth`/`reranker_model`
already take (#29, #30, #31) — this
ticket's own acceptance criterion ("quota depth, the floor value and quota-vs-free-for-all are all
config") is satisfied the way this codebase already defines that phrase, and "quota vs free-for-all"
specifically is just which arm runs (`RETRIEVAL_ARMS["rung1"]` vs `["rung5"]`), already true before
this ticket touched anything.

**The `PipelineRetriever` (LangChain boundary) is not updated in this ticket.** The same gap
ADR-0018 named for rung 4's traced `rerank_fn` — `_get_relevant_documents` special-cases
`arm == "rung4"` so a Langfuse-wrapped `rerank_fn` reaches `retrieve_rung4` instead of going through
the untraced generic `retrieve()` dispatcher — applies identically to rung 5, since `retrieve_rung5`
calls the same reranker internally. Wiring `arm == "rung5"` through that same special case is
LangChain/Langfuse boundary work, paired with @Zameloth per the #31 precedent this ticket inherits.

## Rationale

- **A margin-bounded tiebreak over a hard ordering** because "prefer" doesn't authorize letting the
  preference outrank the floor itself in importance — the floor exists so that a bad article never
  fills a slot at all, and a hard ordering would let a barely-passing expansion-sourced article
  displace search-sourced articles the floor and the cross-encoder both rate as clearly better. The
  margin keeps the preference doing exactly the job SPEC's "not equivalent" rationale describes:
  deciding between candidates whose relevance is genuinely close, not overriding a clear score gap.
- **Zero, not "any shortfall," as the floor-not-met condition** because the acceptance criteria
  explicitly separate "fewer than 4 is valid" from "no article clears the floor enters the marker" —
  collapsing them would make partial fill indistinguishable from total failure, exactly the
  distinction "zero-article rate" (SPEC §12.7) exists to measure.
- **0.5 over an arbitrary tuned-looking number** because nothing in SPEC or the corpus justifies a
  more specific value yet, and a fake-precise default (e.g. `0.62`) would misrepresent a config
  knob the ladder hasn't calibrated as a measured one.
- **A synthetic `Candidate` over a new return field** because `RetrievalResult.contexts` is already
  the field every downstream consumer reads for "what does generation see," and CLAUDE.md's
  no-speculative-abstraction rule argues against growing the return shape for a concept the existing
  shape already accommodates.
- **`bool | None` over a defaulted `bool`** because a rung where quota assembly never ran has no
  floor to have met or missed — `True` there would be a fabricated answer to an inapplicable
  question, the same category of bug SPEC's own "absent means empty, the consumer that needs it
  fails on it" posture (`rag.config`) already rules out elsewhere.

## Consequences

- `rag.retrieval.quota` imports nothing from `rag.retrieval.pipeline`, `rerank`, or Qdrant —
  `assemble_quota` is exercised entirely against synthetic `Candidate`s, the same posture
  `rank_candidates` already has.
- A reader diffing rung 4 vs rung 5's fat objects sees identical `candidate_pools` and
  `short_circuit_path`, and only `contexts`/`floor_met` differ — the intended signal, since rung 5 is
  a claim about final slot-filling (SPEC §12.7's zero-article-rate question), not about candidate
  depth or which leg found what.
- The no-article marker's sentinel id (`no-article-marker`) is now a reserved value no real point id
  may ever collide with; nothing enforces that at the Qdrant layer, the same trust boundary
  `NO_ARTICLE_MARKER_ID` shares with every other named constant in this codebase that isn't
  independently re-derived from the store.
- `langchain_retriever.py` still special-cases only `arm == "rung4"`; calling `retrieve(..., arm="rung5")`
  through the generic LangChain retriever today reaches `retrieve_rung5`'s own default (untraced)
  reranker resolution, the same known gap rung 4 already had before ADR-0018's paired follow-up.
