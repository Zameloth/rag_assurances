# ADR-0001 — A consumer assistant over the public Tier-1 corpus, answering in two parts

- **Status**: Accepted — 2026-08-03
- **Ticket**: [#2](https://github.com/Zameloth/rag_assurances/issues/2)
- **Spec**: [`SPEC.md` §1](../../SPEC.md#1-purpose-and-scope)

## Context

Three candidate use cases were available: a generic explainer, a regulatory lookup tool, and
contract-grounded Q&A over a user's own policy. The corpus survey ([#3](https://github.com/Zameloth/rag_assurances/issues/3))
made them unequal: the first two rest on service-public.fr fiches and the Code des assurances,
both Licence Ouverte 2.0 and **already cross-linked via `<dc:source>`**, while insurer
*conditions générales* parse cleanly but are **not redistributable** (MAIF forbids it in
writing; AXA is silent, which is not permission).

Insurance advice is also a regulated act in France, so the boundary between explaining and
advising had to be drawn before any prompt was written.

## Decision

**A single-purpose assistant for a curious consumer, over public French insurance law only.**

- **Scope is a rule, not a document list**: any fiche whose `<dc:source>` resolves into the
  Code des assurances, plus the in-force Code. Every consumer document therefore has a legal
  counterpart in-corpus.
- **Answers are two-part**: a consumer-French explanation, plus the *fondement juridique*
  citing article(s) where one exists.
- **The refusal line is recommendation, not personalisation.** Mapping a described situation
  onto the applicable rule is *in* scope; recommending a product or a course of action is not.
- **Multi-turn**, with history.

## Consequences

- The scope rule **drops pure *assurance maladie*** — those fiches rest on the Code de la
  sécurité sociale and would be answerable but never citable, reading as retrieval failures
  when they are really scope failures. That trap is why out-of-corpus is later a *model
  judgement*, never inferred from retrieval failure ([ADR-0009](0009-typed-answer-envelope-and-citation-containment.md)).
- **Situational questions must be answered, not refused.** They exploit the `<Cas>` branches
  that a naive chunker destroys — which is why `cas_label` exists
  ([ADR-0007](0007-flat-payload-two-indexes-cid-identity.md)) and why merges never cross a
  `<Cas>` boundary ([ADR-0003](0003-structural-chunking-under-a-512-token-band.md)).
- **Tier 3 drops out entirely**, taking the unresolved TDM-exception question with it. **Tier 2
  (ACPR) leaves the corpus** but survives as a *system-prompt input* — 2024-R-03 is what fixes
  where the refusal line falls — so its missing reuse licence never has to be resolved.
- The two-part contract gives evaluation **two independently measurable things** rather than
  one fuzzy blob.
- Multi-turn requires a condensation stage ([ADR-0008](0008-condensation-fires-on-history-only-and-cannot-add-references.md)).
- The corpus is **in-force only**; the residual "do citations carry a version date?" resolves
  as a corpus-level snapshot stamp.
