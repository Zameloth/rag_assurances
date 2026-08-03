# ADR-0009 — A typed answer envelope, and citation validity as containment in the retrieved context

- **Status**: Accepted — 2026-08-03
- **Tickets**: [#10](https://github.com/Zameloth/rag_assurances/issues/10), corrected by [#12](https://github.com/Zameloth/rag_assurances/issues/12), [#15](https://github.com/Zameloth/rag_assurances/issues/15), [#17](https://github.com/Zameloth/rag_assurances/issues/17)
- **Spec**: [`SPEC.md` §10](../../SPEC.md#10-generation)

## Context

The answer contract is two-part and the refusal boundary is drawn, but nothing yet said what the
model actually returns, how citations are verified, or how a refusal is distinguished from a
retrieval failure.

## Decision

**`mistralai/mistral-large-2512` through OpenRouter, returning a schema-enforced discriminated
union, with citation validity defined as `cited ⊆ retrieved_context`.**

```
type: "reponse" | "refus"
explanation:          str                    # unconstrained prose, both branches
fondement_juridique:  [{article_id, gloss}]  # citation_id, copied verbatim
aucun_fondement:      str | None
motif:                enum | None            # recommandation_produit | conseil_action | hors_corpus
```

Pinned routing is mandatory: `provider.require_parameters: true`, `allow_fallbacks: false`, a
pinned `order`, and `response_format: {type: json_schema, strict: true}`.

## Rationale

- **The schema constrains the envelope, not the writing.** Three things fall out that free
  markdown cannot give: the citation guardrail loops over a **typed field** rather than regexing
  prose (a model-authored citation string is a formatting lottery); explanation quality and legal
  grounding become **literally two fields**; and the no-article marker becomes a field the model
  must fill, not a phrase it can silently drop.
- **Four terminal states stay distinguishable.** No-article is an *answer*, not a refusal.
  Regulated-act refusals carry **no retrieval signal at all** — retrieval succeeds perfectly on
  *"quelle assurance auto choisir ?"* and the system must still decline. **Out-of-corpus must
  never be inferred from retrieval failure**: a scope failure and a retrieval failure have
  completely different fixes.
- **Refusals still answer the informational part.** The regulated act is *recommending*, not
  *explaining*; a bare refusal would decline something service-public.fr does under state mandate,
  using the same text.
- **`⊆ retrieved_context`, not `⊆ corpus`.** An article that exists but was never retrieved is
  still fabrication — produced from parametric memory, not from the pipeline — and a corpus-wide
  lookup waves it through. Real-but-wrong citations are the *worse* failure precisely because they
  survive naive checking. **Never auto-repaired**: a repaired answer hides the failure from the
  eval built to find it.
- **Pinned routing is not a nicety.** OpenRouter's structured-output support is **per-endpoint,
  not per-model**, and some providers treat `strict` as a hint — which would silently degrade the
  guarantee that motivated the schema in the first place, surfacing as sporadic parse errors on
  some runs and not others.
- **No research ticket was spent on the model.** The French-native-≠-legal-capable warning is
  about *embedders*; generation never performs that hop. The real risk is contract-faithfulness,
  which is exactly what the eval measures — so the model is an **ablatable arm**, not an
  assumption.

## Consequences

- **History carries prior prose with prior citation lists stripped.** Otherwise legitimate
  conversational back-reference (*"comme vu, l'article L121-1…"*) is recorded as a hallucination.
  Widening the check to the session union was rejected — it **decays turn by turn**, weakest
  exactly when a conversation is long enough to need it. This rule is now load-bearing for the
  condenser too.
- **The prompt excludes `provenance` and URLs.** The quota already encodes the expansion
  preference structurally; signalling it twice risks under-citing search-sourced articles with no
  visible cause. A model given a URL writes URLs into prose, which an id-based check cannot
  validate.
- **Three of five generation metrics need no judge at all** — state accuracy, citation validity
  and citation correctness are `==` and set containment, precisely because the envelope is typed.
- The chain returns a **fat object**, because evaluators see only the return value.
- **Known risk, recorded**: one call does three jobs — answer, police the refusal line, split
  refusable from answerable. The most iteration-prone component in the system; the typed `motif`
  is what keeps it measurable.
