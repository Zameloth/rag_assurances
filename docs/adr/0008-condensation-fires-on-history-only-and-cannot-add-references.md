# ADR-0008 — Condensation fires only on history, never reaches generation, and cannot add an article reference

- **Status**: Accepted — 2026-08-03
- **Ticket**: [#17](https://github.com/Zameloth/rag_assurances/issues/17)
- **Spec**: [`SPEC.md` §8](../../SPEC.md#8-query-condensation)

## Context

The assistant is multi-turn, so *"et si je suis locataire ?"* is meaningless to a retriever on
its own. Something must turn `(history, follow-up)` into a standalone query.

Neither the retrieval nor the metadata decision said **which text** the article-reference scan
reads — and that omission matters in both directions, because condensation can *destroy* and
*manufacture* references.

## Decision

**A separate pre-retrieval LLM call on `CONDENSER_MODEL`
(`mistralai/mistral-small-3.2-24b-instruct`), firing only when history exists, whose output
never leaves the retriever.**

- **The article-reference scan reads the raw user turn only.** A hit **skips condensation
  entirely**; the condensed query is **never re-scanned**.
- Output is a one-field strict schema `{"requete": str}` plus a deterministic sanitizer
  enforcing, among other things, **`refs(condensed) ⊆ refs(raw_turn)`**. Every trip falls back
  to the raw turn.
- **No "does this need condensing?" gate** — passthrough is handled *in the prompt*.
- History window: **3 exchanges, both roles, trimmed server-side** on turn count *and* raw size.
- `condensed_query` and a code-computed `condense_status` enum join the chain's return value.

## Rationale

- **Manufacture is the dangerous direction.** A condenser can emit `L121-1` from parametric
  memory; short-circuiting on that returns an exactly-retrieved, wholly confident **wrong**
  article — and `cited ⊆ retrieved_context` **passes it**, because it really was retrieved.
  Scanning the raw turn closes the short-circuit half **by construction**; the subset rule closes
  the search-leg half. Nothing legitimate is lost, because history stripping
  ([ADR-0009](0009-typed-answer-envelope-and-citation-containment.md)) leaves no honest source
  for an unasked-for reference.
- **A gate would spend a call to save a call**, and its false negatives are invisible.
- **Typed output, not a bare completion**, because preamble (*« Voici la question
  reformulée : »*) is mild noise on the dense leg but **real lexical weight against the sparse
  leg** that carries article recall.
- **One field, not two**: a `passthrough` flag can disagree with the text; `condensed == raw`
  cannot.
- **The model is a separate config key on variable isolation, not cost** (~$0.04 vs ~$0.26 across
  a campaign). The generation model is ablatable, and the generation dataset holds the 10
  multi-turn items that are *the only items measuring the condenser*. One shared key would swap
  the condenser on exactly those items — two variables, no visible symptom.
- **Local is out**: against a ~4.5 GB budget with no swap, the condenser was the **last
  discretionary resident model**.
- **The condensed query does not reach generation**, so a condenser bug cannot surface as an
  answer-quality failure with a retrieval-stage cause.

## Consequences

- **The ladder's zero condenser cost is guaranteed** by the `history == []` skip, not assumed.
- **The short-circuit's query-side regex must differ from the field validator.** The anchored
  full-match pattern never matches a reference inside a sentence, and simply dropping the anchors
  reintroduces the `L113-15-2` → `L113-15` truncation. **Two patterns, one normalizer.**
- Few-shot examples must **not** be drawn from the golden set, or the measurement is partly one of
  memorisation.
- Both roles are kept in the window because the golden set scripts and freezes the prior assistant
  turn; dropping assistant turns would make those fixtures carry text the system never sees.
- **Recorded unsolved**: partial subject change, and mild over-rewriting. Neither is measurable at
  10 items.
