# ADR-0015 — Rung 1 is two per-register legs merged dense-only to top-8; the arm is selected by a code constant, not an env var

- **Status**: Accepted — 2026-09-05
- **Ticket**: [#28](https://github.com/Zameloth/rag_assurances/issues/28)
- **Spec**: [`SPEC.md` §6.1](../../SPEC.md#61-the-store-is-a-query-engine-not-a-vectorstore), [`SPEC.md` §9.2](../../SPEC.md#92-the-three-retrieval-paths), [`SPEC.md` §12.7](../../SPEC.md#127-the-pre-registered-rule), [`SPEC.md` §16.3](../../SPEC.md#163-configuration)

## Context

SPEC §12.7's ladder table names rung 1 in one phrase: *"naive baseline — single index, dense-only,
no expansion, no rerank, top-8."* That phrase predates ADR-0005, which already commits ingest to
**two** Qdrant collections behind two aliases (`fiches`, `articles`) — there is no literal single
collection for rung 1 to query. §12.7 is explicitly pre-registered *"before any rung runs"* to stop
a metric being picked after seeing numbers, so reconciling this wording gap belongs in a decision
record, not only in `rag.retrieval.pipeline`'s module docstring.

A second, smaller question rides along: issue #28's acceptance criteria ask that "the arm is
selectable by config, so rung 1 stays runnable after later rungs land." SPEC §16.3 fixes the
`.env` variable table exhaustively; adding an entry there for something that is an ablation choice,
not a secret or a deployment fact, would be the kind of scope SPEC §16.3 doesn't claim.

A third question, pair-programmed rather than agent-authored (per this project's LangChain/Langfuse
split — see the issue's collaboration-split comment): SPEC §6.1 mandates the `BaseRetriever`
subclass itself, "so Langfuse's auto-tracing still sees a retriever span," but doesn't spell out
what `Document.metadata` should carry beyond "register and provenance," nor which of Langfuse's two
integration paths (LangChain's `CallbackHandler` vs. a hand-wrapped span) satisfies "a first-class
`retriever` observation... for free."

## Decision

**"Single index" reads as "one vector kind" (dense-only), not "one collection."** Rung 1 queries
both the `fiches` and `articles` aliases independently — SPEC §9.2's fiche leg / article leg
split, at their already-fixed depth of 20 each (`LEG_CANDIDATE_LIMIT`) — takes only the dense half
of each candidate's BGE-M3 embedding, merges the two pools by raw cosine score with no per-leg
weighting (that lands with #29), and slices to the top 8. No `<dc:source>` expansion, no rerank,
no register quota — those are later ladder rows.

**The retrieval arm is selected by a named code constant plus an injectable function parameter**
(`RETRIEVAL_ARMS: dict[str, RetrieveFn]`, `DEFAULT_RETRIEVAL_ARM`), mirroring `rag.ingest.pipeline`'s
existing `ARTICLES_ARM`/`FICHES_ARM` pattern, rather than a `RETRIEVAL_ARM` environment variable.

**`PipelineRetriever.Document.metadata` carries exactly `register` and `provenance`, both as
plain strings** (`Register.value`, and `provenance` as a sorted list of `Provenance.value`, not
the `frozenset` of enums `Candidate` itself carries) — nothing else from `Candidate.payload` is
copied across, and `Candidate.id` becomes `Document.id` (a first-class field on LangChain's
`Document`), not a third `metadata` key.

**Langfuse observability uses LangChain's own `CallbackHandler`
(`langfuse.langchain.CallbackHandler`), attached via `config={"callbacks": [...]}` at invoke
time — not a hand-wrapped span inside `_get_relevant_documents`.** `PipelineRetriever` itself
carries zero Langfuse-specific code as a result.

**#28 proves this wiring works — it does not turn it on.** No call site (`rag.query` included)
attaches a `CallbackHandler` today; that's real production wiring for a later ticket, once there
is a full retriever-plus-generation chain to attach it to.

## Rationale

- **Two legs, not one merged collection**, because ADR-0005 already settled the storage question
  and nothing in #28 reopens it. Interpreting "single index" as "single collection" would demand
  re-merging `fiches` and `articles` — a storage-architecture change with no ticket behind it —
  just to satisfy one adjective in a table cell written before that architecture existed.
- **Per-leg depth stays at SPEC §9.2's `top-20`** rather than shrinking to something rung-1-sized,
  so the per-leg candidate pools in the fat object (`RetrievalResult.candidate_pools`) are
  comparable across every rung from the start — the eval harness (not yet built) reads candidate-
  depth recall the same way regardless of which rung produced the pool.
- **A code constant, not an env var, selects the arm** because SPEC §16.3's table is described as
  the configuration surface end to end ("every variable the pipeline reads appears here"); which
  retrieval arm runs is a code-level ablation choice exactly like `ARTICLES_ARM`/`FICHES_ARM`
  already are, not an environment-specific fact like `QDRANT_URL`.
- **`Document.metadata` stays minimal** because `RetrievalResult.contexts`/`candidate_pools`
  (the fat object) already carry the full `Candidate` — including `score` and the raw `payload`
  — for anything eval-side. `Document` is the generation-facing contract, not a second copy of
  the eval one; widening it is a one-line change for whichever later ticket has a real consumer
  (e.g. citation building) rather than a speculative default now.
- **`CallbackHandler` over a hand-wrapped span** because it's the literal mechanism behind SPEC
  §6.1's claim that a real `BaseRetriever` gets Langfuse tracing "for free" — hand-wrapping a span
  would duplicate what LangChain's own callback propagation already does for a bona fide
  `Runnable`, and would need updating every time this retriever's shape changes.
- **No production call site wired yet** because #28 has no generation chain for a `CallbackHandler`
  to usefully sit on top of — `rag.query` calling `retrieve()` directly (not through
  `PipelineRetriever`) is unaffected either way. Wiring it in now would be plumbing with no
  consumer, the same reasoning `rag.query`'s own module docstring gives for staying on the plain
  seam rather than the LangChain-wrapped one.

## Consequences

- `rag.retrieval.pipeline.retrieve_rung1`'s per-leg limit (20) and final cap (8) are two different
  numbers on purpose; a future reader should not "fix" the per-leg limit down to 8 expecting it to
  match the ladder's rung-1 headline figure.
- Rung 2 (#29) reuses `search_leg` and `merge_candidates` unchanged and only changes how the merged
  pool is scored (per-leg weighted fusion instead of raw dense score) — this ADR's shape is what
  makes that a small diff rather than a rewrite.
- Adding a second retrieval arm to `RETRIEVAL_ARMS` is the whole integration surface for a future
  rung; no `Settings`/`.env.example` change is implied by that addition.
- `langchain` (the metapackage, not just `langchain-core`) is a direct dependency purely because
  `langfuse.langchain.CallbackHandler` hard-imports it for a `__version__` check it never uses
  beyond that branch — nothing in this codebase imports `langchain` or the `langgraph` stack it
  pulls in transitively. A future reader should not read its presence in `pyproject.toml` as this
  project using LangChain agents/graphs.
- `src/rag/retrieval/langchain_retriever.py` having no tracing code is intentional, not a gap —
  don't add a Langfuse import there to "make sure" observability is wired; that's exactly the
  duplication the `CallbackHandler` decision above rules out.
- Whichever ticket wires `CallbackHandler` into a real call site still owns choosing what triggers
  attaching it (every `rag.query` invocation? only the served app? sampled?) — this ADR settles
  the *mechanism*, not that policy.
