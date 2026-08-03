# ADR-0012 — Langfuse Cloud on the free tier, with tracing gated by environment

- **Status**: Accepted — 2026-08-03
- **Tickets**: [#5](https://github.com/Zameloth/rag_assurances/issues/5), [#6](https://github.com/Zameloth/rag_assurances/issues/6)
- **Spec**: [`SPEC.md` §11](../../SPEC.md#11-observability), [§16.3](../../SPEC.md#163-configuration)

## Context

Langfuse is a premise of the project, so the question was never *whether* but *what it actually
provides* and *what it constrains*. Its capabilities were read out of the server source rather
than the docs.

## Decision

**Langfuse Cloud, Hobby free tier, EU region (`https://cloud.langfuse.com`), project
`rag-assurances`, server v4, `langfuse>=4.14`.**

Secrets live in a gitignored `.env` at the repo root, loaded via `python-dotenv`, with
`.env.example` committed as the template.

**`LANGFUSE_TRACING` defaults false in dev**, is forced true inside `run_experiment`, and is on
in prod.

## Rationale

- **Self-hosting was rejected**: six containers (web, worker, ClickHouse, Postgres, Redis, MinIO)
  at a recommended 16 GiB, it teaches nothing about RAG eval, and it *removes* a feature — code
  evaluators are disabled without a separately configured dispatcher.
- **The managed retrieval evaluators are not the metrics they are named after.** Context Precision
  is a single 0/1 verdict with **no rank awareness**; Context Recall's prompt **never references
  the ground truth**, making it closer to faithfulness. Both are still v1. Managed **Faithfulness
  v2** does implement decompose/verify/ratio and is usable. So **all retrieval quality is
  hand-rolled**.
- **Tracing is one line of LangChain wiring** and captures retrieval as a first-class `retriever`
  observation, so "retrieval failed vs generation failed" is distinguishable for free. Reranking
  is **not** auto-traced and needs a hand-wrapped span.
- **Free-tier limits, confirmed in the UI rather than estimated**: 50k units/month, **30 days data
  access**, 2 users. At ~12 units per item per run that is ample for experiments — **the real burn
  risk is interactive dev tracing**, which at a hundred queries a day exceeds the entire ladder.
- **`.env` over `pass`/keyring/profile exports**: both add a step to every run and every snippet,
  for a solo project with no shared secrets. Rotation is a key-regenerate plus one file edit.

## Consequences

- **The task must return `{"answer": ..., "contexts": [...]}` and more** — evaluators see only the
  task's return value, never the trace. This is a requirement on the *pipeline's return shape*,
  not an eval detail.
- **30-day retention forces per-item scores into git** ([ADR-0011](0011-pre-registered-metrics-and-adoption-rule.md)),
  not aggregates as first assumed — the adoption rule counts items across runs weeks apart.
- **`max_concurrency` must be set to 5–10**; the SDK default of 50 will rate-limit OpenRouter.
- **Two silent version traps**: the host variable is **`LANGFUSE_BASE_URL`**, not the SDK v3
  `LANGFUSE_HOST` (on EU cloud the wrong name fails *silently*, because the fallback happens to be
  correct); and `DatasetItemClient.run()` was removed in SDK v4, so most older RAG-eval tutorials
  are dead code.
- **Retrieval-only experiments are the cheap lever** — the task is arbitrary Python, so the entire
  ablation ladder runs **without paying for generation**. This is what makes a six-rung ladder
  affordable and shapes the two-dataset split.
- **Every managed judge prompt is English while all content is French.** Recorded as a risk, and
  later discharged **empirically** on the calibration set rather than assumed either way.
- Langfuse connects to OpenRouter as provider *OpenAI* with a custom Base URL; the gateway must do
  tool calling in OpenAI format, since managed judges extract `score` and `reasoning` via a
  function call.
- **Nothing bit during provisioning** — no credit card, no waitlist, no approval delay.
