# ADR-0013 — A FastAPI + HTMX typed-object renderer, sleeping behind the existing Traefik

- **Status**: Accepted — 2026-08-03
- **Ticket**: [#13](https://github.com/Zameloth/rag_assurances/issues/13)
- **Spec**: [`SPEC.md` §13](../../SPEC.md#13-application), [§14](../../SPEC.md#14-deployment)

## Context

Deployment was a premise — the author's own VPS via docker-compose — so only the interface was
genuinely open. It looked like a shallow pick between chat frameworks.

One constraint reordered the whole comparison before it started.

## Decision

**FastAPI + Jinja/HTMX, one `rag-assurances` container beside `qdrant`, both sleeping via Sablier
behind the VPS's existing Traefik v2.11.**

- **No token streaming.** Staged progress via **SSE carrying progress events only**.
- Dual endpoints: `POST /api/ask` returns the envelope as JSON with free OpenAPI docs;
  `POST /ask` returns the HTMX partial.
- **Stateless** — history is client-side and posted back each turn.
- **`/health` means models loaded AND Qdrant reachable AND the alias matches `index_lock.json`.**
- Public, with three guards: bot-UA **403 at Traefik without waking**, `rateLimit`, and a hard
  OpenRouter credit cap.
- Prod tracing **on**; dev tracing off.

## Rationale

- **A strict `json_schema` response streams as unreadable raw JSON**, so token streaming — the
  headline feature every chat framework sells — is worth nothing here. **The app is a typed-object
  renderer, not a chat stream**: four states are four partials. Dropping the strict schema for the
  demo path was rejected outright — it would make the demoed system a *different* system from the
  measured one.
- Streamlit and Gradio lose on **portfolio saturation** (the bar is *doesn't look like everyone
  else's project*); Gradio lost twice over, since HF Spaces integration was already deleted by the
  VPS premise. Chainlit wins the pure chat-framework comparison and loses to the fact that this
  isn't a chat product.
- **The dual endpoint makes the typed contract inspectable by curl**, for a project whose thesis
  *is* a typed contract.
- **The licence attribution needs a durable surface.** LO 2.0's three-part credit is a *condition*,
  not decoration, and message-bubble frameworks make a persistent footer awkward.
- **Sleep is about the weights, not the web server** (~5 GB vs ~150 MB). In-process unloading was
  **reversed**: it is *more* application code, and `gc.collect()` returning that memory to the OS
  is not guaranteed, while a stopped container returns it with certainty. Container sleep was also
  wrongly flagged as out-of-scope infra and as forcing an api/web split — Traefik already exists
  and *is* the always-on component.
- **Sleep makes exposure a spend question**: bots would otherwise wake ~5 GB per crawl.
- **Stateless is the natural partner to a container that stops whenever it likes.** Server-side
  sessions would silently eat a conversation that idled past the timeout.

## Consequences

- **The pipeline stays an importable library** — `compare.py` and the app share one code path.
  Non-negotiable: otherwise the evaluated and served systems drift.
- **RAM is the binding constraint, and it is concurrency not capacity.** ~6.1 GB free with the
  other demos asleep, ~5.3 GB with one awake, **no swap**; realistic peak with both models is
  ~5.2–5.5 GB once the torch runtime and transient rerank activations are counted. The full
  reranker makes the two demos **mutually exclusive**, adjudicated by an OOM killer that picks its
  own victim → **`mem_limit` per container** so the blast radius stays local, and a
  **pre-registered rule** against a **~4.5 GB** budget: prod runs the ladder-winning arm if it
  fits, else the cheapest that does, with the divergence recorded in the README.
- **The reranker is now a RAM lever as well as a quality lever**; int8 ONNX is the one lever that
  could rescue the full one.
- **Qdrant joins the sleep group.** The VPS rule sleeps the whole group, so `/health` had to grow a
  second clause — and later a third.
- **`traefik.docker.allownonrunning=true` is mandatory**, or sleep becomes a permanent outage.
- **Client-controlled history** is a named injection surface. Bounded on two sides: server-side
  trimming, and `cited ⊆ retrieved_context`, which a forged history cannot defeat.
- **Uptime-Kuma at a 6 h interval is an outage detector**, not a deploy check — and the bot filter
  must be written as a **denylist**, or it traps the monitor.
- **Recorded edge case**: an HTMX POST or SSE connection arriving at a *sleeping* container gets
  Sablier's 200-with-HTML waiting page. Two small fixes, a build-time choice.
