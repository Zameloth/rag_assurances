# ADR-0026 — Self-hosted on aspire: the repo prepares, the host deploys

- **Status**: Accepted — 2026-10-12. Supersedes the deployment premise of
  [ADR-0013](0013-fastapi-htmx-app-sleeping-behind-traefik.md) (VPS, Traefik v2.11, Uptime-Kuma);
  its application decisions stand.
- **Ticket**: [#54](https://github.com/Zameloth/rag_assurances/issues/54)
- **Spec**: [`SPEC.md` §14](../../SPEC.md#14-deployment), [§15.7](../../SPEC.md#157-mismatch--fail-loud-with-the-ordering-scripted), [§17.4](../../SPEC.md#174-the-xhr-arriving-at-a-sleeping-container-edge-case)

## Context

The deploy target moved from the VPS to **aspire**, a self-hosted laptop with **32 GB of RAM**,
reached only through a **Cloudflare Tunnel** that terminates TLS in front of **Traefik v3.7**.
Deployment is owned by a separate infra repo, `self-host`: it lists the project in its catalogue,
clones this repo, writes a `.env` at its root, and runs `docker compose build && up -d` — or a
per-project override — on every new commit to `main`. Sablier is installed there. Uptime-Kuma is
gone: a project in sleep is probed by Prometheus through **Sablier's blocking API**, every 30 min,
which answers 200 only once the group is ready.

Three of #54's premises dissolved with the VPS: the ~4.5 GB RAM budget, `make deploy` as the
operator's command, and the Uptime-Kuma keyword monitor.

## Decision

**This repo carries a self-sufficient `compose.yml` and nothing that deploys.**

- **`compose.yml` at the root is prod**, the file `docker compose` picks by default; the dev store
  moves to `compose.dev.yml`, behind `make up`.
- **No `make deploy`, no CD.** The deploy sequence is handed to `self-host` as the project's
  override: `build` → `--profile restore run --rm restore` → `up -d --wait --remove-orphans`.
  `--wait` fails unless every container is healthy, and the app's healthcheck *is* the
  three-clause `/health` — so the deploy still verifies itself, without a `curl` through the tunnel.
- **Restore is its own `restore` service**, behind a profile, on the app image. `run
  rag-assurances` would start a one-off container carrying the Traefik and Sablier labels.
- **The BGE-M3 weights are baked into the image**, at the revision `index_lock.json` pins, in a
  build stage that depends on the lock alone; the runtime is `HF_HUB_OFFLINE=1`.
- **Traefik v3 syntax**: bot denylist as `HeaderRegexp` on a priority router whose service is
  `noop@internal` and whose only middleware is an `ipAllowList` nothing matches — the 403.
  Rate limits are keyed on **`CF-Connecting-IP`**, with a stricter one on the three ask routes.
- **§17.4 is fixed client-side**: the page probes `/health` before each submit and, while the
  answer is not the app's JSON, waits under a « Réveil » stage.
- **Only secrets come from `.env`**; the model ids and `LANGFUSE_TRACING=true` are in
  `compose.yml`, so which arm prod runs is in git.
- **Monitoring is `self-host`'s**: group `rag` goes in its `veille.yml`.

## Rationale

- **Weights in the image rather than a volume filled at deploy.** A volume needs a prefetch step
  in the override and a second thing to keep in sync with the lock; a first start that downloads
  2.3 GB is the wake-path network call SPEC §15.5 forbids. Fetching `main` would also make the
  query encoder whatever BAAI pushed last, not the revision that embedded the served index.
  *Price, accepted*: a 2.3 GB layer — paid once per lock change thanks to the separate stage, on a
  host with the disk for it.
- **Client pre-flight over Sablier's blocking strategy on `/ask`.** A blocking request is held
  until the group is ready, but Cloudflare cuts an origin response at **100 s**, which a cold
  start can approach; and the visitor would watch a frozen indicator. The pre-flight keeps the
  question and history in the page, says what is happening, and costs one GET per question. Any
  JSON answer counts as awake — a 503 included, since dev's `/health` is 503 by design and a real
  failure is the answer route's to report.
- **`--wait` over `curl -f https://rag.theo-eloy.fr/health`.** The curl tested the tunnel too —
  which `self-host` already probes on its own — and needs the host to resolve its own public name.
  `--wait` tests exactly the `/health` the deploy can break.
- **`CF-Connecting-IP`**, because behind the tunnel every request's remote address is
  cloudflared's: keyed on it, the rate limit would be one bucket shared by every visitor — a
  global throttle, not a per-visitor one.

## Consequences

- **The RAM rule is moot.** Rung 1 ships (ADR-0024) — BGE-M3 alone, no reranker — and the host has
  32 GB. `mem_limit` stays (6 GB app, 1 GB Qdrant) for its other job: a blown budget kills this
  container, not a neighbour. No restart policy, so an OOM kill waits for the next visit.
- **Prod tracing is one `ask` trace per question** (`rag.app.tracing`): the condenser and
  generation calls as generation observations via LangChain's callback handler, retrieval as a
  `retriever` observation through a `retrieve_fn` seam on `run_chain` — `retrieve` itself, wrapped,
  so the served path stays the measured one. ~9 units a question.
- **The OpenRouter credit cap is a dashboard act**: prod gets its own key, with a credit limit.
- **Bots get no link preview** — a shared link renders bare. The same trade SPEC §14.5 made.
- **Migrating dev**: the dev store's project name changed, so the old container must be removed
  once (`docker compose -p rag_assurances -f compose.dev.yml down`); its volume is kept by name.
