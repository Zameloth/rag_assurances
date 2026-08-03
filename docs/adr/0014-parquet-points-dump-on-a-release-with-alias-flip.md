# ADR-0014 — The index ships as a Parquet points dump on a GitHub Release, made live by an alias flip

- **Status**: Accepted — 2026-08-03
- **Ticket**: [#18](https://github.com/Zameloth/rag_assurances/issues/18)
- **Spec**: [`SPEC.md` §15](../../SPEC.md#15-index-delivery-and-operations)

## Context

Two earlier decisions squeezed this into a real question: Qdrant now **sleeps with the app** on a
10-minute session, so re-embedding on VPS CPU is not viable; and the Qdrant snapshot is **not repo
content**, so `git clone` is not the channel.

The obvious answer was a Qdrant snapshot pulled by `snapshots/recover`, which accepts a remote URL
and a SHA256 checksum — zero new code.

## Decision

**A Parquet points dump, one file per register, published as a GitHub Release asset, restored by
the ingest library into `<register>__<release-tag>` and made live by an alias flip gated on a
verified count. Restore is a deploy step, never a boot step. Built locally as the tail of a ladder
run.**

`index_lock.json` is committed as the provenance pointer; mismatch **fails loud**, with the
ordering scripted into `make deploy`.

## Rationale

- **The snapshot lost to a decision already made.** `indexing_threshold=0` means HNSW never
  builds, so **there is no built index for a snapshot to preserve**. Stripped of that it is worse
  transport: version-coupled, and it seals the *collection configuration* — the named-vector
  layout, the payload indexes, the threshold ruling itself — inside an opaque binary. With a points
  dump, **only the vectors are derived**; the configuration stays in git as the code that creates
  the collection, which is the same code dev runs.
- **The anti-parquet argument inverts.** Binary was banned for the corpus because it defeats human
  review; **this artifact is never reviewed**, only sha256-verified. The decisive property is
  **exactness** — serializing 3.8M floats through text is the one step that could silently change
  one, and Parquet stores fp32 natively.
- **Built locally, not in CI.** A CI rebuild is a *re-derivation* on different hardware and a
  different torch build — numerically almost identical, and **not the artifact that was scored**.
  Publishing the exact file the ladder ran against extends the demoed-must-equal-measured rule from
  the schema to the vectors themselves, and costs nothing.
- **The trigger is not an event listener.** Every candidate trigger — corpus refresh, a chunking
  change, a payload change, a new ladder winner — is *already a deliberate reviewed human act*.
  Publishing is a runbook step.
- **Sleep-safety comes free from the aliases.** An SSH restore does not refresh the Sablier session
  and can be killed mid-upsert — but a partial write **never acquires the alias**, so the failure
  mode is "re-run", not "half an index". The flip is atomic.
- **Boot-time self-healing was rejected**: it puts `github.com` in the critical wake path, turning a
  visitor's first page load into a cold-start failure.
- **Serving a stale index with a warning was rejected** — it makes the demoed system knowingly not
  the measured system.

## Consequences

- **"Which arm ships" decomposes** into **index-bearing** (embedder, chunk config, enrichment) and
  **runtime** (reranker, weights, floor, cap). The RAM rule is **runtime only** and does not reach
  the published artifact — **except if rung 6 wins**, which puts two embedders co-resident.
- **`/health` gains a third clause**: the alias target must match `index_lock.json`. The alias name
  carries the release tag, so this identity check costs nothing.
- **Uptime-Kuma's 6 h interval cannot double as a deploy check**, so `make deploy` ends in its own
  `curl -f /health`.
- **Any public artifact carrying corpus payloads must travel with `corpus_manifest.json`** — the
  dump contains verbatim DILA text, so publishing it is a *redistribution* under LO 2.0 and needs
  the attribution attached to the artifact, not merely to the repo.
- **The previous generation stays on disk**, so rollback is one alias flip and zero bytes
  downloaded.
- **Named, not solved**: between the alias flip and the app restart the old app queries the new
  collection. Harmless when only vectors changed; a payload-schema move could fail a query in that
  few-second window.
