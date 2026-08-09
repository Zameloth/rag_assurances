# Coding standards — rag_assurances

What `/code-review`'s Standards axis checks against, consolidated from `pyproject.toml`, `CONTEXT.md`
and the ADRs. If a rule here conflicts with something in `SPEC.md` or an ADR, the ADR wins — file
an issue to update this doc.

## Tooling (enforced by `make check` — don't re-flag what CI already catches)

- Python `>=3.12,<3.14`.
- `mypy --strict` over `src/`, `tests/`, `scripts/`.
- `ruff check .` — line length 100, rule set `E4 E7 E9 F I UP B SIM`. `E5` (line-length lint) is
  deliberately off: several files predate the config with longer lines, and retrofitting them is
  a separate change from turning the linter on.
- `pytest` with coverage, `fail_under = 90` (`tool.coverage.report`).

## Naming — use `CONTEXT.md`'s vocabulary

`CONTEXT.md` is the ubiquitous-language glossary. Use its terms in code, tests, issue titles and
prompts; avoid the synonyms it marks ✗ (e.g. *point* not *vector*/*record*/*row*, *the band* not
*chunk size*, *expansion* not *fiche-to-article lookup*). If a concept you need isn't in the
glossary, that's a signal: either you're inventing language the project doesn't use, or there's a
real gap worth raising for `/domain-modeling`.

## Design conventions ([CONTEXT.md](CONTEXT.md#conventions-worth-stating))

- **Measure before deciding.** Several past decisions reversed on a measurement, not an assumption
  (`texte` has zero newlines, 18% of article numbers have 3+ segments, etc.). New code making a
  data-shape assumption should point at what was measured, not assert it.
- **A typed field beats an inference.** When a component takes one of several branches
  (`motif`, `condense_status`, `expected_state`), record which branch was taken as a typed field
  rather than letting a caller reconstruct it downstream. Reach for this whenever a diff adds a
  new discriminated outcome.
- **Failures must localise.** Keep metrics/checks for distinct failure modes distinct (candidate
  vs. final recall, citation correctness vs. article recall, the condensed query kept out of
  generation) rather than folding them into one signal that can't say which thing broke.
- **Inconclusive keeps the incumbent.** Applies to any adoption decision, not just the retrieval
  ladder — a change without a clear win doesn't ship.

## Architectural rules (from ADRs / SPEC — hard violations, not judgement calls)

- **Config only from `.env`, never inline** (SPEC §16.3, [ADR-0012](docs/adr/0012-langfuse-cloud-and-tracing-gates.md)).
- **Qdrant via the native `qdrant-client`, never a LangChain `VectorStore` wrapper** (SPEC §6.1,
  [ADR-0005](docs/adr/0005-qdrant-two-collections-behind-aliases.md)).
- **No physical collection name in code** — an index-bearing arm is addressed through its stable
  alias; switching arms is an alias update ([ADR-0005](docs/adr/0005-qdrant-two-collections-behind-aliases.md)).
- **Sources and decisions are committed, never derived binaries** — the golden set and per-item
  eval scores are the only named exceptions
  ([ADR-0002](docs/adr/0002-commit-the-filtered-corpus-never-derived-binaries.md)).
- **`compare.py` and the web app share one pipeline code path**; the app adds no retrieval/
  generation logic of its own — otherwise the evaluated system and the served system drift.
- **The app is stateless.** History is client-side, posted back each turn, and is therefore
  untrusted input — trim/validate it server-side, never trust its shape.
- **`cited ⊆ retrieved_context`**, not `⊆ corpus`, and this check is never auto-repaired
  ([ADR-0009](docs/adr/0009-typed-answer-envelope-and-citation-containment.md)).
- **Per-item eval scores are what's persisted**, not just aggregates — Langfuse's free tier drops
  traces at 30 days, so an aggregate-only result can't be reconstructed later.

## Testing

- Real-engine tests take the `qdrant_server` fixture and skip when nothing answers at
  `QDRANT_URL`; the suite must stay green without a live store.
- `QdrantClient(":memory:")` is for plumbing assertions only — never for recall or ranking numbers
  (SPEC §6.3). A test asserting a recall/ranking number against the in-memory client is a
  standards violation, not just a weak test.

## Baseline smell check

On top of the above, `/code-review`'s Standards axis always applies the Fowler smell baseline
(Mysterious Name, Duplicated Code, Feature Envy, Data Clumps, Primitive Obsession, Repeated
Switches, Shotgun Surgery, Divergent Change, Speculative Generality, Message Chains, Middle Man,
Refused Bequest) as judgement calls, not hard violations. Where this file endorses something the
baseline would otherwise flag, this file wins.
