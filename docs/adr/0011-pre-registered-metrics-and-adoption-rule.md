# ADR-0011 — Pre-registered primary metrics and a paired-delta adoption rule, with no significance gate

- **Status**: Accepted — 2026-08-03
- **Ticket**: [#12](https://github.com/Zameloth/rag_assurances/issues/12)
- **Spec**: [`SPEC.md` §12.5–12.12](../../SPEC.md#12-evaluation)

## Context

The ladder has six rungs and ~44 items. Langfuse's managed Context Precision and Context Recall
do not implement the metrics they are named after, so all retrieval quality is hand-rolled. At
this N, the temptation to reach for a p-value — or to pick a rung's metric after seeing its
numbers — is the real threat to the whole exercise.

## Decision

**Two datasets, two regimes. A deterministic retrieval ladder decided by a rule committed before
any rung runs, and a generation eval where three of five metrics need no judge.**

- **Eight retrieval numbers, four of which decide**: fiche recall@4, article recall@4,
  **zero-article rate**, **floor correctness**. The rest are diagnostic.
- **One primary metric named per rung, in a table committed before the ladder runs.**
- **Adopt iff net discordant pairs on the primary ≥ 4** *and* no other decision metric regresses
  by more than 1 net item. **Otherwise keep the incumbent. Inconclusive always resolves to no
  change.**
- Sign-test *p* is computed and persisted but **is never the gate**.
- **Per-item scores are persisted to git**, not aggregates. `compare.py` is the arbiter;
  **Langfuse is the trace viewer, not the comparison surface.**

## Rationale

- **Recall is monotone in retrieving more.** A retriever that always fills all four article slots
  maximises article recall@4 and can never be punished for it — and the only items that punish
  over-retrieval are the 8 `reponse_sans_article`, which recall cannot score. Without floor
  correctness and zero-article rate, rungs 4 and 5 are scored on a metric that structurally
  favours the more aggressive arm, and the relevance floor is measured by nothing.
- **No MRR/nDCG**, but the *right* reason: `gold_articles` is multi-gold and MRR keys on the
  first hit; articles arrive partly by editorial join rather than by rank; and the pipeline cuts
  hard at 4, so rank movement above the cut is never cashed in. *(The tempting stronger claim —
  "the architecture discards rank" — is wrong, and should not be repeated: the cut to 4 slots is
  itself a rank operation.)* Graded sensitivity comes from **depth curves** instead.
- **Significance testing is unavailable at this N.** Clearing p<0.05 two-sided needs ~6 discordant
  pairs all one-way, so a significance gate rejects genuine improvements *and* invites the
  "p = 0.11, close enough" rationalisation. **A p-value here launders a judgement call as a null
  result.**
- **Rule 3 does the real work.** Rungs 5 and 6 are *expected* to be inconclusive; under this rule
  that is a clean recorded outcome rather than an argument.
- **Two datasets, not one.** The ladder is deterministic and API-free; generation eval is
  LLM-in-the-loop, noisy and paid on both sides. A single always-full-chain task would burn spend
  on every rung of a *retrieval* experiment and inject generation noise into the one measurement
  chain that could have been noise-free.
- **The judge is family-separated and validated against a built set.** "Calibrate against ~20
  sampled items" is worthless — ~18 of 20 are fine, both raters agree, and nothing is learned.
  **12 clean/faulted pairs**, detection on ≥ 10 of 12, and **systematic leniency disqualifies
  regardless of rate**, because false passes are exactly what this eval exists to catch.
- **Per-item scores, not aggregates**: the adoption rule counts items across two runs, the ladder
  spans weeks, and free-tier retention is 30 days.

## Consequences

- **Corrections to the retrieval design**: rung 5's primary becomes zero-article rate; rung 6's
  arm must be **e5-dense + M3-sparse**; the reranker must be hand-wrapped in a span or rung 4
  measures nothing.
- **Two pre-ladder A/Bs, no seventh rung.** They cost no statistical power because the two
  registers' recalls are already separate numbers — and a *joint* arm would be the lossy option.
- **The calibration set outlives its first use** as a judge regression test — the only mechanism
  by which "the instrument didn't move" is checkable.
- **The French-vs-English judge question is discharged empirically**, not by argument.
- **`LANGFUSE_TRACING` defaults false in dev.** The budget fails on *debugging*, not experiments:
  interactive tracing at a hundred queries a day exceeds the entire ladder. Sampling was
  rejected — a 10% sample hands you a random trace when you need the one you are confused about.
- **Known risk**: the pre-registered table is only protective if committed **before** the ladder
  runs. Written afterwards, per-rung primaries become post-hoc metric selection.
- **Accepted gap**: explanation *precision* stays unmeasured. A critique judge was rejected as
  the only number in the system with no falsifiable target.
