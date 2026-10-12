# The judge calibration set

`judge-set.yaml` — 12 pairs, 24 answers, every one human-labelled (SPEC §12.10, #48).
Each pair is a **real pipeline answer** and a **faulted twin** carrying exactly one planted
fault; the judge passes the set when it scores the twin strictly lower on ≥ 10 of 12 pairs
with no false pass.

## It is a regression test, not a one-off

The set exists to answer one question every time the scoring side moves: *did the
instrument move?* Re-run it whenever

- `JUDGE_MODEL` changes,
- a judge prompt changes (`rag.eval.judge`, or the prompt language config),
- OpenRouter resolves a different provider for the judge (the run records the resolved
  provider — compare it against the last run's).

```sh
uv run python scripts/validate_calibration_set.py   # structure, against the golden set
uv run python scripts/run_judge_calibration.py      # ~24 judge calls, writes eval/runs/calibration-*.json
```

A pair is self-contained (question, expected points, the rendered context, the citable ids),
so a re-run never touches Qdrant and never drifts with the index: if the verdict changes,
the judge changed.

Do **not** regenerate the clean answers to "refresh" the set. A new pipeline answer is a new
fixture, which needs new human labels; editing a pair goes through
`scripts/author_calibration_pair.py`, which validates before it writes.

## Composition

| Archetype | Target metric | Pairs |
|---|---|---|
| `fabricated_citation` | faithfulness | gs-013, gs-015, gs-043, gs-046 |
| `claim_absent_from_context` | faithfulness | gs-037, gs-044, gs-045 |
| `dropped_expected_point` | point coverage | gs-010, gs-016, gs-032 |
| `refusal_without_explanation` | point coverage | gs-051, gs-054 |

All 12 clean answers are labelled `pass`, all 12 twins `fail`.

**Faults are taken from the corpus, so they are plausible by construction.** Fabricated
citations are real articles that were *not* retrieved for the item — on-topic near misses
from the article leg (e.g. L121-10 for gs-013, L212-1 for gs-015), the real-but-wrong class
that survives a corpus-wide existence check. Absent claims are real rules lifted from
another article and misapplied (e.g. the *assurance frontière* 30/90-day limit in gs-037,
L113-2's notice period in gs-044/gs-045). The twins were drafted with Claude Code from that
corpus material; every label is the author's, given after reading each clean answer against
its context.

**Only two refusal pairs.** Of the twelve golden refusal items with expected points, only
gs-051 and gs-054 produced a real answer a human could label `pass` — the others did not
refuse, refused bare, missed points, or cited articles that were never retrieved. The
twelfth pair went to `fabricated_citation` instead.

## Known caveats — labelled `pass` with one unsupported detail

Two clean answers carry a detail the context does not support. The author kept them `pass`
because the detail is outside the metric the pair is scored on — calibration scores each
pair on its archetype's target metric only (`TARGET_METRIC`).

- **gs-032** (point coverage) — the clean answer says cat-nat cover needs *"un arrêté
  interministériel"*; no retrieved context says so, though the golden item's third expected
  point requires it. That sentence is exactly what the twin drops.
- **gs-051** (point coverage) — the clean answer cites *L114-1* inline, flagging itself that
  it is not in the context (the two-year delay it attaches to is). It also suggests chasing
  the insurer, which the context does not mention.

If the set is ever scored on **faithfulness** for these two answers, a judge marking them
down is right, not lenient — relabel them before reading that as a false fail.
