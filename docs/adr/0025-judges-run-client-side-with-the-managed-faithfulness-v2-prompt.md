# ADR-0025 — The judges run client-side, with the managed Faithfulness v2 prompt, so routing stays pinned and the resolved provider is recorded

- **Status**: Accepted
- **Ticket**: #47
- **Spec**: SPEC.md §11.1, §12.9, §12.10, §12.11, ADR-0011, ADR-0012

## Context

SPEC §12.10 asks for two things that pull against each other:

1. **"Start with managed Faithfulness v2"**, Langfuse's server-side LLM-as-a-judge template.
   It is the only managed RAG judge that implements the procedure it is named after
   (`docs/research/langfuse-rag-eval.md` §5.3).
2. **`allow_fallbacks: false`, a pinned `provider.order`, and the resolved provider recorded in
   every persisted run.** OpenRouter routing is a variance source inside the eval harness.

A managed evaluator runs on Langfuse's workers through an **LLM connection**. Checked against
the installed SDK (`langfuse` 4.15, `api/llm_connections`, `api/evaluators`), a connection
carries a provider name, an adapter, a key, a base URL, a model list, extra *headers* and (for
OpenAI) a `useResponsesApi` flag. It has **no request-body field**, and OpenRouter's `provider`
routing block is a body field. An OpenRouter preset used as the model id could pin the routing,
but nothing can bring the resolved provider back. The judge call is also made by Langfuse, not by us, so the
response's `provider` field never reaches anything we persist. So a managed evaluator can meet
(1) but not (2). A third problem: its scores land on traces asynchronously, which suits
`dataset.run_experiment()`'s per-item persistence (SPEC §12.11) poorly, and suits a calibration
harness scoring hand-authored answers (which have no pipeline trace) even worse.

## Decision

**Both judges run client-side**, as `dataset.run_experiment()` evaluators for generation runs
and directly in `rag.eval.calibration` for calibration runs. They share one `Judge`
(`rag.eval.judge`) behind one `JudgeCall` seam (`rag.eval.judge_chain`):

- `ChatOpenAI` on OpenRouter with `require_parameters`, `allow_fallbacks: false` and a
  one-element `order` from `JUDGE_PROVIDER`.
- **No `temperature` and no `parallel_tool_calls` in the request.** The first live call showed
  that no Claude Sonnet 5 endpoint supports either, and `require_parameters` turns an
  unsupported parameter into a 404, not a silent drop. The judge can't be pinned to
  temperature 0. The calibration set is what shows its scores are stable enough, which is one
  more reason re-running it is the regression test.
- `ProviderReportingChatOpenAI` keeps OpenRouter's `provider` response field, which
  `ChatOpenAI` otherwise drops. `with_structured_output(include_raw=True)` hands it through.
  **A response without a provider raises**; it is never pinned as an empty string.
- **A failed judge call fails the run, after persisting it.** `dataset.run_experiment()`
  swallows evaluator exceptions, which would turn a failure into a bare `None`. So the evaluator
  catches each metric's failure itself and records it as the item's `judge_error`. The run is
  written, and then `JudgeRunError` is raised.
- Extraction is via **function calling**, the same path managed judges use (SPEC §11.1).

**"Managed Faithfulness v2" means its prompt, verbatim.** `FAITHFULNESS_V2_PROMPT_EN` is the
template from Langfuse's `managed-evaluators.json` (v2, 2026-04-17), and `FaithfulnessOutput`
is its `outputDefinition`. What moves is where it runs, not what it asks or how the answer is
read.

**Point coverage is ours**, French-prompted by default. The judge returns one verdict per
numbered `expected_point` and **the score is computed in code**. A verdict set that doesn't
match the points one-for-one raises; it is never read as "not asserted".

**Prompt language is config, per metric** (`JUDGE_FAITHFULNESS_LANGUAGE`,
`JUDGE_POINT_COVERAGE_LANGUAGE`, defaults `en`/`fr`), and every run header records it. The
FR/EN A/B is one environment variable over the same 24 calibration answers.

**The judge sees the whole envelope**, rendered by `render_answer`: explanation, every
`fondement_juridique` entry, and the motif or `aucun_fondement`. A fabricated citation lives in
the typed field, not in the prose.

**The Langfuse OpenRouter connection is still configured** (`scripts/configure_langfuse_judge_connection.py`:
provider name `openrouter`, adapter OpenAI, gateway base URL, only `JUDGE_MODEL` exposed). It
makes the managed evaluators usable from the UI, for example live on production traces, but
no persisted eval score goes through it.

**Calibration** (`rag.eval.calibration`) scores each pair on the one metric its fault archetype
targets. It reports per pair whether the twin scored strictly lower, and gives each answer's
error direction against its **human label**: false pass or false fail, at `pass_threshold`
(default 1.0, since a twin is labelled *fail* because one thing in it is wrong). Errors are
counted per metric, and **any false pass disqualifies**. False fails are reported, but they
never offset a false pass: counting leniency net of harshness would let a judge that fails
clean answers hide one that passes faulted twins. The authoring helper asks the author to label
both answers; the real answer is not assumed clean.

The French faithfulness prompt is a straight translation of the English one's four steps, with
nothing added, so the FR/EN A/B varies language only.

## Consequences

- The judged scores are reproducible to the same standard as the generation arm. Each is pinned
  to a provider that is *observed*, not assumed. That is stronger than
  `RunHeader.generation_provider`, which still records the requested provider.
- If Langfuse revises Faithfulness v2, nothing here follows it automatically. That is
  deliberate: the calibrated instrument is this string, and a silent server-side prompt change
  is exactly the drift SPEC §12.10's regression test exists to catch.
- Judge calls are not Langfuse evaluator traces. Their scores reach Langfuse as experiment
  scores, with the reasoning as the comment and the resolved provider in the metadata.
- Generation runs from before #47 still load: the judged fields default to `None`. A Langfuse
  generation dataset synced before #47 lacks `expected_points` and fails loudly until re-synced.
- **Not done here: the Ragas fallback.** SPEC §12.10 says to build it only if calibration fails,
  and calibration has not run yet. It needs the 12 hand-authored pairs (the human ticket this one
  unblocks).
