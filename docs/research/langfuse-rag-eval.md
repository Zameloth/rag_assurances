# Langfuse for RAG evaluation, and how it wires into LangChain

**Research ticket:** [Zameloth/rag_assurances#5](https://github.com/Zameloth/rag_assurances/issues/5)
**Date of research:** 2026-08-01
**Scope:** establish Langfuse's *real* capabilities so the later eval-design tickets are built on facts, not assumptions.
**Project context:** solo learning/portfolio RAG over French insurance documents. LangChain + Langfuse are fixed by the project premise. Decided eval depth: **tracing + a golden eval set** — retrieval and answer quality must be measurable, not vibes.

## Method and source policy

Every claim below is cited. Sources are ranked:

- **P1 — code**: the Langfuse server repo (`langfuse/langfuse`), the Python SDK repo (`langfuse/langfuse-python`), PyPI release metadata. This is ground truth.
- **P2 — official docs**: `langfuse.com/docs`, `langfuse.com/pricing`.
- **P3 — first-party long-form**: the Langfuse blog / cookbooks in `langfuse/langfuse-docs`. Written by the vendor, but marketing-adjacent; used only for worked examples.

Where the docs (P2) and the code (P1) disagree, **the code wins** and the disagreement is flagged. This matters more than usual here: a major version shipped three days before this research was written, and parts of the docs still describe the previous one.

---

## 0. Version currency — read this first

This is the single most important section, because Langfuse shipped a major version *days* ago and much of the public documentation has not caught up.

| Component | Current version | Evidence |
| :--- | :--- | :--- |
| Langfuse server | **v4.2.0** (2026-07-31) | `gh api repos/langfuse/langfuse/releases` |
| Langfuse server v4 GA | **v4.0.0, 2026-07-29** | same |
| Langfuse server v3 (still maintained) | v3.224.4 (2026-07-30) | same |
| Python SDK | **4.14.2** (2026-07-30) | [PyPI `langfuse`](https://pypi.org/pypi/langfuse/json), `requires_python >=3.10,<4.0` |
| JS/TS SDK | v5 | [Versions & Compatibility](https://langfuse.com/docs/compatibility) |

Notable: **server v4.0.0 went GA on 2026-07-29 and v4.2.0 landed on 2026-07-31 — one and two days before this document was written.** Langfuse is still shipping v3 patches in parallel (v3.224.4 on 2026-07-30), so v3 is maintained, not dead.

### What v4 changed

Langfuse v4 is an **"observations-first data model built for agentic systems"** — you can "query any LLM call, tool execution, or agent step directly, without costly joins or read-time deduplication" ([v4 docs](https://langfuse.com/docs/v4)). The headline is performance: "Dashboard load times for large projects improve by at least 10x."

Two v4 changes that touch eval design:

1. **Trace-level input/output is deprecated.** Data is now read from the relevant *observations* instead ([v4 docs](https://langfuse.com/docs/v4)). Anything that assumed "the trace has an input and an output" needs to name an observation instead.
2. **Cloud rollout dates**: organizations created after **2026-04-14** run v4 by default; Cloud projects created after **2026-05-20** are locked to enriched observations export ([v4 docs](https://langfuse.com/docs/v4)). A project we create today is a v4 project. There is no v3 decision to make.

### Compatibility rule

> "Each Langfuse server major version aims to support the current and the previous SDK major version of each language."
> — [Versions & Compatibility](https://langfuse.com/docs/compatibility)

Upgrading a server v3 → v4 keeps Python SDK v3+/v4 and JS SDK v4/v5 working. Python SDK v2 and JS SDK v3 lose real-time trace visibility on v4, and trace ingestion via the legacy API is unsupported on the v4 data model ([compatibility](https://langfuse.com/docs/compatibility)). **Langfuse Cloud always runs the latest server**, so on Cloud we only track our own SDK version.

### ⚠️ Documentation lag — flagged explicitly

The [LangChain integration page](https://langfuse.com/integrations/frameworks/langchain) is still labelled **SDK v3.x.x**, even though Python SDK v4 is current and is what `pip install langfuse` gives you today. I verified against the SDK source rather than trusting the page (see §1). **The good news: the LangChain import path and basic usage are unchanged between v3 and v4** — so the page's snippets still work. The v3→v4 differences that *do* bite are listed in §1.3.

---

## 1. Tracing: how the LangChain integration actually works

### 1.1 Mechanism — it is a callback handler

Yes, it is a standard LangChain callback handler. Langfuse hooks LangChain's callback mechanism:

> "The Langfuse `CallbackHandler` automatically captures detailed traces of your LangChain executions, LLMs, tools, and retrievers to evaluate and debug your application."
> — [LangChain tracing docs](https://langfuse.com/docs/integrations/langchain/tracing)

Verified in the SDK source: `langfuse/langchain/CallbackHandler.py` (1815 lines) implements `on_chain_start`/`on_chain_end`/`on_chain_error`, `on_llm_start`/`on_llm_end`/`on_llm_error`, `on_tool_start`/`on_tool_end`/`on_tool_error`, `on_retriever_start`/`on_retriever_end`/`on_retriever_error`, and `on_llm_new_token` ([source](https://github.com/langfuse/langfuse-python/blob/main/langfuse/langchain/CallbackHandler.py)).

Practical consequence: **tracing is nearly free to add.** You pass one object into `config={"callbacks": [...]}` and every LangChain primitive in the chain reports itself. No manual instrumentation of the happy path.

### 1.2 Setup code (from the official docs)

```python
# pip install langfuse langchain langchain_openai
```

Environment variables ([docs](https://langfuse.com/docs/integrations/langchain/tracing)):

```bash
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_BASE_URL="https://cloud.langfuse.com"   # EU: https://cloud.langfuse.com
```

The canonical snippet ([LangChain integration page](https://langfuse.com/integrations/frameworks/langchain)):

```python
from langfuse import get_client
from langfuse.langchain import CallbackHandler

# Singleton client, reads the env vars above
langfuse = get_client()

# The handler takes NO constructor args for trace attributes
langfuse_handler = CallbackHandler()

result = chain.invoke(
    {"question": "Quelle est la franchise sur un bris de glace ?"},
    config={"callbacks": [langfuse_handler]},
)
```

Trace attributes (session, user, tags) are set through **magic `langfuse_`-prefixed metadata keys** on the invoke config, not on the handler ([docs](https://langfuse.com/integrations/frameworks/langchain)):

```python
response = chain.invoke(
    {"question": "..."},
    config={
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_user_id": "user-5678",
            "langfuse_session_id": "session-1234",
            "langfuse_tags": ["prod", "fr-insurance"],
        },
    },
)

# Retrieve the trace id in order to attach a score later
trace_id = langfuse_handler.last_trace_id

langfuse.create_score(
    trace_id=trace_id,
    name="correctness",
    value=1,
    data_type="NUMERIC",
)

langfuse.shutdown()   # required in short-lived scripts — the SDK batches
```

`last_trace_id` is real: it is a declared attribute on the handler (`self.last_trace_id: Optional[str] = None`, line 192 of `CallbackHandler.py`) and is assigned on chain start and on generation start ([source](https://github.com/langfuse/langfuse-python/blob/main/langfuse/langchain/CallbackHandler.py)).

⚠️ **`langfuse.shutdown()` matters for us.** The SDK batches and flushes asynchronously. A script that runs an eval and exits immediately can lose the tail of its data. Every eval script we write must flush before exit.

### 1.3 v3 → v4 differences that affect us

From the [Python v3 → v4 upgrade path](https://langfuse.com/docs/observability/sdk/upgrade-path/python-v3-to-v4):

- **Import path is unchanged**: `from langfuse.langchain import CallbackHandler`. (This is why the v3-labelled docs still work.)
- **`CallbackHandler(update_trace=True)` now raises `TypeError`** — the parameter was removed. Use `propagate_attributes()` instead.
- `update_current_trace()` is split into `propagate_attributes()` (for `user_id`, `session_id`, `metadata`, `tags`, `version`), `set_current_trace_io()` (deprecated), and `set_current_trace_as_public()`.
- **Unified observation API**: `start_span()` → `start_observation(name=...)`; `start_generation()` → `start_observation(name=..., as_type="generation", ...)`. Same for the `start_as_current_*` variants.
- **`DatasetItemClient.run()` is removed entirely** — replaced by `dataset.run_experiment(...)`. Any pre-v4 tutorial using `item.observe()` / `item.run()` is dead code. **This is the single biggest trap when following older RAG-eval tutorials.**
- Pydantic v1 support dropped (v2 required). Metadata values are `dict[str, str]` with a **200-character limit per value**; `user_id`/`session_id` are validated as strings, max 200 chars.
- Smart span filtering: by default only spans from the Langfuse SDK, spans with `gen_ai.*` attributes, or known LLM instrumentation scopes are exported. Non-LLM infra spans (HTTP, DB, queues) are filtered out.

⚠️ The **200-char metadata value limit** is a real constraint for us: we cannot stuff a retrieved French policy chunk into trace metadata. Chunks belong in observation input/output, which is where they naturally land anyway (§1.4).

### 1.4 Granularity — what a RAG trace actually captures

**Retrieval is captured as a first-class observation type, with the documents.** Verified in source, not inferred. `on_retriever_start` opens an observation with `as_type` derived from `_get_observation_type_from_serialized(serialized, "retriever", ...)` and sets `input=query`:

```python
observation_type = self._get_observation_type_from_serialized(
    serialized, "retriever", **kwargs
)
span = parent_observation.start_observation(
    name=span_name,
    as_type=observation_type,
    metadata=span_metadata,
    input=query,
    level=...,
)
```

and `on_retriever_end` writes the retrieved documents as the observation output:

```python
def on_retriever_end(self, documents: Sequence[Document], *, run_id, parent_run_id=None, **kwargs):
    observation = self._detach_observation(run_id)
    if observation is not None:
        observation.update(output=documents, input=kwargs.get("inputs")).end()
```

— [`CallbackHandler.py`, `on_retriever_start` / `on_retriever_end`](https://github.com/langfuse/langfuse-python/blob/main/langfuse/langchain/CallbackHandler.py)

So for a standard LangChain RAG chain we get, automatically:

| Pipeline step | Captured? | How |
| :--- | :--- | :--- |
| Overall request | ✅ trace | root chain run |
| Chain / RunnableSequence nesting | ✅ span/chain observations | `on_chain_start`/`on_chain_end` |
| **Retrieval** | ✅ **`retriever` observation**, query as input, **retrieved `Document` objects as output** | `on_retriever_start`/`on_retriever_end` |
| **Generation** | ✅ **`generation` observation** with model, token counts, cost | `on_llm_start`/`on_llm_end`, `as_type="generation"` |
| Tool calls | ✅ tool observations | `on_tool_start`/`on_tool_end` |
| Streaming tokens | ✅ | `on_llm_new_token` |
| Errors | ✅ per-step, with `level` | `on_*_error` |
| **Rerank** | ⚠️ **only if it is a LangChain component** | see below |

⚠️ **Reranking is the gap.** There is no `on_rerank` callback in LangChain. A reranker gets traced only if it is expressed as a LangChain component that itself fires callbacks — e.g. a `ContextualCompressionRetriever`, which *is* a retriever and so fires `on_retriever_*`. If we implement reranking as a plain Python function (a very common choice: call Cohere/JinaAI/a cross-encoder directly), **it will be invisible in the trace** unless we wrap it manually:

```python
with langfuse.start_as_current_observation(as_type="span", name="rerank") as span:
    span.update(input=candidates)
    reranked = my_reranker(query, candidates)
    span.update(output=reranked)
```

This is the same manual-wrap pattern Langfuse themselves use for retrieval in their own RAG example (§3.3). **Design implication: if the eval plan wants to measure rerank lift, budget for a hand-wrapped observation around the reranker.**

### 1.5 Data model — what a trace shows

> A trace represents "a single request or operation, for example one chatbot interaction from the user's question to the final response." Observations are "the individual steps of your application: LLM calls, tool calls, retrieval steps, and so on."
> — [Observability data model](https://langfuse.com/docs/observability/data-model)

Observation types include span, generation, agent, tool, chain, **retriever**, embedding, evaluator, and guardrail ([data model](https://langfuse.com/docs/observability/data-model)). Trace-level attributes (`user_id`, `session_id`, `tags`, `metadata`) propagate across nested observations.

In practice a trace view gives us the nested tree, per-step input/output, latency per step, and token/cost on generations. For debugging a bad French-insurance answer this is exactly the right shape: you open the trace, look at the `retriever` observation's output documents, and immediately see whether the failure was **retrieval** (wrong chunks came back) or **generation** (right chunks, bad answer). That separation is the whole point of tracing a RAG pipeline, and we get it for free from the callback handler.

---

## 2. Datasets — the golden eval set

### 2.1 Shape

> "A dataset is a collection of inputs and expected outputs and is used to test your application."
> — [Datasets docs](https://langfuse.com/docs/evaluation/features/datasets)

A **dataset item** has three parts:

- `input` — any object (dict, string, …)
- `expected_output` — *optional* reference answer
- `metadata` — *optional* structured info

That `expected_output` is optional is important for us: it means reference-free metrics (faithfulness, answer relevance) can run on items where we never wrote a gold answer, while reference-based metrics (correctness, context recall) need the gold answer filled in. We can build the golden set incrementally.

```python
langfuse.create_dataset(
    name="assurances-golden-v1",
    description="Golden Q/A set, French insurance policies",
    metadata={"author": "Theo", "type": "benchmark"},
)

langfuse.create_dataset_item(
    dataset_name="assurances-golden-v1",
    input={"question": "Quelle est la franchise en cas de bris de glace ?"},
    expected_output={"answer": "..."},
    metadata={"policy": "MRH", "difficulty": "easy"},
    source_trace_id="<trace_id>",   # optional: promote a real production trace
)
```
— [Datasets docs](https://langfuse.com/docs/evaluation/features/datasets)

### 2.2 Versioning — automatic and timestamp-based

> "Every `add`, `update`, `delete`, or `archive` of dataset items produces a new dataset version."
> — [Datasets docs](https://langfuse.com/docs/evaluation/features/datasets)

Versions are timestamps. The API returns the latest by default; you pin a historical version explicitly:

```python
from datetime import datetime, timezone

version_timestamp = datetime(2026, 7, 15, 6, 30, 0, tzinfo=timezone.utc)
dataset_at_version = langfuse.get_dataset(
    name="assurances-golden-v1",
    version=version_timestamp,
)
```

✅ **This is genuinely good for us.** It means "did the score go up because the pipeline improved, or because I edited the dataset?" is an answerable question. Versioning is automatic — no git-for-datasets discipline required.

⚠️ But note the versions are *timestamps*, not semantic labels. Reproducing "the run I did in March" means recording the timestamp we used. **Design implication: the experiment name should encode the dataset version**, or we should store it in experiment `metadata`.

### 2.3 Linking items back to production traces

Dataset items carry `source_trace_id` and optionally `source_observation_id` ([Datasets docs](https://langfuse.com/docs/evaluation/features/datasets)). That is the "I saw a bad answer in prod → turn it into a regression test case" loop, and it is first-class. For a portfolio project this is a nice story to be able to tell.

---

## 3. Experiments / runs — comparing variant A vs B

### 3.1 The data model

- **Dataset** → collection of **DatasetItems**
- **DatasetRun** (an experiment run) executes the dataset through the app
- **DatasetRunItem** is the join: it references a DatasetItem, the **Trace** created when that item ran, and optionally an Observation inside it
- **Scores** attach to traces/observations

— [Experiments data model](https://langfuse.com/docs/evaluation/experiments/data-model)

So `DatasetRunItem` is the link that makes "which test case produced this trace, and what did it score?" navigable in both directions. This is precisely the plumbing the ticket's "measurable, not vibes" requirement needs.

### 3.2 `run_experiment` — the current API (verified against source)

The v4 signature, read from the SDK rather than the docs ([`langfuse/_client/datasets.py`](https://github.com/langfuse/langfuse-python/blob/main/langfuse/_client/datasets.py)):

```python
def run_experiment(
    self,
    *,
    name: str,
    run_name: Optional[str] = None,
    description: Optional[str] = None,
    task: TaskFunction,
    evaluators: List[EvaluatorFunction] = [],
    composite_evaluator: Optional[CompositeEvaluatorFunction] = None,
    run_evaluators: List[RunEvaluatorFunction] = [],
    max_concurrency: int = 50,
    metadata: Optional[Dict[str, Any]] = None,
) -> ExperimentResult:
```

Three evaluator tiers, which is more than the docs prose suggests:

| Parameter | Runs per | Use for |
| :--- | :--- | :--- |
| `evaluators` | each item | faithfulness, relevance, correctness per question |
| `composite_evaluator` | each item, *after* item evaluators, receives their results | weighted score, pass/fail gate combining metrics |
| `run_evaluators` | the whole run | aggregate stats — mean score, % passing, recall@k over the set |

`max_concurrency` defaults to **50**. ⚠️ For us that means an experiment can fire 50 parallel LLM calls; against a rate-limited API key that will 429. **Design implication: set `max_concurrency` explicitly (5–10) in our eval scripts.**

The canonical example ([Experiments via SDK](https://langfuse.com/docs/evaluation/experiments/experiments-via-sdk)):

```python
from langfuse import get_client, Evaluation
from langfuse.openai import OpenAI

langfuse = get_client()

def my_task(*, item, **kwargs):
    question = item.input
    response = OpenAI().chat.completions.create(
        model="gpt-4.1", messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

def accuracy_evaluator(*, input, output, expected_output, metadata, **kwargs):
    if expected_output and expected_output.lower() in output.lower():
        return Evaluation(name="accuracy", value=1.0, comment="Correct answer found")
    return Evaluation(name="accuracy", value=0.0, comment="Incorrect answer")

dataset = langfuse.get_dataset("my-evaluation-dataset")

result = dataset.run_experiment(
    name="Production Model Test",
    description="Monthly evaluation of our production model",
    task=my_task,
    evaluators=[accuracy_evaluator],
)

print(result.format())
```

Note the calling convention: `task` and evaluators are **keyword-only** (`*` in the signature), and both should accept `**kwargs` for forward compatibility.

### 3.3 A vs B: the actual pattern

There is no dedicated "compare(A, B)" API. **The pattern is: loop over your variants, call `run_experiment` once per variant against the same dataset, then compare runs in the UI.** From Langfuse's own RAG guide ([`2025-10-28-rag-observability-and-evals.mdx`](https://github.com/langfuse/langfuse-docs/blob/main/content/blog/2025-10-28-rag-observability-and-evals.mdx)):

```python
dataset = langfuse.get_dataset(name="rag_bot_evals")
chunk_sizes = [128, 256, 512]

for chunk_size in chunk_sizes:
    dataset.run_experiment(
        name=f"Chunk precision: chunk_size {chunk_size} and chunk_overlap 0",
        task=create_retriever_task(chunk_size=chunk_size, chunk_overlap=0),
        evaluators=[relevant_chunks_evaluator],
    )
```

> "Each experiment runs your retriever against every question in the dataset, evaluates the results, and stores the scores in Langfuse. You can then view all experiments side by side in the UI, comparing average relevance scores to see which configuration performs best."
> — [ibid.](https://github.com/langfuse/langfuse-docs/blob/main/content/blog/2025-10-28-rag-observability-and-evals.mdx)

The SDK docstring confirms the intent: `dataset.run_experiment()` gives "Automatic dataset run creation and linking in Langfuse UI", "Built-in experiment tracking and versioning", and "**Easy comparison between different experiment runs**" ([`datasets.py`](https://github.com/langfuse/langfuse-python/blob/main/langfuse/_client/datasets.py)).

⚠️ **Honest caveat**: I could not find a docs page that specifies exactly which aggregate columns the run-comparison view displays (`/docs/evaluation/experiments/comparison` is a 404). The side-by-side comparison view exists and is screenshotted in the RAG guide; the precise column set is unverified. Treat "compare runs in the UI" as confirmed and "the UI shows exactly X, Y, Z per run" as unconfirmed.

### 3.4 Component-level vs end-to-end evaluation

Worth calling out, because it directly serves this project's "retrieval *and* answer quality must be measurable" requirement. Langfuse's own guidance:

> "**The key part is that you can evaluate components of your RAG pipeline independently from the full application**. This lets you rapidly test different chunking strategies without running expensive LLM calls every time. By isolating the retrieval component, you can iterate faster and make data-driven decisions about your document processing pipeline."
> — [RAG observability and evals guide](https://github.com/langfuse/langfuse-docs/blob/main/content/blog/2025-10-28-rag-observability-and-evals.mdx)

The `task` function is arbitrary Python, so a "retrieval-only" task that returns `{"documents": [...]}` without calling a generator LLM is perfectly legal — and much cheaper per run. ✅ **This is the highest-leverage finding for our eval design**: retrieval experiments (chunk size, overlap, top-k, rerank on/off) can be run without paying for generation at all.

### 3.5 LangChain inside an experiment task

⚠️ **Gap in the docs.** The [Experiments via SDK](https://langfuse.com/docs/evaluation/experiments/experiments-via-sdk) page uses only the raw OpenAI client; it contains **no example of using LangChain inside a `task`**. Since our pipeline is LangChain, this is the exact seam we care about and it is undocumented.

The working pattern, taken from Langfuse's own RAG application code, is to trace inside the task with `@observe()` + the callback handler, letting the experiment runner's trace be the parent:

```python
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler

langfuse = get_client()
langfuse_handler = CallbackHandler()

@observe()  # creates a trace for each question
def rag_bot(question: str) -> RagBotResponse:
    retriever = get_retriever(urls, chunk_size=256, chunk_overlap=0)

    # Retrieval is wrapped MANUALLY here, as a retriever observation
    with langfuse.start_as_current_observation(
        as_type="retriever",
        name="retrieve_documents",
        input=question,
    ) as span:
        docs = retriever.invoke(question)
        span.update(output=docs)

    # Generation goes through the LangChain callback handler
    ai_msg = bot.invoke(
        [
            {"role": "system", "content": instructions},
            {"role": "user", "content": question},
        ],
        config={"callbacks": [langfuse_handler]},
    )

    return {"answer": ai_msg.content, "documents": docs}
```
— [RAG observability and evals guide](https://github.com/langfuse/langfuse-docs/blob/main/content/blog/2025-10-28-rag-observability-and-evals.mdx)

Two things to notice, both relevant to us:

1. **They wrap retrieval by hand** with `start_as_current_observation(as_type="retriever", ...)` even though the callback handler *can* capture retrievers automatically (§1.4). Reason: they call `retriever.invoke()` outside the callback-instrumented chain, so no callback fires. **Rule of thumb for our code: anything invoked outside the `config={"callbacks": [...]}` chain must be wrapped manually.**
2. **The task returns both the answer and the documents** (`{"answer": ..., "documents": docs}`). ✅ **This is a load-bearing design decision for us**: RAG metrics need the retrieved contexts, and evaluators receive only the task's return value — not the trace. **If the task does not return the retrieved contexts, faithfulness and context-precision evaluators cannot be computed.** This must be baked into the eval-design ticket.

---

## 4. Scores — the four mechanisms

Langfuse's scoring surface has four distinct entry points.

### 4.1 Custom code-based scores (SDK)

```python
from langfuse import get_client
langfuse = get_client()

langfuse.create_score(
    name="correctness",
    value=0.9,
    trace_id="trace_id_here",
    observation_id="observation_id_here",  # optional
    data_type="NUMERIC",
    comment="Factually correct",
)
```

Or inside a span context:

```python
with langfuse.start_as_current_observation(as_type="span", name="my-operation") as span:
    span.score(name="correctness", value=0.9, data_type="NUMERIC")
    span.score_trace(name="overall_quality", value=0.95, data_type="NUMERIC")

# or via context helpers
with langfuse.start_as_current_observation(as_type="span", name="my-operation"):
    langfuse.score_current_span(name="correctness", value=0.9, data_type="NUMERIC")
    langfuse.score_current_trace(name="overall_quality", value=0.95, data_type="NUMERIC")
```
— [Custom scores](https://langfuse.com/docs/evaluation/evaluation-methods/custom-scores)

**Four data types**: `NUMERIC` (float), `CATEGORICAL` (string), `BOOLEAN` (0/1), `TEXT` (1–500 chars) ([ibid.](https://langfuse.com/docs/evaluation/evaluation-methods/custom-scores)).

✅ This is the unconstrained escape hatch. Anything we can compute in Python — exact match, recall@k over known-relevant chunk ids, a French-language regex for policy references, real Ragas — can become a score.

### 4.2 Managed LLM-as-a-judge evaluators (built-in)

Langfuse ships a catalog of prebuilt judge prompts, "built and maintained by us and partners like **Ragas**" ([LLM-as-a-Judge](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge)).

The docs do not list them. **I read the authoritative catalog out of the server source**: [`worker/src/constants/managed-evaluators.json`](https://github.com/langfuse/langfuse/blob/main/worker/src/constants/managed-evaluators.json) — **23 evaluators**:

| Evaluator | Maintainer | RAG-relevant |
| :--- | :--- | :--- |
| Hallucination | Langfuse | ✅ |
| Helpfulness | Langfuse | |
| Relevance | Langfuse | ✅ |
| Toxicity | Langfuse | |
| Correctness | Langfuse | ✅ |
| Contextrelevance | Langfuse | ✅ |
| Contextcorrectness | Langfuse | ✅ |
| Conciseness | Langfuse | |
| User Distress | Langfuse | |
| User Disagreement | Langfuse | |
| Out-of-Scope Request | Langfuse | ✅ (useful for insurance scope-guarding) |
| **Answer Correctness** | Ragas | ✅ |
| **Answer Relevance** | Ragas | ✅ |
| Answer Critic | Ragas | |
| **Context Precision** | Ragas | ✅ |
| **Context Recall** | Ragas | ✅ |
| **Faithfulness** (v1 and v2) | Ragas | ✅ |
| Goal Accuracy | Ragas | |
| Simple Criteria | Ragas | |
| SQL Semantic Equivalence | Ragas | |
| Topic Adherence Classification | Ragas | |
| Topic Adherence Refusal | Ragas | |

**So: yes, faithfulness / answer relevance / context precision / context recall all ship out of the box.** That answers the ticket's question directly — but with a serious caveat in §5 that changes the recommendation.

Requirements: an **LLM Connection must be configured**, and "the chosen default model supports structured output" ([LLM-as-a-Judge](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge)). We supply our own model API key. ⚠️ **The judge model costs money on our own LLM bill, separately from Langfuse units.**

### 4.3 Code evaluators (server-side)

> Code evaluators let you "run custom Python or TypeScript logic in Langfuse and return one or more scores."
> — [Code evaluators](https://langfuse.com/docs/evaluation/evaluation-methods/code-evaluators)

They run **server-side**, asynchronously, on observations or experiment data. Good for deterministic checks: exact match, regex, JSON/schema validation, business rules.

```python
def evaluate(ctx: EvaluationContext) -> EvaluationResult:
    expected_output = ctx.experiment.item_expected_output if ctx.experiment else None
    matches = expected_output is not None and ctx.observation.output == expected_output

    return EvaluationResult(
        scores=[
            Score(
                name="Exact match",
                value=matches,
                data_type="BOOLEAN",
                comment="Output matches expected output." if matches else "Mismatch.",
            )
        ]
    )
```

⚠️ **Self-hosting caveat**: code evaluators require "a configured code evaluator dispatcher" and are **disabled without one** ([ibid.](https://langfuse.com/docs/evaluation/evaluation-methods/code-evaluators)). This is a point *against* self-hosting for us — it is extra infrastructure to get a feature Cloud gives us for free. Note this is distinct from §4.1 client-side scores, which always work.

### 4.4 Human annotation

Manual annotation via UI and **annotation queues**, plus user feedback and free-text notes ([Evaluation overview](https://langfuse.com/docs/evaluation/overview)). ⚠️ Free tier: **1 annotation queue** ([pricing](https://langfuse.com/pricing)). For a solo project, one queue is enough.

For French insurance specifically, human annotation matters more than usual: an LLM judge scoring French insurance answers is itself a source of error, and a small hand-annotated slice is the only way to know whether the judge is trustworthy. **Recommend calibrating any judge against ~20 hand-annotated items before believing its numbers.**

---

## 5. Ragas — built-in, but ⚠️ the built-in version is not the real metric

This is the most important finding in the document and it is easy to miss.

### 5.1 Two different things share the name "Ragas"

**(a) Managed Ragas evaluators inside Langfuse** — the catalog rows in §4.2. These are **single-LLM-call prompt templates**, not the Ragas algorithm.

**(b) The actual Ragas Python library** — used as a code evaluator, running the real multi-step algorithm.

### 5.2 The managed versions are simplified — evidence

Reading the prompts straight out of [`managed-evaluators.json`](https://github.com/langfuse/langfuse/blob/main/worker/src/constants/managed-evaluators.json):

**Context Precision (v1, 2025-05-20)** — the entire prompt:

> "Given question, answer and context verify if the context was useful in arriving at the given answer. Question: `{{question}}` Answer: `{{answer}}` Context: `{{context}}`"

with output definition: `"Give verdict as '1' if useful and '0' if not"`.

Real Ragas context precision is **rank-aware**: it computes average precision@k over the *ordered list of retrieved chunks*. The Langfuse managed version collapses the whole context into one blob and emits a single 0/1 verdict. **It cannot measure ranking quality at all.** For a RAG project where the interesting question is "are the right chunks ranked highest", this metric as shipped does not answer it.

**Context Recall (v1, 2025-05-20)**:

> "Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. Context: `{{context}}` Answer: `{{answer}}`"

⚠️ Note this compares the **answer** to the context — it does not use the ground-truth reference at all. Real Ragas context recall measures how much of the *ground-truth answer* is covered by the retrieved context. As written, this template is closer to faithfulness than to recall. **The name is misleading.**

This is a known, acknowledged problem. In [langfuse discussion #9687, "Multi-step metrics from Ragas"](https://github.com/orgs/langfuse/discussions/9687), a user documents that faithfulness only executed the first step of the Ragas procedure (decompose into statements) while skipping verification and the ratio calculation. A Langfuse maintainer replied only *"Thanks for sharing, I will take a look at this."* Another commenter asked bluntly: *"The Ragas-maintained evaluators in Langfuse do essentially not work?"*

### 5.3 Faithfulness has since been fixed — the others have not

Credit where due, and this is fresh information. There is a **Faithfulness v2 template dated 2026-04-17** which implements the full procedure in a single prompt:

> "1. Deconstruction: Break the "Answer" down into a list of atomic, self-contained statements. Do not use pronouns; replace them with the actual subjects.
> 2. Verification: For each statement, check if it is supported by the "Context."
> 3. Verdict: Assign a 1 if the statement is directly supported by the context, or a 0 if it is not supported or contradicted. Provide a brief reason for each.
> 4. **Calculation: Calculate the final faithfulness score as: Total Verdicts of 1 divided by Total Number of Statements.**"

— [`managed-evaluators.json`, Faithfulness v2](https://github.com/langfuse/langfuse/blob/main/worker/src/constants/managed-evaluators.json)

**Faithfulness is now trustworthy.** But checking the `version` and `updated_at` of every template in the catalog: **Context Precision, Context Recall, and Answer Relevance are all still v1, dated 2025-05-20, and have not been revised.** The fix was applied to faithfulness only.

⚠️ **Direct constraint on our eval design:**
- ✅ Managed **Faithfulness (v2)** — usable, matches the real definition.
- ❌ Managed **Context Precision** — do not use for retrieval quality; it is a single 0/1 verdict with no ranking awareness.
- ❌ Managed **Context Recall** — do not use; the prompt does not reference ground truth and does not measure what the name implies.
- ⚠️ Managed **Answer Relevance** — v1, uses the question-generation heuristic; probably acceptable, but verify against hand annotation.

### 5.4 The real Ragas library, as a code evaluator

For the metrics we cannot trust as managed evaluators, the integration path is **bring-your-own-code**:

> "Ragas metrics plug directly into Langfuse experiments as evaluator functions."
> — [Ragas integration](https://langfuse.com/integrations/frameworks/ragas)

```python
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

metrics = [Faithfulness(), ResponseRelevancy()]
llm = LangchainLLMWrapper(ChatOpenAI())
embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

for metric in metrics:
    if isinstance(metric, MetricWithLLM):
        metric.llm = llm
    if isinstance(metric, MetricWithEmbeddings):
        metric.embeddings = embeddings
```

Wrap each as a Langfuse evaluator:

```python
def make_ragas_evaluator(metric):
    async def evaluator(*, input, output, **kwargs):
        sample = SingleTurnSample(
            user_input=input["question"],
            retrieved_contexts=output["contexts"],   # ← task must return contexts
            response=output["answer"],
        )
        score = await metric.single_turn_ascore(sample)
        return Evaluation(name=metric.name, value=float(score))
    return evaluator

ragas_evaluators = [make_ragas_evaluator(metric) for metric in metrics]

result = dataset.run_experiment(
    name="Ragas baseline",
    task=rag_task,
    evaluators=ragas_evaluators,
)
```
— [Ragas integration](https://langfuse.com/integrations/frameworks/ragas)

✅ **Ragas wraps cleanly**, uses LangChain wrappers (so it reuses our existing LLM/embedding objects), and evaluators can be `async`. Note again `retrieved_contexts=output["contexts"]` — confirming §3.5: **the task must return retrieved contexts.**

For scoring live production traffic (as opposed to experiments), the documented flow is manual three-step: fetch traces via SDK → evaluate batches with Ragas → write scores back with `langfuse.create_score()` ([ibid.](https://langfuse.com/integrations/frameworks/ragas)).

⚠️ **Ragas is a separate dependency with its own version churn**, and it needs an embeddings model in addition to a judge LLM. That is a second LLM bill and a second thing that can break.

---

## 6. Cloud vs self-host

### 6.1 Langfuse Cloud free tier ("Hobby")

From [langfuse.com/pricing](https://langfuse.com/pricing):

| | Hobby (free) | Core $29/mo | Pro $199/mo |
| :--- | :--- | :--- | :--- |
| Units included | **50k / month** | 100k, then $8/100k | 100k, then $8/100k |
| **Data access (retention)** | **30 days** | 90 days | 3 years |
| Users | 2 | unlimited | unlimited |
| Monitors | 2 | 20 | 50 |
| Annotation queues | 1 | 3 | unlimited |
| General API rate limit | 30 req/min | | |
| Metrics API v2 | 100 req/day | | |
| Ingestion throughput | 1,000 req/min | | |
| LLM-as-a-judge | available | | |

Enterprise is $2,499/mo — irrelevant here.

### 6.2 ⚠️ What a "unit" is — the constraint that actually bites

> "Units = Count of Traces + Count of Observations + Count of Scores"
>
> "Any trace, observation, or score stored in Langfuse counts as a billable unit, **whether it is sent by your application or created by Langfuse features such as LLM-as-a-Judge, Annotation Queues, or experiments**."
> — [Billable units](https://langfuse.com/docs/administration/billable-units)

This is the crucial detail: **observations and evaluator-generated scores each count.** A "trace" is not one unit — a trace with 8 steps and 4 scores is 13 units.

**Estimate for this project** (my arithmetic, from the unit formula — not a vendor figure):

A traced LangChain RAG call ≈ 1 trace + ~6–8 observations (chain, retriever, prompt, generation, sub-chains) ≈ **~8 units**. Add 3–4 evaluator scores ≈ **~12 units per dataset item per experiment run**.

| Scenario | Units | Fits in 50k/mo? |
| :--- | :--- | :--- |
| One experiment, 50-item golden set, 4 metrics | ~600 | ✅ ~80 runs/month |
| One experiment, 200-item golden set, 4 metrics | ~2,400 | ✅ ~20 runs/month |
| Retrieval-only experiment (no generation), 50 items | ~250 | ✅ very cheap |
| Interactive dev tracing, ~100 queries/day | ~24,000/mo | ⚠️ half the budget |

**Verdict: 50k units/month is comfortable for a solo portfolio project.** The golden-set experiments are cheap; the thing that would actually eat the quota is leaving tracing on during heavy interactive development. Mitigation if needed: sample traces in dev, or run retrieval-only experiments while iterating on chunking.

### 6.3 ⚠️ The 30-day retention limit is the real constraint

This is a sharper problem than the unit cap for a project that runs over months.

**On the free tier, experiment results older than 30 days are gone.** For a portfolio project developed over a semester, that means:

- You cannot compare today's run against your baseline from three months ago.
- The "look how the metrics improved over the project" narrative — genuinely valuable in a portfolio — is not reconstructible from Langfuse alone after 30 days.

**Mitigation (recommended regardless of hosting choice): persist experiment results outside Langfuse.** `run_experiment` returns an `ExperimentResult`; dump the per-run aggregate scores to a small CSV/JSON committed to the repo. Cheap insurance, and it makes the metric history part of the repo's own story rather than a vendor's retention policy. This should be an explicit item in the eval-design ticket.

### 6.4 Self-hosting

**License**: the Langfuse repo is source-available with an MIT-ish core plus an EE carve-out — "All content that resides under the `ee/`, `web/src/ee/`, and/or `worker/src/ee/` directories … is licensed under the license defined in `ee/LICENSE`" ([LICENSE](https://github.com/langfuse/langfuse/blob/main/LICENSE), Copyright 2023-2026 Langfuse GmbH). The docs state "All core Langfuse features and APIs are available in Langfuse OSS (MIT licensed) without any limits" ([license key docs](https://langfuse.com/self-hosting/license-key)).

**EE-only** (i.e. *not* free when self-hosting): Project-level RBAC roles, Protected Prompt Labels, Data Retention Policies, Audit Logs, Server-Side Data Masking, UI Customization, Organization Creators, Org Management API and SCIM, Instance Management API ([ibid.](https://langfuse.com/self-hosting/license-key)). ✅ **None of these matter for a solo project** — datasets, experiments, scores, and LLM-as-a-judge are all in the free core.

**Required infrastructure** ([self-hosting docs](https://langfuse.com/self-hosting)):

1. **Postgres** — "main database for transactional workloads"
2. **ClickHouse** — "High-performance OLAP database which stores traces, observations, and scores"
3. **Redis/Valkey** — "Used for queue and cache operations"
4. **S3 / blob storage** — "persist all incoming events, multi-modal inputs, and large exports"
5. **langfuse-web** + **langfuse-worker** containers

> "All infrastructure components (ClickHouse and Postgres) must run with their timezone set to UTC."

Confirmed from the actual [`docker-compose.yml`](https://github.com/langfuse/langfuse/blob/main/docker-compose.yml) — **six containers**:

```
langfuse-worker   docker.io/langfuse/langfuse-worker:4
langfuse-web      docker.io/langfuse/langfuse:4
clickhouse        docker.io/clickhouse/clickhouse-server:25.12
minio             cgr.dev/chainguard/minio
redis             docker.io/redis:7
postgres          docker.io/postgres:17
```

(Note the images are already pinned to tag `:4` — the compose file tracks server v4.)

**Commands** ([docker compose docs](https://langfuse.com/self-hosting/docker-compose)):

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up
# upgrade:
docker compose up --pull always
```

**Resources**: the guide recommends "at least 4 cores and 16 GiB of memory, e.g. a t3.xlarge on AWS" plus ~100GB storage ([ibid.](https://langfuse.com/self-hosting/docker-compose)). Docker Compose "lacks high-availability, scaling capabilities, and backup functionality"; Kubernetes is recommended for production.

⚠️ **16 GiB of RAM for six containers, to evaluate a solo RAG project, is a bad trade** — ClickHouse alone is a heavyweight. Plus code evaluators (§4.3) need an extra dispatcher to work at all.

---

## 7. Constraints on eval design — consolidated

Everything below is something a later eval-design ticket must account for.

| # | Constraint | Impact | Source |
| :--- | :--- | :--- | :--- |
| C1 | **Managed Context Precision / Context Recall do not implement the real metrics.** Context Precision is a single 0/1 verdict with no rank awareness; Context Recall never sees ground truth. | **Retrieval quality must be hand-rolled** — real Ragas as a code evaluator, or our own recall@k against known-relevant chunk ids. | §5.2, [managed-evaluators.json](https://github.com/langfuse/langfuse/blob/main/worker/src/constants/managed-evaluators.json), [disc. #9687](https://github.com/orgs/langfuse/discussions/9687) |
| C2 | **The task function must return the retrieved contexts.** Evaluators see only the task's return value, not the trace. | Fix the task's return shape (`{"answer": ..., "contexts": [...]}`) before writing any evaluator. | §3.5, §5.4 |
| C3 | **Free-tier retention is 30 days.** | Long-run metric history is lost. Persist `ExperimentResult` aggregates to the repo. | §6.3, [pricing](https://langfuse.com/pricing) |
| C4 | **Units = traces + observations + scores**, including evaluator-generated scores. | ~12 units per item per run. 50k/mo is fine for experiments; heavy dev tracing is the risk. | §6.2, [billable units](https://langfuse.com/docs/administration/billable-units) |
| C5 | **Reranking is not auto-traced** unless it is a LangChain component. | Hand-wrap the reranker in a span if rerank lift is to be measured. | §1.4 |
| C6 | **`max_concurrency` defaults to 50.** | Will rate-limit our LLM provider. Set it to 5–10 explicitly. | §3.2, [datasets.py](https://github.com/langfuse/langfuse-python/blob/main/langfuse/_client/datasets.py) |
| C7 | **`DatasetItemClient.run()` was removed in SDK v4.** | Most older RAG-eval tutorials are dead code. Use `dataset.run_experiment()`. | §1.3 |
| C8 | **Metadata values are capped at 200 chars.** | Retrieved French policy chunks cannot go in trace metadata. | §1.3 |
| C9 | **Judge LLM cost is on our own bill**, separate from Langfuse units; Ragas additionally needs an embeddings model. | 4 metrics × 200 items × N variants is a real LLM spend. Keep the golden set small and deliberate. | §4.2, §5.4 |
| C10 | **Code evaluators need a dispatcher when self-hosted.** | Argues against self-hosting. | §4.3 |
| C11 | **Docs lag the v4 release** (LangChain page still labelled v3; `/experiments/comparison` 404s). | Verify against SDK source when something looks off. | §0 |
| C12 | **LLM judges on French text are unvalidated.** Every managed template's prompt is English. | Calibrate against ~20 hand-annotated French items before trusting judge scores. | §4.4 |

C12 deserves emphasis: every managed evaluator prompt in the catalog is written in English, and our corpus and answers are French. Whether an English-prompted judge reliably scores French insurance answers is an **open empirical question**, not something the docs answer. That is a genuine risk to "measurable, not vibes" — an unvalidated judge is just vibes with a number attached.

---

## SUMMARY

### Built-in — we get this for free

- **Tracing via LangChain callback handler.** One-line wiring (`config={"callbacks": [CallbackHandler()]}`). Captures chains, LLM generations (with tokens + cost), tools, and **retrieval as a first-class `retriever` observation with the retrieved documents as output** — verified in SDK source, not just docs. This alone separates "retrieval failed" from "generation failed", which is the main debugging question in RAG.
- **Datasets with automatic timestamp versioning.** Every add/update/delete produces a new version; historical versions are retrievable. `expected_output` is optional, so the golden set can grow incrementally. Items can be promoted from real traces via `source_trace_id`.
- **Experiments** via `dataset.run_experiment()`, with three evaluator tiers (per-item, composite, whole-run). A vs B = loop over variants against one dataset, compare runs in the UI.
- **Component-level evaluation**: the `task` is arbitrary Python, so retrieval-only experiments (chunking, top-k, rerank) run without paying for generation. Highest-leverage capability for this project.
- **Scores**: 4 data types, settable from SDK, server-side code evaluators, manual annotation queues, and 23 managed LLM-as-a-judge evaluators.
- **Managed Faithfulness (v2, 2026-04-17)** — genuinely implements decomposition → verification → ratio. Trustworthy.

### We have to build it

- **Retrieval quality metrics.** ⚠️ The headline finding: managed **Context Precision** and **Context Recall** are single-LLM-call approximations that do not implement the metrics they are named after (Context Precision has no rank awareness; Context Recall never reads the ground truth). Both are still v1 from 2025-05-20 and were not fixed when Faithfulness was. **Wire in the real Ragas library as code evaluators, and/or hand-roll recall@k against known-relevant chunk ids.**
- **Task return contract** — must return retrieved contexts alongside the answer, or no RAG metric can be computed.
- **Rerank tracing** — hand-wrapped span if reranking is not a LangChain component.
- **Metric history persistence** — dump per-run aggregates to the repo; the free tier forgets after 30 days.
- **Judge calibration on French** — all managed prompts are English; validate against a hand-annotated slice before trusting the numbers.

### Recommendation: **Langfuse Cloud, free Hobby tier**

For a solo learning/portfolio project this is clear-cut.

- The free tier's 50k units/month comfortably covers ~80 runs/month of a 50-item golden set with 4 metrics; retrieval-only experiments are cheaper still. The unit cap is not the binding constraint.
- Everything we need — datasets, experiments, scores, LLM-as-a-judge, one annotation queue — is available on the free tier. The EE-gated features (RBAC, audit logs, SCIM, data masking) are all irrelevant to a solo project.
- Self-hosting costs **six containers and a recommended 16 GiB / 4 cores** to run ClickHouse + Postgres + Redis + MinIO + web + worker. That is real infrastructure work that teaches us nothing about RAG evaluation, and it *removes* a feature (code evaluators need a separate dispatcher). Wrong trade for this project.
- Cloud also always runs the latest server, so we sidestep the v3/v4 compatibility matrix entirely — worth something given v4 went GA three days ago.

**The one thing to handle regardless**: the 30-day retention window. Persist experiment aggregates into the repo from day one, so the project's metric history survives independently of the free tier — which is also the better portfolio artifact.

**Version to build against**: Langfuse Cloud (server v4) + Python SDK v4 (`langfuse>=4.14`, Python ≥3.10). Treat the v3-labelled LangChain docs page as stale-but-still-correct on imports; verify anything surprising against the SDK source.
