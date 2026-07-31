# Which embedding model for French insurance text?

Research note for [issue #4](https://github.com/Zameloth/rag_assurances/issues/4).
Date: 2026-08-01. All figures traced to primary sources (official model cards, `config.json`
on the Hugging Face Hub, the MTEB raw results repository, and the models' own papers).

**Constraint taken as given:** embeddings run locally and free, so re-embedding the whole
corpus for an A/B is cheap. A paid API is reserved for generation. Mistral's embedding API is
included only as a paid comparison baseline.

**Corpus:** French insurance / legal / administrative text — Code des assurances articles,
conditions générales, consumer guidance.

---

## TL;DR

1. The only public French retrieval benchmark that really tests *legal* French is **BSARD**
   (Belgian statutory articles). Every other French retrieval task in MTEB(fra) is
   general-purpose — the MTEB-French authors say so explicitly about Syntec, the one that
   *looks* legal. Ranking models on "French average" is therefore close to useless here.
2. On BSARD, the French-specialist models are the **worst**, not the best.
   `Solon-embeddings-large-0.1` scores nDCG@10 **2.08** and recall@100 **12.61**, against
   **24.61 / 66.22** for `multilingual-e5-large-instruct` — while *beating* e5 on
   general-French tasks. French-specific training is not the axis that matters for this corpus.
3. **BGE-M3** has no MTEB(fra) retrieval numbers at all, but its own paper reports French
   long-document retrieval (MLDR-fr) where **sparse (82.7) beats dense (73.8)** and the
   hybrid reaches **84.2**. For long, terminology-dense legal text, the lexical leg is doing
   most of the work — which is exactly the hybrid-retrieval question downstream.
4. Recommendation: **BGE-M3 as the default**, **multilingual-e5-large-instruct as the runner-up
   to A/B**. Full rationale at the end.

---

## 1. How to read the French numbers

### What MTEB(fra) actually contains

The `MTEB(fra, v1)` benchmark is defined in the MTEB source
([`mteb/benchmarks/benchmarks/benchmarks.py`](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/benchmarks/benchmarks/benchmarks.py)).
Its **Retrieval** category is exactly five tasks:

`AlloprofRetrieval`, `BSARDRetrieval`, `MintakaRetrieval`, `SyntecRetrieval`, `XPQARetrieval`.

The other categories (Classification, Clustering, Pair Classification, Reranking, STS) are *not*
retrieval. A quoted "MTEB-FR score" is usually the **average across all 8 task types**, which for
a RAG project is the wrong number — a model can win the average on classification and clustering
while being mediocre at retrieval. Everything below is retrieval-specific unless stated.

Benchmark citation: Ciancone et al., *MTEB-French: Resources for French Sentence Embedding
Evaluation and Analysis*, [arXiv:2405.20468](https://arxiv.org/abs/2405.20468).

### Which of those tasks is actually legal/administrative

| Task | Domain | Notes (source: [arXiv:2405.20468](https://arxiv.org/abs/2405.20468)) |
|---|---|---|
| **BSARDRetrieval** | **Belgian statutory law, French** | 22,600+ statutory articles, questions written/labelled by experienced jurists. The genuine legal-retrieval test. Original dataset: Louis & Spanakis, ACL 2022, [arXiv:2108.11792](https://arxiv.org/abs/2108.11792) |
| SyntecRetrieval | French *convention collective* (Syntec) | Looks administrative, but the MTEB-French authors state that "the language used does not feature the specificity of the legal vocabulary" — they treat it as a **general-purpose** dataset. ~90 documents, 100 queries. Tiny and near-saturated (top models >87). |
| AlloprofRetrieval | Québec school-homework help | General-purpose |
| MintakaRetrieval | Open-domain trivia QA | General-purpose |
| XPQARetrieval (fra-fra) | Product/e-commerce QA | General-purpose |

**This is the central finding of the note:** of the five, only BSARD probes legal French, and
it is by far the hardest (top open model ~25 nDCG@10 vs ~85 on Syntec). Any model choice made on
the MTEB-FR average is being made on general French.

`FQuADRetrieval` (French Wikipedia QA) appears in the results repo but is **not** part of
MTEB(fra) v1; it is included below only as extra signal.

### Scoring conventions

- Scores are **nDCG@10 ×100** unless labelled otherwise.
- For BSARD, MTEB's `main_score` is **recall@100**, not nDCG@10 — because nDCG@10 is near the
  floor for every model. Both are reported below so the numbers are not accidentally compared
  across metrics.
- Source for every cell: the raw JSON in
  [`embeddings-benchmark/results`](https://github.com/embeddings-benchmark/results), at
  `results/<model>/<revision>/<Task>.json` (test split). These are the files the public
  leaderboard is built from.

---

## 2. French retrieval results (primary data)

Read down the **BSARD** columns for legal French, and the others for general French.

| Model | BSARD nDCG@10 | BSARD R@100 | Syntec | Alloprof | Mintaka-fr | XPQA fra-fra | FQuAD |
|---|---|---|---|---|---|---|---|
| **bge-multilingual-gemma2** (9B) | **28.52** | 65.77 | **90.37** | 58.50 | 62.53 | 77.42 | – |
| jina-embeddings-v3 | 24.76 | 64.41 | 83.85 | 54.38 | 26.91 | **77.58** | – |
| **multilingual-e5-large-instruct** | 24.61 | **66.22** | 87.80 | 52.12 | 33.49 | 72.72 | 83.72 |
| Cohere-embed-multilingual-v3.0 (API) | 22.91 | 65.31 | 88.59 | 51.51 | 34.56 | 69.72 | – |
| multilingual-e5-large | 21.28 | 63.06 | 82.38 | 39.34 | – | 61.38 | 82.51 |
| Lajavaness/bilingual-embedding-large | 19.58 | 64.41 | 84.20 | 47.60 | 32.62 | 66.47 | – |
| Solon-embeddings-mini-beta-1.1 | 19.46 | 59.46 | 78.70 | 45.27 | 19.25 | 65.66 | – |
| gte-multilingual-base | 19.06 | 62.61 | 83.04 | 53.64 | 34.71 | 67.32 | – |
| multilingual-e5-base | 18.82 | 55.41 | 82.86 | 34.45 | 30.96 | 59.56 | 82.48 |
| multilingual-e5-small | 14.54 | 52.70 | 73.46 | 27.38 | 25.00 | 57.17 | 78.78 |
| paraphrase-multilingual-mpnet-base-v2 | 13.19 | 42.79 | 76.00 | 30.80 | 24.45 | 46.22 | 60.08 |
| **Solon-embeddings-large-0.1** | **2.08** | **12.61** | 84.60 | 46.94 | 30.07 | 70.22 | – |
| sentence_croissant_alpha_v0.4 | 0.16 | 9.91 | 76.96 | 49.44 | 31.06 | 62.81 | 69.24 |
| openai/text-embedding-3-large (API) | – | – | 87.36 | 60.27 | 62.88 | 76.53 | – |

Not in the table because **no MTEB(fra) retrieval results exist** for them in the results repo
(verified by listing every result file in their directories): **BGE-M3**, **Qwen3-Embedding**
(0.6B / 4B / 8B), **EmbeddingGemma-300m**, **granite-embedding r2**, **mistral-embed**.
See §4 and §6.

French *reranking* results, which do exist for some of those (nDCG-style `main_score` ×100,
`AlloprofReranking`, test split):

| Model | AlloprofReranking | SyntecReranking |
|---|---|---|
| Qwen3-Embedding-0.6B | 80.38 | – |
| multilingual-e5-large-instruct | 74.68 | 89.95 |
| **BGE-M3** | 73.87 | – |

This is the only apples-to-apples French retrieval-flavoured number available for BGE-M3 in the
official results, and it sits essentially level with e5-large-instruct.

### The Solon anomaly — read before dismissing Solon

`Solon-embeddings-large-0.1` is the French-specialist model, and on BSARD it collapses:
nDCG@10 2.08, recall@100 12.61
([raw JSON](https://github.com/embeddings-benchmark/results/blob/main/results/OrdalieTech__Solon-embeddings-large-0.1/external/BSARDRetrieval.json)).
The same pattern hits the other French-specific model, `sentence_croissant_alpha_v0.4` (0.16 / 9.91).
Meanwhile Solon is *fine* on Syntec (84.60) and XPQA fra-fra (70.22).

Honest caveat: I could not determine from primary sources whether this is a true capability gap
or an evaluation artefact. Solon's model card instructs "Add `query : ` before the query to
retrieve to increase performance"
([model card](https://huggingface.co/OrdalieTech/Solon-embeddings-large-0.1)), and if the harness
omitted that prefix, asymmetric retrieval degrades. Against that theory: the same run scored well
on Syntec and XPQA, which are also asymmetric retrieval; and BSARD's much larger corpus (22.6k
articles vs Syntec's ~90) punishes a weak model far harder. The result is also submitted as an
`external` result with no recorded `mteb_version`, so it is less traceable than the e5 rows.

**Practical stance:** do not adopt Solon as the default on the strength of "it's the French one".
If you want to keep it in the running, re-run it yourself *with the `query : ` prefix* on your
own eval set — that is a 20-minute check given the local-and-free constraint.

---

## 3. Specifications and local-hardware profile

All values read from the model's own `config.json` / `sentence_bert_config.json` on the Hub and
from the Hub API's `safetensors.total` parameter count (i.e. primary, not a write-up).

| Model | Params | Dims | Max tokens | Licence | Sparse / hybrid | CPU-viable? |
|---|---|---|---|---|---|---|
| **BAAI/bge-m3** | ~568M (XLM-R-large backbone; weights 2.27 GB fp32) | 1024 | **8192** | **MIT** | **Yes — dense + sparse (lexical weights) + ColBERT multi-vector** | Yes |
| **intfloat/multilingual-e5-large-instruct** | 559,890,432 | 1024 | 512 | **MIT** | No | Yes |
| intfloat/multilingual-e5-large | 559,890,946 | 1024 | 512 | MIT | No | Yes |
| intfloat/multilingual-e5-base | 278,044,162 | 768 | 512 | MIT | No | Yes, comfortably |
| intfloat/multilingual-e5-small | 117,654,272 | 384 | 512 | MIT | No | Yes, trivially |
| OrdalieTech/Solon-embeddings-large-0.1 | 559,890,432 | 1024 | 512 | MIT | No | Yes |
| OrdalieTech/Solon-embeddings-base-0.1 | 278,043,648 | 768 | 512 | MIT | No | Yes |
| dangvantuan/sentence-camembert-large | 336,662,018 | 1024 | **128** | Apache-2.0 | No | Yes |
| dangvantuan/sentence-camembert-base | 110,622,466 | 768 | **128** | Apache-2.0 | No | Yes |
| Lajavaness/bilingual-embedding-large | 559,890,432 | 1024 | 512 | Apache-2.0 | No | Yes |
| jinaai/jina-embeddings-v3 | 572,310,396 | 1024 (Matryoshka) | 8194 | **CC-BY-NC-4.0** | No | Yes |
| Alibaba-NLP/gte-multilingual-base | 305,369,089 | 768 | 8192 | Apache-2.0 | No | Yes, comfortably |
| Qwen/Qwen3-Embedding-0.6B | 595,776,512 | ≤1024 (MRL 32–1024) | 32768 | Apache-2.0 | No | Yes |
| Qwen/Qwen3-Embedding-4B | 4,021,774,336 | 2560 | 40960 | Apache-2.0 | No | Painful on CPU |
| Qwen/Qwen3-Embedding-8B | 7,567,295,488 | 4096 | 40960 | Apache-2.0 | No | No (GPU) |
| BAAI/bge-multilingual-gemma2 | ~9B | – | – | **Gemma licence** | No | No (GPU) |
| google/embeddinggemma-300m | 302,863,104 | 768 (MRL 512/256/128) | 2048 | **Gemma licence** | No | Yes, designed for on-device |
| mistral-embed (API) | – | 1024 | 8k | Commercial (Premier) | No | n/a — $0.10 / M tokens |

Notes on the hardware column:

- These are **derived from parameter counts, not measured**. I did not benchmark throughput —
  `sentence-transformers` is not installed in this environment and installing a torch stack to
  time a model was out of scope. Treat the ordering as reliable and any absolute latency as
  unknown until you measure it.
- Rough memory: fp32 ≈ params × 4 bytes. The ~560M-parameter class (BGE-M3, e5-large*, Solon,
  jina-v3) is ~2.2 GB resident; the ~300M class (gte-multilingual-base, EmbeddingGemma) ~1.2 GB.
  The machine this was run on has 16 cores / 30 GB RAM, so RAM is not the binding constraint —
  wall-clock throughput over the corpus is.
- **ONNX runtime weights are published on the Hub** for `bge-m3`, `multilingual-e5-large` and
  `multilingual-e5-large-instruct` (an `onnx/model.onnx` file exists in each repo). They are
  **not** published for Solon, gte-multilingual-base, Qwen3-Embedding-0.6B, EmbeddingGemma-300m
  or bilingual-embedding-large. ONNX + int8 is the single biggest CPU speed lever, so this is a
  real practical differentiator.
- BGE-M3's 8192-token window is only a *cost* if you use it. Long inputs are quadratic-ish in
  attention; embedding 8k-token chunks on CPU is much slower than embedding 512-token chunks.
  You get the option, not an obligation.

### Licence flags

- **MIT** (BGE-M3, all multilingual-e5, Solon) — unencumbered.
- **CC-BY-NC-4.0** (jina-embeddings-v3) — **non-commercial only**. Fine for a learning/portfolio
  repo, but it forecloses any later commercial use, and it is worth not building a habit on it.
- **Gemma licence** (EmbeddingGemma, bge-multilingual-gemma2) — requires accepting Google's terms
  and carries use restrictions; not a blocker for this project but not OSI-open either.

---

## 4. BGE-M3 in detail (the model with no MTEB-fra numbers)

BGE-M3 is absent from MTEB(fra) retrieval — verified by listing every result file under
[`results/BAAI__bge-m3/5617a9f.../`](https://github.com/embeddings-benchmark/results/tree/main/results/BAAI__bge-m3):
it has `AlloprofReranking`, `AlloProfClusteringS2S.v2`, `MintakaRetrieval` (ja/es subsets only)
and `XPQARetrieval` (es subsets only), but no `BSARDRetrieval`, `SyntecRetrieval`,
`AlloprofRetrieval` or French Mintaka/XPQA subsets. So it cannot be placed in the table in §2.

Its own paper supplies French numbers instead — Chen et al., *M3-Embedding*,
[arXiv:2402.03216](https://arxiv.org/abs/2402.03216), nDCG@10, French column:

| Benchmark (fr) | Dense | Sparse | Multi-vector | Dense+Sparse | All |
|---|---|---|---|---|---|
| **MIRACL fr** (short passages) | 78.6 | 65.4 | 80.1 | 79.7 | **80.4** |
| **MLDR fr** (**long documents**) | 73.8 | **82.7** | 77.2 | **84.2** | 83.9 |

This table is the most on-point evidence in the whole note for an insurance/legal corpus:

- On **short** passages, dense clearly beats sparse (78.6 vs 65.4).
- On **long documents**, the relationship **inverts** — sparse (82.7) beats dense (73.8) by nearly
  9 points, and the hybrid adds a further 1.5 over sparse alone.
- Code des assurances articles and conditions générales are long and dominated by exact
  terminology (*franchise*, *garantie décennale*, *déchéance*, article references like
  *L.113-2*). That is precisely the regime where the lexical signal carries the retrieval, and
  where a pure-dense model leaves points on the table.

Multi-functionality is verifiable from the repo contents, not just the card: `colbert_linear.pt`
and `sparse_linear.pt` ship alongside `pytorch_model.bin`
([model card](https://huggingface.co/BAAI/bge-m3)). The card also documents integration with
Milvus and Vespa for hybrid retrieval, and text-embeddings-inference support.

**Currency check:** BGE-M3 was released Feb 2024 and last modified 2024-07-03. I checked BAAI's
Hub listing sorted by last-modified: everything newer is a different product line
(`bge-multilingual-gemma2` 9B dense-only, `bge-code-v1`, `bge-reasoner-embed-qwen3-8b`, the BGE-VL
vision models). **As of 2026-08-01 there is no BGE-M3 successor** — it remains the only widely
adopted model shipping dense + sparse + ColBERT from one forward pass, and it is still pulling
~35M downloads/month, so ecosystem support is not a risk.

---

## 5. Model-by-model verdicts

**BGE-M3** — MIT, 1024-dim, 8192 tokens, ~568M params, CPU-viable, ONNX available, native
dense+sparse+ColBERT. No MTEB(fra) retrieval data; French evidence comes from its own paper
(MIRACL-fr, MLDR-fr above) and from AlloprofReranking 73.87, level with e5-large-instruct.
Ageing (July 2024) but unsuperseded in its category.

**multilingual-e5-large-instruct** — MIT, 1024-dim, 512 tokens, 560M params, ONNX available.
The **strongest MIT-licensed local model on BSARD** (24.61 nDCG@10 / 66.22 R@100) and strong
across the board (Syntec 87.80, Alloprof 52.12, XPQA fra-fra 72.72, FQuAD 83.72). Requires a
one-sentence task instruction on the **query** side only; documents get no prefix
([model card](https://huggingface.co/intfloat/multilingual-e5-large-instruct)). The 512-token cap
forces chunking, which for this corpus is arguably correct anyway.

**multilingual-e5-large / base / small** — MIT, 512 tokens, 1024/768/384 dims. Clean size ladder
with a clean quality ladder on BSARD R@100: 63.06 / 55.41 / 52.70. The non-instruct `large` is
notably weaker than `-instruct` on every French retrieval task. `-small` (117M) is a legitimate
fast-iteration model while you build the pipeline. All need `query:` / `passage:` prefixes.

**Solon-embeddings (OrdalieTech)** — MIT, French-specific, XLM-R-large backbone, 1024-dim,
512 tokens. Its card reports a mean of 0.7490 over 9 French benchmarks and mMARCO-fr recall@10
55.5 / @100 82.7 — note those are the *card's own* mMARCO figures, not MTEB retrieval. Good on
general French, catastrophic on BSARD (§2). `Solon-embeddings-mini-beta-1.1` actually scores
*better* on BSARD (19.46 / 59.46) than the large model, which reinforces that something specific
is wrong with the large model's legal retrieval rather than the family being weak.
Last modified March 2024; no v0.2 or successor exists on the Hub as of today.

**Sentence-CamemBERT (dangvantuan) and CamemBERT-derived models** — **rule these out.**
`sentence-camembert-large`'s `sentence_bert_config.json` sets `max_seq_length: 128`, and the base
model likewise. 128 tokens cannot hold a single article of the Code des assurances. It was tuned
on `stsb_multi_mt` French (sentence-similarity), and the card reports STS Pearson/Spearman
(85.9 / 85.8 on test) — **STS, not retrieval**. It is a sentence-similarity model, not a RAG
retriever, and it has no MTEB(fra) retrieval entry at all. The vocabulary is 32k French-only, so
it would also mangle the English/Latin terms that appear in insurance contracts.
`Lajavaness/bilingual-embedding-large` (Apache-2.0, XLM-R-based, FR/EN, 512 tokens) is the
respectable modern descendant of the "French-tuned" idea, and it is merely mid-table
(BSARD 19.58 / 64.41).

**Mistral embeddings (paid baseline)** — `mistral-embed`, version **23.12**, 1024 dims, 8k
context, **$0.10 per million tokens**
([model card](https://docs.mistral.ai/models/model-cards/mistral-embed-23-12)). Mistral's models
overview lists only two embedding models, `mistral-embed` (23.12) and `codestral-embed-25-05`
(code-specific), so **the general-text embedding model has not been refreshed since December
2023**. It has **no entry in the MTEB results repository at all** (verified across the full
659-model listing), so there is no traceable French retrieval number for it — the MTEB-French
paper evaluated it, but I could not pull a per-task French figure from a primary artefact.
Conclusion: it is a weak paid baseline. It is neither cheaper than local (local is free) nor
demonstrably better on French legal text, and it is 2.5 years stale. If you ever want a paid
ceiling to measure against, `text-embedding-3-large` (Syntec 87.36, Alloprof 60.27, XPQA 76.53)
or Cohere multilingual-v3.0 (BSARD 22.91 / 65.31) have actual French numbers.

### Candidates you did not list, that I checked

**bge-multilingual-gemma2** — the **best French retrieval scores of anything measured**
(BSARD 28.52 / 65.77, Syntec 90.37, Alloprof 58.50, Mintaka-fr 62.53, XPQA 77.42). But it is a
**9B** Gemma-2-based model under the **Gemma licence**. Not CPU-viable; violates the local-and-free
constraint in practice. Worth knowing as the quality ceiling for open models on French.

**jina-embeddings-v3** — genuinely competitive (BSARD 24.76 / 64.41, best-in-class XPQA fra-fra
77.58, 8194-token context, task-specific LoRA prompts built into
`config_sentence_transformers.json`). Blocked on **CC-BY-NC-4.0**. If licence were not a concern
this would be a serious contender.

**Qwen3-Embedding (0.6B / 4B / 8B)** — Apache-2.0, 32k+ context, Matryoshka dims (32–1024 for
0.6B), released June 2025, and the 8B was #1 on the MTEB multilingual leaderboard at release
(70.58, per its
[model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)). **But**: the card reports
*aggregated* multilingual scores, not French, and the results repo has **no MTEB(fra) retrieval
runs** for any Qwen3-Embedding size. The one French data point that does exist —
AlloprofReranking **80.38** for the 0.6B — is the best of any model I pulled, which makes this
the most interesting unmeasured candidate. The 0.6B is CPU-viable; 4B/8B are not. No ONNX weights
published. **Flagging this as the main "might already have superseded my recommendation"
uncertainty.**

**gte-multilingual-base** (Apache-2.0, 305M, 768 dims, **8192 tokens**) — the best
size/context/licence trade-off in the table, mid-table on BSARD (19.06 / 62.61) but strong on
Alloprof (53.64) for a 305M model. A reasonable third A/B arm if CPU throughput turns out to be
the binding constraint.

**EmbeddingGemma-300m** (Sept 2025, 768 dims + Matryoshka, 2048 tokens, Gemma licence,
MTEB Multilingual v2 61.15) — built for on-device, but again **no French retrieval numbers**, and
no ONNX weights on the Hub.

---

## 6. How well do these handle legal/administrative French specifically?

Answering the question as asked, and marking where evidence runs out:

- **There is evidence, and it is BSARD.** It is the only public French retrieval benchmark built
  on real statutory text with jurist-written queries. Its BSARD column in §2 is the single most
  transferable number to a Code des assurances corpus.
- **Everything on BSARD is bad in absolute terms.** The best open local model manages nDCG@10
  ≈ 25 and recall@100 ≈ 66, where the same models hit 82–88 on general-French Syntec. The
  original BSARD paper's best *fine-tuned* dense baseline reached 74.8% R@100
  ([arXiv:2108.11792](https://arxiv.org/abs/2108.11792)) — i.e. even domain fine-tuning only gets
  you to ~75. **Expect zero-shot retrieval on French legal text to be substantially harder than
  the general French numbers suggest, and plan the retrieval stack (hybrid, reranking, chunking)
  accordingly rather than expecting the embedding model to solve it.**
- **French-specific ≠ legal-capable.** The two French-specialist models are the two worst on
  BSARD while being fine on general French. Whatever the mechanism (§2 caveat), "trained on
  French" is not evidence of legal competence.
- **Syntec is a trap.** It is the dataset that looks like the right proxy — a real *convention
  collective* — and the benchmark's own authors say its language lacks legal specificity. With
  ~90 documents it is also near-saturated. Do not use Syntec to pick your model.
- **Where evidence does not exist, I am not going to invent it:** there is **no** public French
  *insurance*-domain retrieval benchmark that I could find, and **no** French legal-domain
  embedding model with published retrieval numbers. There is also **no** primary evidence about
  legal-French performance for BGE-M3, Qwen3-Embedding, EmbeddingGemma or mistral-embed, because
  none of them has been run on BSARD. Claims about their legal-French quality — in either
  direction — would be speculation.
- The nearest domain-specialised option is `voyageai/voyage-law-2` (present in the MTEB results
  repo), but it is a paid API and English-oriented; I found no French legal evidence for it.

**Consequence for this project:** the strongest available evidence for *this* corpus is not any
single leaderboard cell — it is the MLDR-fr row in §4 showing that on long French documents,
lexical retrieval outperforms dense. That argues for choosing an embedding model that makes
hybrid retrieval cheap to reach, and for building your own ~50-query eval set over the actual
corpus early, because the public benchmarks will not settle this.

---

## 7. Gaps, caveats and currency

- **Not measured:** CPU throughput. All hardware statements are derived from parameter counts.
- **Not comparable:** BGE-M3's French numbers come from its own paper (MIRACL/MLDR), everyone
  else's from MTEB(fra). Different corpora, different query distributions. They cannot be put in
  one ranking, and I have not done so.
- **Version skew:** the MTEB rows carry different `mteb_version` values (1.12.75 → 1.38.9) and
  some are `external` submissions with no recorded version. Small differences (<2 points) between
  rows should not be treated as meaningful.
- **MTEB v2 vs v1:** MTEB(fra) v1 is what these retrieval tasks belong to. Aggregate MTEB v2 /
  MMTEB scores quoted elsewhere are not comparable to these.
- **Biggest currency risk (today is 2026-08-01, and my training data ends May 2026):** the
  MTEB(fra) retrieval tasks were largely run against 2024-era models. Qwen3-Embedding (Jun 2025),
  EmbeddingGemma (Sep 2025), granite-embedding r2 and the 2026 API models have **no French
  retrieval entries**. It is entirely possible that a 2025–2026 model beats both of my picks on
  French legal text and simply has not been benchmarked on it. If you want to de-risk this cheaply,
  add **Qwen3-Embedding-0.6B** as a third arm of the A/B — Apache-2.0, CPU-viable, and its one
  French data point (AlloprofReranking 80.38) is the best I found.
- I could not query the live MTEB leaderboard UI (it is a Gradio app); I read the underlying raw
  results repository instead, which is the same data source and is more traceable.

---

## RECOMMENDATION

### Default: `BAAI/bge-m3`

Rationale, in order of weight:

1. **It is the only candidate whose own evidence speaks to long French documents**, and that
   evidence says the thing that matters most for this corpus: on MLDR-fr, sparse (82.7) beats
   dense (73.8) and hybrid reaches 84.2. Insurance and legal text is long and terminology-exact —
   article numbers, defined terms, policy vocabulary — which is the regime where pure-dense
   retrieval underperforms.
2. **It makes the downstream hybrid-retrieval decision nearly free.** Issue #4 blocks #8, #9 and
   #14. BGE-M3 emits dense vectors, learned sparse lexical weights and ColBERT multi-vectors from
   a single forward pass (`sparse_linear.pt` and `colbert_linear.pt` ship in the repo). Choosing
   any other model means that when the hybrid ticket lands you either bolt on a separate BM25
   stack or re-embed the corpus with a different model. Choosing BGE-M3 turns that ticket into a
   configuration change.
3. **8192-token context** removes chunking from the list of things that must be right on day one,
   and lets you A/B chunk sizes without changing models.
4. **MIT licence, ~568M params, ONNX weights published** — free, CPU-viable, unencumbered, and
   with the main CPU-optimisation path available.
5. Unsuperseded: no BGE-M3 replacement exists as of 2026-08-01, and it is still the de facto
   standard for multilingual hybrid (~35M downloads/month).

Accepted weakness, stated plainly: **BGE-M3 has never been evaluated on BSARD**, so I am
recommending it partly on architectural fit rather than on a head-to-head French legal number.
Its one comparable French figure (AlloprofReranking 73.87) is level with, not ahead of, the
runner-up. This is exactly why the A/B below is not a formality.

### Runner-up to A/B: `intfloat/multilingual-e5-large-instruct`

1. **It has the best measured French *legal* retrieval of any MIT-licensed, CPU-viable model:**
   BSARD nDCG@10 24.61 and recall@100 66.22 — ahead of e5-large (21.28 / 63.06),
   bilingual-embedding-large (19.58 / 64.41), gte-multilingual-base (19.06 / 62.61) and far ahead
   of Solon-large (2.08 / 12.61).
2. It is not a one-task fluke: it is top-3 among local models on Syntec (87.80), Alloprof (52.12),
   XPQA fra-fra (72.72) and FQuAD (83.72), and 89.95 on SyntecReranking.
3. Same size class (560M), same licence (MIT), ONNX published — so swapping between the two is a
   pipeline config change plus a re-embed, which is free under your constraint.
4. Its 512-token limit is the real trade-off. It forces disciplined chunking, which for
   article-structured legal text (one article ≈ one chunk) is defensible and may even help.

### How to decide between them

Do not decide on public benchmarks — they disagree and neither covers French insurance. Build a
**~50-query eval set over your actual corpus** (real questions against real Code des assurances
articles and conditions générales), wire it through Langfuse, and measure recall@10 / recall@50.
Given that re-embedding is free, run both, and add **Qwen3-Embedding-0.6B** as a cheap third arm
to cover the currency risk in §7.

Two decision rules worth committing to in advance:

- If BGE-M3's **dense-only** retrieval is within ~2 points of e5-large-instruct, keep BGE-M3 — the
  free sparse leg will more than repay that gap when the hybrid ticket lands.
- If e5-large-instruct wins **dense-only by a wide margin**, that is a signal your chunks are
  short and dense-dominated; take e5-large-instruct and plan a separate BM25 leg for hybrid.

Explicitly **not** recommended: Sentence-CamemBERT and CamemBERT-derived models (128-token limit,
STS-tuned, not retrievers); `jina-embeddings-v3` (strong, but CC-BY-NC-4.0);
`bge-multilingual-gemma2` (best French scores, but 9B and Gemma-licensed);
`mistral-embed` (stale since Dec 2023, no traceable French retrieval number, and it buys you
nothing over a free local model). `Solon-embeddings-large-0.1` should not be the default, but a
20-minute re-run **with its required `query : ` prefix** would settle whether its BSARD collapse
is real — worth doing before writing off the French-specialist line entirely.

---

## Sources

Primary sources used, all consulted 2026-08-01:

- MTEB raw results (per-task JSON, test splits): <https://github.com/embeddings-benchmark/results>
- MTEB(fra) benchmark definition: <https://github.com/embeddings-benchmark/mteb/blob/main/mteb/benchmarks/benchmarks/benchmarks.py>
- Ciancone et al., *MTEB-French*: <https://arxiv.org/abs/2405.20468>
- Louis & Spanakis, *BSARD* (ACL 2022): <https://arxiv.org/abs/2108.11792>
- Chen et al., *M3-Embedding (BGE-M3)*: <https://arxiv.org/abs/2402.03216>
- Hugging Face model cards and raw `config.json` / `sentence_bert_config.json` /
  `config_sentence_transformers.json` for: `BAAI/bge-m3`, `BAAI/bge-multilingual-gemma2`,
  `intfloat/multilingual-e5-{small,base,large,large-instruct}`,
  `OrdalieTech/Solon-embeddings-{base,large}-0.1`,
  `dangvantuan/sentence-camembert-{base,large}`, `Lajavaness/bilingual-embedding-large`,
  `jinaai/jina-embeddings-v3`, `Alibaba-NLP/gte-multilingual-base`,
  `Qwen/Qwen3-Embedding-{0.6B,4B,8B}`, `google/embeddinggemma-300m`
- Hugging Face Hub API (`/api/models/...`) for parameter counts, licences, file listings and
  last-modified dates
- Mistral model card and models overview: <https://docs.mistral.ai/models/model-cards/mistral-embed-23-12>
