# Context — rag_assurances

A French-language RAG assistant that answers a **curious consumer**'s questions about
French insurance law, in two parts: an explanation in consumer French, plus the Code des
assurances article behind it.

This file is the project's **ubiquitous language**. Use these terms — in code, in issue
titles, in test names, in prompts — and avoid the synonyms marked ✗. Full design detail
lives in [`SPEC.md`](SPEC.md); the reasoning behind each decision lives in
[`docs/adr/`](docs/adr/) and in the issues linked from
[the map](https://github.com/Zameloth/rag_assurances/issues/1).

---

## The two registers

The single most load-bearing distinction in the system. Questions arrive in one register
and must be answered from both.

| Term | Meaning |
|---|---|
| **register** | Which of the two document populations a chunk belongs to: `fiche` or `article`. A **retrieval annotation**, never a stored payload field — it is derivable from the collection queried, and a stored copy can only ever be wrong. |
| **fiche** | A service-public.fr consumer-guidance document, in **consumer French**. 87 in corpus. Ships as verbatim DILA XML. Identity: `fiche_id` (`F2594`). ✗ *page*, *document*, *guide* |
| **article** | An in-force Code des assurances article, in **legal French**. 2,377 in corpus. Identity: `legiarti_cid`. ✗ *law*, *statute*, *loi* |
| **consumer French / legal French** | The two language registers. Bridging them is the retrieval problem the whole system exists to solve. |

---

## Corpus

| Term | Meaning |
|---|---|
| **the scope rule** | The definition of what is in corpus: *any service-public.fr insurance fiche whose `<dc:source>` resolves into the Code des assurances, plus the in-force Code des assurances itself.* A **rule, not a document list** — say so whenever it is written down, because a list is what readers assume. |
| **Tier 1** | The public, Licence Ouverte 2.0 corpus: fiches + Code des assurances. The only tier that exists. *(Tiers 2 (ACPR) and 3 (insurer contracts) were surveyed and ruled out of scope.)* |
| **conditions générales / CG** | Insurer contract wordings. **Out of scope and never in the repo** — not redistributable. The term appears only in rejection notes. |
| **`<dc:source>`** | The DILA XML element on each fiche naming the `LEGISCTA` **sections** it rests on. An **editorial join**, section-level not article-level. Drives expansion; explicitly *not* ground truth. |
| **`LEGISCTA` / section** | A Code des assurances section id. Stored on articles as `section_id` (indexed), on fiche chunks as `section_ids` (a list, read but never filtered on). |
| **`cid` vs version id** | Every article carries two LEGIARTI ids: `cid` (the **chronicle** — stable across amendments) and `id` (this **version**). **`cid` is identity** everywhere — point ids, gold labels, joins. The version id has exactly two jobs: provenance, and the Légifrance click-through URL. 52% of articles diverge. ✗ using bare *LEGIARTI* without saying which |
| **corpus refresh** | A **manual, reviewed commit** that re-fetches upstream and reports added / removed / **changed-text-under-the-same-`cid`**. Never scheduled. |
| **`corpus_manifest.json`** | Script-emitted. Two jobs: the Licence Ouverte three-part attribution record, and the **provenance pin** (`sha256`, `retrieved_at`, `document_count`, `mirror_of`). |
| **sources and decisions, never derived binaries** | The repo's committing principle. Named exceptions: the golden set and per-item eval scores. |

---

## Chunking and indexing

| Term | Meaning |
|---|---|
| **point** | One indexed unit in Qdrant — a chunk plus its payload plus its dense and sparse vectors. **3,687 in total** (2,805 articles + 882 fiches). ✗ *vector*, *record*, *row* when the whole unit is meant |
| **chunk** | The text of a point. `text` is **always the raw chunk** — verbatim source, minus stripped `<table>` content. Enrichment never changes it. |
| **the band** | The 512-token ceiling (BGE-M3 tokenizer), shared by both embedder arms. ✗ *chunk size*, *max length* |
| **the merge floor** | 100 tokens. Adjacent sibling nodes merge up to it, **never across a `<Chapitre>`/`<SousChapitre>`/`<Cas>` boundary**. |
| **body elements** | The four fiche elements carrying prose: `<Introduction>`, `<Texte>`, `<Conclusion>`, `<ListeSituations>`. The other 34 are navigation. |
| **`<Cas>` / `cas_label`** | A conditional branch in a fiche (*"Si vous êtes locataire"*). `cas_label` carries the condition onto the chunk, because chunk text often does not repeat it. On 98 fiche chunks. |
| **stub point** | A point emitted for an annexe left under 32 prose tokens after table-stripping: `citation_id` + title + a `table non indexée` marker + URL. Keeps it retrievable and citable. |
| **`citation_id`** | DILA's raw `num`, **verbatim, never null, never normalized**. What the model copies and what citation checks compare. |
| **`lookup_key`** | The strict-normalized article number, **indexed, nullable** (null for the 21 annexes). The short-circuit's only key. Distinct from `citation_id` — one field cannot serve both jobs. |
| **arm** | One configuration of the pipeline. An index-bearing arm is a **real collection behind a stable alias**, so switching is an alias update and no code names a physical collection. See also **rung**. |
| **index-bearing vs runtime** | Which arm "ships" decomposes into two: **index-bearing** (embedder, chunk config, enrichment) is fixed in the published artifact; **runtime** (reranker, weights, floor, expansion cap) is app config. The RAM rule is runtime-only. |

---

## Retrieval

| Term | Meaning |
|---|---|
| **the short-circuit** | The path that skips search entirely when the **raw user turn** carries an article reference that passes the `lookup_key` membership check. Resolves by metadata lookup. |
| **the membership check** | Normalizing a scanned reference and confirming it is in the loaded `lookup_key` set **before** short-circuiting. Turns any future regex gap into normal search rather than a confidently wrong article. |
| **fiche leg / article leg** | The two hybrid search paths. Fiche leg is dense-leaning, article leg sparse-leaning — the MIRACL/MLDR regime inverts with document length. |
| **expansion** | The third path: top-3 fiches → their `<dc:source>` sections → a **filtered vector search** within those sections, capped at 40. The **spine** of the architecture — it *removes* the consumer→legal hop rather than improving it. ✗ *fiche-to-article lookup*, *chaining* |
| **provenance** | Whether a candidate arrived by expansion, by search, or by the short-circuit's metadata **lookup** (#28). A retrieval **annotation** and a **set**, not a scalar — an article can arrive by both legs and the sets are unioned. `lookup` never joins that union: the short-circuit skips search entirely, so a lookup-provenance candidate is never merged with a search- or expansion-sourced one. |
| **the quota** | Context assembly: **4 fiche chunks + 4 articles**, filled separately after reranking, article slots preferring expansion-sourced. |
| **the relevance floor** | The threshold on article slots. **Pad nothing.** |
| **the no-article marker** | What enters the context when the floor is not met. The model must *state* the absence; it is an answer, not a refusal. |
| **condensation** | The pre-retrieval rewrite of `(history, follow-up)` into a standalone query. Fires **only when history exists** and **never when the short-circuit fires**. Its output **never reaches generation**. ✗ *query rewriting*, *reformulation* |
| **passthrough** | The condenser returning the question verbatim — handled *in the prompt*, detected by string equality. There is no "does this need condensing?" gate. |
| **`condense_status`** | The typed enum recording which condenser path was taken, **computed by code, never by the model**. Fallback and passthrough rates are derived views over it. |
| **reference monotonicity** | `refs(condensed) ⊆ refs(raw_turn)`, enforced by the sanitizer. Makes a fabricated article reference unable to enter the pipeline from the condenser at all. |

---

## Generation

| Term | Meaning |
|---|---|
| **the envelope** | The schema-enforced discriminated union the model returns — `reponse` \| `refus`, with `explanation`, `fondement_juridique`, `aucun_fondement`, `motif`. The schema constrains the **envelope, not the writing**. ✗ *the answer object*, *the response* |
| **the four terminal states** | `reponse` · `reponse_sans_article` · regulated-act `refus` · `hors_corpus` `refus`. Kept distinguishable on purpose: a **scope** failure and a **retrieval** failure have completely different fixes. |
| **`fondement_juridique`** | The typed citation list — the second half of the answer contract. ✗ *sources*, *references*, *citations* when the field is meant |
| **`motif`** | The refusal-class enum: `recommandation_produit` \| `conseil_action` \| `hors_corpus`. |
| **the refusal line** | **Recommendation, not personalisation.** Mapping a described situation onto the applicable rule is *in* scope; recommending a product or a course of action is not. Refusals still answer the informational part. |
| **`cited ⊆ retrieved_context`** | The citation guardrail. Deliberately **not** `⊆ corpus` — a real-but-unretrieved article is parametric-memory fabrication. **Never auto-repaired.** |
| **history stripping** | Prior assistant turns carry prose with prior `fondement_juridique` lists **removed**. Load-bearing for two components: it keeps the citation check strictly per-turn, and it leaves the condenser no honest source for an unasked-for reference. |
| **the disclaimer** | « information, pas conseil ». **App-rendered boilerplate**, never generated per call. |

---

## Evaluation

| Term | Meaning |
|---|---|
| **the golden set** | 60 hand-annotated items in one repo-canonical YAML file. The most expensive artifact in the repo. ✗ *test set*, *eval set* when this specific file is meant |
| **the ladder** | The **six-rung ablation** of the retrieval stack, one variable per rung, previous winner frozen. Fully deterministic and API-free. |
| **rung** | One step of the ladder. Rung 3 (`<dc:source>` expansion) is **the headline experiment**: if a curated editorial join doesn't beat embedding similarity on article recall, the two-register architecture is decoration. |
| **the working set** | The ~44 single-turn golden items carrying gold contexts — what the ladder runs. Article-recall metrics use the 36 with non-empty `gold_articles`. |
| **pre-ladder A/B** | A build-configuration choice settled **before rung 1**, on the 38 retrieval-bearing items, under the same adoption rule. Two exist: article breadcrumb and fiche header enrichment. **Not rungs.** |
| **the primary metric** | The one metric a rung is decided on, **named before the rung runs**. Committed as a table, or per-rung choice becomes post-hoc metric selection. |
| **decision metric vs diagnostic** | Four decide (fiche/article recall@4, zero-article rate, floor correctness); the rest are **read to understand a rung, never to win one**. |
| **the adoption rule** | Adopt iff net discordant pairs on the primary ≥ **4** and no other decision metric regresses by more than 1 net item. **Otherwise keep the incumbent — inconclusive always resolves to no change.** |
| **paired per-item deltas** | The only legitimate comparison here. Every rung runs the same items, so item difficulty cancels. Never two independent proportions. |
| **discordant pair** | An item where two arms disagree on the primary metric. The unit the adoption rule counts. |
| **floor correctness** | Scores the 8 `reponse_sans_article` items, where **empty `gold_articles` is the correct answer**. Exists because *recall is monotone in retrieving more* and the relevance floor was otherwise measured by nothing. |
| **zero-article rate** | Rung 5's primary. The quota was built against an *intermittent* failure, which a mean cannot see. |
| **`gold_spans`** | Verbatim fiche text, invariant to chunking. A **regression check**, not a tuning signal — nothing in this corpus is cut arbitrarily, so containment cannot discriminate granularity. |
| **`expected_points`** | 1–3 terse assertions per item, scored as **coverage, not similarity**. There are no reference answers. Refusal items carry them too. |
| **the judge** | The LLM-as-judge for the two non-deterministic generation metrics. **Must be a different family from every generation arm.** |
| **the calibration set** | 12 clean/faulted answer **pairs**, hand-authored and hand-labelled. **Built, not sampled** — a sampled set contains no failures and so cannot test a judge whose job is catching failures. Lives on as a **judge regression test**. |
| **`compare.py`** | The arbiter of the adoption rule. **Langfuse is the trace viewer and run log, not the comparison surface.** |
| **per-item scores** | What gets persisted to git — not aggregates. The adoption rule counts items, and Langfuse's free tier drops traces at 30 days. |

---

## Application and operations

| Term | Meaning |
|---|---|
| **the pipeline is a library** | `compare.py` and the web app share one code path; the app adds no logic. Non-negotiable — otherwise the evaluated system and the served system drift. |
| **stateless** | The app holds no session. History is client-side and posted back each turn — and is therefore **untrusted input**, trimmed server-side. |
| **stage events** | The SSE progress signal (*chargement → condensation → recherche → génération*). **Not token streaming**, which was rejected: a strict `json_schema` response streams as unreadable raw JSON. |
| **sleep / the group** | Container-level sleep via Sablier. **Both app and Qdrant** are in group `rag`; the group name must be globally unique across the VPS. |
| **`/health`** | Three clauses: **models loaded AND Qdrant reachable AND alias target matches `index_lock.json`.** Sablier closes its waiting page on this signal, so anything omitted is handed to a visitor as a failure. |
| **the RAM budget** | **~4.5 GB** — what is free *with the other demo awake*, on a box with **no swap**. The constraint is **concurrency, not capacity**, and the enforcer is an OOM killer that picks its own victim. |
| **the pre-registered arm rule** | Prod runs the ladder-winning arm **if it fits**; else the cheapest that does, with the divergence recorded in the README. Never discovered at deploy time. |
| **the points dump** | The delivery artifact — one Parquet file per register on a GitHub Release. **Not a Qdrant snapshot**: `indexing_threshold=0` means there is no built index to preserve. |
| **the alias flip** | How a restored index goes live: upsert into `<register>__<release-tag>`, verify the count, then move the stable alias. Atomic, so a partial write never acquires the alias. |
| **`index_lock.json`** | The committed provenance pointer — release tag, `corpus_manifest_sha256`, embedder, chunk config, per-register count and asset sha256, ladder rung. |

---

## Conventions worth stating

- **Measure before deciding.** Several decisions here reversed on a measurement: `texte` has
  zero newlines; 18% of article numbers have three or more segments; `<ListeSituations>` is a
  body element; overlap would repair five chunks. Assertions, not assumptions.
- **A typed field beats an inference.** `motif`, `condense_status` and `expected_state` all
  exist because a component doing several jobs stays auditable only when the branch it took is
  recorded rather than reconstructed downstream.
- **Failures must localise.** Recall at candidate *and* final depth; citation correctness
  distinct from article recall; the condensed query kept out of generation. Each pair separates
  two failures with different fixes.
- **Inconclusive keeps the incumbent.** Everywhere, not just on the ladder.
