"""Cross-encoder rerank — the rung-4 arm (SPEC §9.4, §12.7, §14.4, ADR-0018, #31).

Scores the fused candidate pool against the query with a cross-encoder and replaces each
candidate's `score` outright. ADR-0017 already named this "rung 4's problem": the merged
pool arrives with three incomparable scales (hybrid-fused dense+sparse on both search legs,
filtered dense cosine on the expansion pool), and a cross-encoder relevance score is the one
scale the whole pool can share.

**Two RAM levers, both parameters with named-constant defaults, not a code change** (SPEC
§14.4, this ticket's own acceptance criterion): the cheap arm (`CHEAP_RERANKER_MODEL`,
306M against the default's 568M) and `RerankerBackend.ONNX_INT8`, a local dynamic
quantisation of whichever `model_id` is loaded (fp16 is poorly supported on CPU per SPEC
§9.4, so int8 is the lever named, not fp16). `jinaai/jina-reranker-v2-base-multilingual` is
never wired in here — CC-BY-NC-4.0 in a public portfolio repo.

**Not a LangChain component** (SPEC §9.4, §11.1) — nothing in this module imports
`langfuse` or `langchain`. Hand-wrapping a call to `rerank()` in a Langfuse span, so rung
4's own measurement isn't silently blind, is paired work with @Zameloth per the #31 issue
comment — it belongs at the LangChain/Langfuse retriever boundary (`langchain_retriever.py`
/ wherever that wrapper grows), not here.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any, Protocol, cast

from rag.retrieval.candidates import Candidate

__all__ = [
    "CHEAP_RERANKER_MODEL",
    "DEFAULT_RERANKER_MODEL",
    "MODEL_CACHE_DIR",
    "RerankFn",
    "RerankerBackend",
    "rerank",
]

# SPEC §9.4 — the ladder's primary rung-4 arm.
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# SPEC §9.4/§14.4 — the first RAM lever if the full reranker doesn't fit the ~4.5 GB prod
# budget alongside BGE-M3 and the other demo (§14.4's pre-registered rule).
CHEAP_RERANKER_MODEL = "Alibaba-NLP/gte-multilingual-reranker-base"

# SPEC §16.1 — gitignored. The same physical directory `rag.ingest.embedder.MODEL_CACHE_DIR`
# points at, but redefined rather than imported: `embedder.py` pulls in `FlagEmbedding` and
# BGE-M3's load path at import time, and this module must stay free of that (the same
# independence `tokenizer.py` already keeps from `embedder.py`, for the same reason).
MODEL_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "hf_cache"

# Where a local int8 ONNX export is cached once quantised, keyed by model id (SPEC §14.4).
_ONNX_CACHE_DIR = MODEL_CACHE_DIR / "onnx-int8"
_ONNX_QUANTIZED_FILE = "model_quantized.onnx"


class RerankerBackend(enum.Enum):
    """SPEC §9.4/§14.4's second RAM lever: `ONNX_INT8` is a local dynamic quantisation of
    whichever `model_id` is loaded, reached for when fp32 doesn't fit the prod budget — fp16
    is poorly supported on CPU (SPEC §9.4), so int8 is the lever named, not fp16."""

    FP32 = "fp32"
    ONNX_INT8 = "onnx_int8"


class _ScoringModel(Protocol):
    """The one method both backends expose — `FlagAutoReranker.from_finetuned`'s own
    `AbsReranker.compute_score` shape, so `rerank()` never branches on backend beyond
    loading (`_load_reranker` picks the implementation; `rerank` calls whichever it gets
    back the same way)."""

    def compute_score(
        self, sentence_pairs: list[tuple[str, str]], *, normalize: bool = ...
    ) -> Sequence[float]: ...


RerankFn = Callable[[str, list[Candidate]], list[Candidate]]


def rerank(
    query: str,
    candidates: list[Candidate],
    *,
    model_id: str = DEFAULT_RERANKER_MODEL,
    backend: RerankerBackend = RerankerBackend.FP32,
) -> list[Candidate]:
    """Score `candidates` against `query` with a cross-encoder, and return them reordered
    by that score descending, each candidate's `score` replaced with it.

    `model_id`/`backend` default to the SPEC §9.4 primary arm but are real parameters — the
    same shape `expansion_depth`/`fiche_weights` already take (#29, #30) — so the cheap arm
    and int8 ONNX quantisation are reachable by a caller without editing this module (this
    ticket's own acceptance criterion, and SPEC §14.4's deploy-time RAM rule).

    Empty `candidates` returns `[]` without loading a model — mirrors `embed_batch`'s own
    empty-batch short-circuit (`rag.ingest.embedder`).
    """
    if not candidates:
        return []
    model = _load_reranker(model_id, backend)
    pairs = [(query, str(candidate.payload.get("text", ""))) for candidate in candidates]
    scores = model.compute_score(pairs, normalize=True)
    rescored = [
        replace(candidate, score=float(score))
        for candidate, score in zip(candidates, scores, strict=True)
    ]
    return sorted(rescored, key=lambda candidate: candidate.score, reverse=True)


@cache
def _load_reranker(model_id: str, backend: RerankerBackend) -> _ScoringModel:
    """Loads once per `(model_id, backend)` pair rather than once per process
    (`embed_batch`'s `_model()` is `maxsize=1` because BGE-M3 never varies) — the whole
    point of `model_id`/`backend` being parameters is that more than one combination can be
    requested within one process, e.g. an eval run comparing rung 4's fp32 arm against its
    int8 arm."""
    if backend is RerankerBackend.FP32:
        return _load_fp32(model_id)
    return _load_onnx_int8(model_id)


def _load_fp32(model_id: str) -> _ScoringModel:
    from FlagEmbedding import FlagAutoReranker  # deferred: pulls in torch + transformers

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # fp16 buys speed on a GPU this project doesn't have (SPEC §14.4 — CPU-viable by
    # design, and fp16 is poorly supported on CPU per SPEC §9.4). FlagEmbedding ships no
    # stubs (see the `FlagEmbedding.*` mypy override) so its return is `Any`; `cast` states
    # the contract this module relies on rather than let that `Any` propagate silently.
    return cast(
        _ScoringModel,
        FlagAutoReranker.from_finetuned(
            model_id, use_fp16=False, cache_dir=str(MODEL_CACHE_DIR)
        ),
    )


def _load_onnx_int8(model_id: str) -> _ScoringModel:
    """SPEC §14.4's second RAM lever — a **local** dynamic int8 quantisation of `model_id`,
    exported and cached once under `_ONNX_CACHE_DIR`, rather than depending on a
    pre-quantised file published under some assumed name on the hub. Requires the
    `rerank-onnx` dependency group (`optimum[onnxruntime]`) — not installed by default, the
    same posture the `fetch` group already takes for its rarely-run `pyarrow`/`httpx`.
    """
    from optimum.onnxruntime import (  # deferred: optional group, see pyproject.toml
        AutoQuantizationConfig,
        ORTModelForSequenceClassification,
        ORTQuantizer,
    )
    from transformers import AutoTokenizer  # deferred: same optional group

    quantized_dir = _ONNX_CACHE_DIR / model_id.replace("/", "__")
    if not (quantized_dir / _ONNX_QUANTIZED_FILE).exists():
        quantized_dir.mkdir(parents=True, exist_ok=True)
        exported = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
        quantizer = ORTQuantizer.from_pretrained(exported)
        quantizer.quantize(
            save_dir=str(quantized_dir),
            quantization_config=AutoQuantizationConfig.avx512_vnni(
                is_static=False, per_channel=False
            ),
        )
    model = ORTModelForSequenceClassification.from_pretrained(
        quantized_dir, file_name=_ONNX_QUANTIZED_FILE
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return _OnnxScoringModel(model=model, tokenizer=tokenizer)


@dataclass
class _OnnxScoringModel:
    """Wraps an ORT session + tokenizer behind the same `compute_score` shape
    `FlagAutoReranker.from_finetuned` returns, so `rerank()` treats both backends
    identically."""

    model: Any
    tokenizer: Any

    def compute_score(
        self, sentence_pairs: list[tuple[str, str]], *, normalize: bool = True
    ) -> Sequence[float]:
        import torch  # deferred: same optional group as the ONNX loader above

        queries = [pair[0] for pair in sentence_pairs]
        passages = [pair[1] for pair in sentence_pairs]
        inputs = self.tokenizer(
            queries, passages, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1)
        scores = torch.sigmoid(logits) if normalize else logits
        return [float(score) for score in scores.tolist()]
