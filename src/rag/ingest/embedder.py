"""BGE-M3 dense + M3 learned sparse, one forward pass per batch (SPEC §5, ADR-0004, #26).

`BGEM3FlagModel` is BGE-M3's own reference implementation: dense and the model's learned
sparse lexical weights come out of the same `encode()` call, which is what keeps the two
vectors in lockstep on one point rather than a separate embed step per vector kind
(ADR-0005). `return_colbert_vecs` is always `False` — SPEC §5 rejects the ColBERT head on
the model's own French numbers (MLDR-fr dense+sparse 84.2 vs all-three 83.9), and computing
it here would spend the one forward pass this design exists to keep single.

The model loads once per process (`lru_cache`) and its weights land under gitignored
`data/raw/` (SPEC §16.1) rather than the default `~/.cache/huggingface` a fresh machine
would otherwise silently populate. `tokenizer.py`'s tokenizer-only load is untouched by
this — that module never imports this one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import models

from rag.ingest.upsert import Embedding

__all__ = ["DENSE_DIM", "MODEL_CACHE_DIR", "MODEL_ID", "embed_batch"]

MODEL_ID = "BAAI/bge-m3"
DENSE_DIM = 1024

# SPEC §16.1 — "data/raw/ ← gitignored: zip, extraction, parquet, model cache".
MODEL_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "hf_cache"


@lru_cache(maxsize=1)
def _model() -> BGEM3FlagModel:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # fp16 buys speed on a GPU this project doesn't have (SPEC §14.4 — CPU-viable by
    # design); fp32 also matches the ~15 MB dense-footprint sanity check (SPEC §4.4).
    return BGEM3FlagModel(MODEL_ID, use_fp16=False, cache_dir=str(MODEL_CACHE_DIR))


def embed_batch(texts: Sequence[str]) -> list[Embedding]:
    """The `EmbedFn` shape `upsert.py` expects: dense fp32 + M3 learned sparse, one
    forward pass over the whole batch, in the same order as `texts`."""
    if not texts:
        return []
    output = _model().encode(
        list(texts),
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense_vecs = output["dense_vecs"]
    lexical_weights = output["lexical_weights"]
    if len(dense_vecs) and len(dense_vecs[0]) != DENSE_DIM:
        # The acceptance check in #26 is "fp32 1024-dim is what was written" — a wrong
        # width would otherwise surface only as Qdrant's own dimension-mismatch error at
        # upsert time, several steps downstream of the model that actually got it wrong.
        raise ValueError(
            f"BGE-M3 returned {len(dense_vecs[0])}-dim dense vectors, expected {DENSE_DIM}"
        )
    return [
        ([float(x) for x in dense], _sparse_vector(weights))
        for dense, weights in zip(dense_vecs, lexical_weights, strict=True)
    ]


def _sparse_vector(weights: Mapping[str, float]) -> models.SparseVector:
    """`lexical_weights` is `{token_id (as a string): weight}` — Qdrant wants the raw
    indices and values BGE-M3 assigned, not a sparse vector it generates itself
    (ADR-0005), so this is a type conversion, not a transform."""
    indices = [int(token_id) for token_id in weights]
    values = [float(weight) for weight in weights.values()]
    return models.SparseVector(indices=indices, values=values)
