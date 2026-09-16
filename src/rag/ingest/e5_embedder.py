"""`intfloat/multilingual-e5-large-instruct` dense vectors, paired with BGE-M3's sparse
half to form rung 6's e5-dense + M3-sparse arm (SPEC §5, ADR-0004, #39).

**e5 emits no sparse vectors at all** (ADR-0004) — every function here that returns a full
`Embedding` pair gets its sparse half from `rag.ingest.embedder.embed_batch`, the same
BGE-M3 wrapper the default arms use, with M3's own dense half computed and discarded. That
is the accepted cost of reusing one sparse source rather than a second sparse-only code
path (mirrored on the query side by `retrieve_rung1`'s own "compute dense, discard sparse").

**e5-instruct is asymmetric, unlike BGE-M3's one symmetric call.** Its own usage
convention (`FlagEmbedding`'s `E5_MAPPING` for this model id, matching the model card) puts
an instruction prefix on queries only: `encode_queries` runs `QUERY_INSTRUCTION` through
`query_instruction_format` ("Instruct: {}\\nQuery: {}") before embedding, `encode_corpus`
embeds passages raw. Collapsing this into one function with a flag would make "did I query
the passage side by mistake" a silent recall regression instead of a code-review-visible
choice — so indexing (`embed_passages`/`index_embed_batch`) and querying
(`embed_queries`/`query_embed_batch`) are four separate names, never one parameterised by a
bool.

Same `data/raw/hf_cache` cache dir as BGE-M3 (SPEC §16.1) — HF's own cache already
namespaces by model id under it, so there is no collision to avoid by giving this model a
second directory.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from FlagEmbedding import FlagModel

from rag.ingest.embedder import MODEL_CACHE_DIR
from rag.ingest.embedder import embed_batch as _m3_embed_batch
from rag.ingest.upsert import Embedding

__all__ = [
    "DENSE_DIM",
    "MODEL_CACHE_DIR",
    "MODEL_ID",
    "QUERY_INSTRUCTION",
    "embed_passages",
    "embed_queries",
    "index_embed_batch",
    "query_embed_batch",
]

MODEL_ID = "intfloat/multilingual-e5-large-instruct"
DENSE_DIM = 1024  # same width as BGE-M3 — SPEC §6's collections need no dim change for rung 6

# FlagEmbedding's own `E5_MAPPING` entry for this model id pins the format; the instruction
# text is the model card's own example retrieval task description.
QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
_QUERY_INSTRUCTION_FORMAT = "Instruct: {}\nQuery: {}"


@lru_cache(maxsize=1)
def _model() -> FlagModel:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # fp16 buys speed on a GPU this project doesn't have (SPEC §14.4 — CPU-viable by
    # design); fp32 also matches the weight-footprint accounting rung 6's RAM measurement
    # is for (both models resident at their real deploy precision, not a lighter stand-in).
    return FlagModel(
        MODEL_ID,
        # FlagModel's own default is CLS pooling (`AbsEmbedder.DEFAULT_POOLING_METHOD`) —
        # wrong for this model. `multilingual-e5-large-instruct` is architecturally mean
        # pooled (FlagEmbedding's own `E5_MAPPING` entry for this model id pins `mean`,
        # applied automatically only via `FlagAutoModel.from_finetuned`, which this module
        # deliberately doesn't use — see the module docstring). Silently defaulting to CLS
        # would degrade every dense vector without raising anything.
        pooling_method="mean",
        query_instruction_for_retrieval=QUERY_INSTRUCTION,
        query_instruction_format=_QUERY_INSTRUCTION_FORMAT,
        normalize_embeddings=True,
        use_fp16=False,
        cache_dir=str(MODEL_CACHE_DIR),
    )


def embed_passages(texts: Sequence[str]) -> list[list[float]]:
    """Indexing-side dense vectors — `encode_corpus`, no instruction prefix, matching
    e5-instruct's own convention that only queries carry one."""
    if not texts:
        return []
    vectors = _model().encode_corpus(list(texts), convert_to_numpy=True)
    return _checked_dense(vectors)


def embed_queries(texts: Sequence[str]) -> list[list[float]]:
    """Query-side dense vectors — `encode_queries`, `QUERY_INSTRUCTION` prefixed onto every
    text by `FlagModel` itself via `query_instruction_format`, never applied here twice."""
    if not texts:
        return []
    vectors = _model().encode_queries(list(texts), convert_to_numpy=True)
    return _checked_dense(vectors)


def _checked_dense(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    dense = [[float(x) for x in row] for row in vectors]
    if dense and len(dense[0]) != DENSE_DIM:
        # Mirrors `rag.ingest.embedder.embed_batch`'s own check (#26) — a wrong width would
        # otherwise surface only as Qdrant's dimension-mismatch error at upsert/query time.
        raise ValueError(f"{MODEL_ID} returned {len(dense[0])}-dim dense vectors, expected {DENSE_DIM}")
    return dense


def index_embed_batch(texts: Sequence[str]) -> list[Embedding]:
    """The rung-6 arm's indexing-time `EmbedFn` (SPEC §5's e5-dense + M3-sparse): dense
    vectors from e5's passage side (`embed_passages`), sparse vectors from BGE-M3's own
    learned lexical weights on the same raw text, M3's dense half discarded."""
    if not texts:
        return []
    dense_vectors = embed_passages(texts)
    m3_embeddings = _m3_embed_batch(texts)
    return [(dense, sparse) for dense, (_, sparse) in zip(dense_vectors, m3_embeddings, strict=True)]


def query_embed_batch(texts: Sequence[str]) -> list[Embedding]:
    """The rung-6 arm's query-time `EmbedFn` — same shape `index_embed_batch` returns, but
    dense vectors come from e5's query side (`embed_queries`) so the query and the index it
    searches live in the same instructed vector space."""
    if not texts:
        return []
    dense_vectors = embed_queries(texts)
    m3_embeddings = _m3_embed_batch(texts)
    return [(dense, sparse) for dense, (_, sparse) in zip(dense_vectors, m3_embeddings, strict=True)]
