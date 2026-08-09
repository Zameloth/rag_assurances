"""SPEC §5 / ADR-0004, #26 — the BGE-M3 wrapper, against a fake model.

Never downloads or runs the real ~2.3 GB model: the contract under test is the shape
conversion (`dense_vecs`/`lexical_weights` -> the `Embedding` tuples `upsert.py` expects)
and that ColBERT is never requested, not BGE-M3's own retrieval quality.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pytest
from qdrant_client import models

import rag.ingest.embedder as embedder_module
from rag.ingest.embedder import DENSE_DIM, embed_batch


class _FakeBGEM3FlagModel:
    """Records its constructor and `encode()` call args; returns BGE-M3-shaped output.

    `dense_dim` is a class attribute, not a constructor arg, because `_model()` builds
    this with no extra kwargs of its own — the one test exercising a width mismatch sets
    it directly before calling `embed_batch`.
    """

    instances: list[_FakeBGEM3FlagModel] = []
    dense_dim: int = DENSE_DIM

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self.init_kwargs = kwargs
        self.encode_calls: list[dict[str, Any]] = []
        _FakeBGEM3FlagModel.instances.append(self)

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> dict[str, Any]:
        self.encode_calls.append({"sentences": sentences, **kwargs})
        width = _FakeBGEM3FlagModel.dense_dim
        dense_vecs = np.array(
            [[float(i)] + [0.0] * (width - 1) for i in range(len(sentences))], dtype=np.float32
        )
        lexical_weights = [{"10": np.float32(0.5), "20": np.float32(0.25)} for _ in sentences]
        return {"dense_vecs": dense_vecs, "lexical_weights": lexical_weights, "colbert_vecs": None}


@pytest.fixture(autouse=True)
def _fake_model(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Swaps in the fake class and clears the `lru_cache` singleton on both sides of the
    test — a leaked cache would otherwise let one test's fake model answer another's."""
    _FakeBGEM3FlagModel.instances.clear()
    _FakeBGEM3FlagModel.dense_dim = DENSE_DIM
    monkeypatch.setattr(embedder_module, "BGEM3FlagModel", _FakeBGEM3FlagModel)
    embedder_module._model.cache_clear()
    yield
    embedder_module._model.cache_clear()


def test_empty_batch_returns_empty_without_touching_the_model() -> None:
    assert embed_batch([]) == []
    assert _FakeBGEM3FlagModel.instances == []


def test_dense_and_sparse_come_back_in_input_order() -> None:
    result = embed_batch(["premier", "second"])

    assert len(result) == 2
    dense_0, sparse_0 = result[0]
    dense_1, sparse_1 = result[1]
    assert dense_0[0] == 0.0
    assert dense_1[0] == 1.0
    assert isinstance(sparse_0, models.SparseVector)


def test_dense_vectors_are_the_real_1024_width() -> None:
    """SPEC §4.4's ~15 MB footprint sanity check assumes fp32 1024-dim — this is the part
    of that check `make ingest`'s own output can't verify by itself."""
    [(dense, _)] = embed_batch(["texte"])

    assert len(dense) == DENSE_DIM == 1024


def test_a_wrong_dense_width_from_the_model_raises() -> None:
    _FakeBGEM3FlagModel.dense_dim = 4

    with pytest.raises(ValueError, match=str(DENSE_DIM)):
        embed_batch(["texte"])


def test_sparse_vector_converts_string_token_ids_and_numpy_weights() -> None:
    [(_, sparse)] = embed_batch(["texte"])

    assert sparse.indices == [10, 20]
    assert sparse.values == [0.5, 0.25]
    assert all(isinstance(v, float) for v in sparse.values)


def test_colbert_is_never_requested() -> None:
    embed_batch(["texte"])

    [instance] = _FakeBGEM3FlagModel.instances
    [call] = instance.encode_calls
    assert call["return_colbert_vecs"] is False
    assert call["return_dense"] is True
    assert call["return_sparse"] is True


def test_model_loads_once_across_calls() -> None:
    embed_batch(["a"])
    embed_batch(["b"])

    assert len(_FakeBGEM3FlagModel.instances) == 1


def test_model_is_constructed_fp32_with_the_gitignored_cache_dir() -> None:
    embed_batch(["texte"])

    [instance] = _FakeBGEM3FlagModel.instances
    assert instance.model_id == "BAAI/bge-m3"
    assert instance.init_kwargs["use_fp16"] is False
    assert instance.init_kwargs["cache_dir"] == str(embedder_module.MODEL_CACHE_DIR)
