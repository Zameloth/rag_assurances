"""SPEC §5 / ADR-0004, #39 — the e5-dense + M3-sparse wrapper, against fake models.

Never downloads or runs either real model: the contract under test is (1) e5's own
query/passage asymmetry — an instruction prefix on queries, none on passages — reaching
`FlagModel` through the right method, and (2) `index_embed_batch`/`query_embed_batch`
correctly stitching e5's dense half onto M3's sparse half (M3's own dense half discarded),
never the two real models' retrieval quality.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pytest
from qdrant_client import models

import rag.ingest.e5_embedder as e5_embedder_module
from rag.ingest.e5_embedder import (
    DENSE_DIM,
    QUERY_INSTRUCTION,
    embed_passages,
    embed_queries,
    index_embed_batch,
    query_embed_batch,
)


class _FakeFlagModel:
    """Records which of `encode_queries`/`encode_corpus` was called and with what, so the
    query/passage asymmetry is a first-class assertion rather than an implementation detail
    of a shared `encode()` mock."""

    instances: list[_FakeFlagModel] = []
    dense_dim: int = DENSE_DIM

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self.init_kwargs = kwargs
        self.query_calls: list[Sequence[str]] = []
        self.corpus_calls: list[Sequence[str]] = []
        _FakeFlagModel.instances.append(self)

    def _vectors(self, sentences: Sequence[str]) -> np.ndarray[Any, Any]:
        width = _FakeFlagModel.dense_dim
        return np.array(
            [[float(i)] + [0.0] * (width - 1) for i in range(len(sentences))], dtype=np.float32
        )

    def encode_queries(self, sentences: Sequence[str], **kwargs: Any) -> np.ndarray[Any, Any]:
        self.query_calls.append(sentences)
        return self._vectors(sentences)

    def encode_corpus(self, sentences: Sequence[str], **kwargs: Any) -> np.ndarray[Any, Any]:
        self.corpus_calls.append(sentences)
        return self._vectors(sentences)


class _FakeM3Embedding:
    """Stands in for `rag.ingest.embedder.embed_batch` — distinct, recognisable dense/sparse
    values so a test can tell M3's dense half was discarded rather than merely overwritten
    with the same numbers by coincidence."""

    calls: list[Sequence[str]] = []

    @staticmethod
    def embed_batch(texts: Sequence[str]) -> list[tuple[list[float], models.SparseVector]]:
        _FakeM3Embedding.calls.append(texts)
        return [
            ([999.0] * DENSE_DIM, models.SparseVector(indices=[i], values=[0.5]))
            for i in range(len(texts))
        ]


@pytest.fixture(autouse=True)
def _fake_models(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    _FakeFlagModel.instances.clear()
    _FakeFlagModel.dense_dim = DENSE_DIM
    _FakeM3Embedding.calls.clear()
    monkeypatch.setattr(e5_embedder_module, "FlagModel", _FakeFlagModel)
    monkeypatch.setattr(e5_embedder_module, "_m3_embed_batch", _FakeM3Embedding.embed_batch)
    e5_embedder_module._model.cache_clear()
    yield
    e5_embedder_module._model.cache_clear()


def test_empty_batch_returns_empty_without_touching_the_model() -> None:
    assert embed_passages([]) == []
    assert embed_queries([]) == []
    assert _FakeFlagModel.instances == []


def test_embed_passages_calls_encode_corpus_not_encode_queries() -> None:
    embed_passages(["texte"])

    [instance] = _FakeFlagModel.instances
    assert instance.corpus_calls == [["texte"]]
    assert instance.query_calls == []


def test_embed_queries_calls_encode_queries_not_encode_corpus() -> None:
    embed_queries(["question"])

    [instance] = _FakeFlagModel.instances
    assert instance.query_calls == [["question"]]
    assert instance.corpus_calls == []


def test_model_is_constructed_with_the_query_instruction_and_gitignored_cache_dir() -> None:
    embed_passages(["texte"])

    [instance] = _FakeFlagModel.instances
    assert instance.model_id == "intfloat/multilingual-e5-large-instruct"
    assert instance.init_kwargs["query_instruction_for_retrieval"] == QUERY_INSTRUCTION
    assert instance.init_kwargs["query_instruction_format"] == "Instruct: {}\nQuery: {}"
    assert instance.init_kwargs["use_fp16"] is False
    assert instance.init_kwargs["cache_dir"] == str(e5_embedder_module.MODEL_CACHE_DIR)


def test_model_uses_mean_pooling_not_flagmodels_cls_default() -> None:
    """`FlagModel`'s own default is CLS pooling — wrong for this architecturally
    mean-pooled model (FlagEmbedding's own `E5_MAPPING` entry for this model id pins
    `mean`, applied automatically only through `FlagAutoModel.from_finetuned`, which this
    module doesn't use). Silently defaulting to CLS would degrade every dense vector
    without raising anything, so this is pinned directly rather than left implicit."""
    embed_passages(["texte"])

    [instance] = _FakeFlagModel.instances
    assert instance.init_kwargs["pooling_method"] == "mean"


def test_model_loads_once_across_calls() -> None:
    embed_passages(["a"])
    embed_queries(["b"])

    assert len(_FakeFlagModel.instances) == 1


def test_dense_vectors_are_the_real_1024_width() -> None:
    [dense] = embed_passages(["texte"])

    assert len(dense) == DENSE_DIM == 1024


def test_a_wrong_dense_width_from_the_model_raises() -> None:
    _FakeFlagModel.dense_dim = 4

    with pytest.raises(ValueError, match=str(DENSE_DIM)):
        embed_passages(["texte"])


class TestIndexEmbedBatch:
    def test_dense_half_comes_from_e5_passages(self) -> None:
        [(dense, _)] = index_embed_batch(["texte"])

        [instance] = _FakeFlagModel.instances
        assert instance.corpus_calls == [["texte"]]
        assert instance.query_calls == []
        assert dense != [999.0] * DENSE_DIM  # not M3's dense half

    def test_sparse_half_comes_from_m3_and_m3_dense_half_is_discarded(self) -> None:
        [(_, sparse)] = index_embed_batch(["texte"])

        assert _FakeM3Embedding.calls == [["texte"]]
        assert sparse == models.SparseVector(indices=[0], values=[0.5])

    def test_empty_batch_returns_empty(self) -> None:
        assert index_embed_batch([]) == []


class TestQueryEmbedBatch:
    def test_dense_half_comes_from_e5_queries_with_the_instruction_path(self) -> None:
        [(dense, _)] = query_embed_batch(["question"])

        [instance] = _FakeFlagModel.instances
        assert instance.query_calls == [["question"]]
        assert instance.corpus_calls == []
        assert dense != [999.0] * DENSE_DIM

    def test_sparse_half_comes_from_m3(self) -> None:
        [(_, sparse)] = query_embed_batch(["question"])

        assert _FakeM3Embedding.calls == [["question"]]
        assert sparse == models.SparseVector(indices=[0], values=[0.5])

    def test_empty_batch_returns_empty(self) -> None:
        assert query_embed_batch([]) == []
