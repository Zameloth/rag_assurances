"""SPEC §9.4/§14.4, ADR-0018, #31 — the rung-4 cross-encoder, against fake scoring models.

Never downloads or runs the real reranker checkpoints: the contract under test is
`rerank()`'s own logic (pair-building, score replacement, reorder), that `model_id`/
`backend` select without a code change, and the `_OnnxScoringModel` sigmoid-normalisation
math — not `bge-reranker-v2-m3`'s or `gte-multilingual-reranker-base`'s actual relevance
quality.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pytest

import rag.retrieval.rerank as rerank_module
from rag.retrieval.candidates import Candidate, Provenance, Register
from rag.retrieval.rerank import (
    CHEAP_RERANKER_MODEL,
    DEFAULT_RERANKER_MODEL,
    RerankerBackend,
    _OnnxScoringModel,
    rerank,
)


def _candidate(id_: str, text: str, score: float = 0.0) -> Candidate:
    return Candidate(
        id=id_,
        score=score,
        register=Register.ARTICLE,
        payload={"text": text},
        provenance=frozenset({Provenance.SEARCH}),
    )


class _FakeScoringModel:
    """Scores each pair by the passage's length — deterministic, and never the pair
    order the candidates arrived in, so a test can tell "rescored" from "left alone"."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def compute_score(
        self, sentence_pairs: list[tuple[str, str]], *, normalize: bool = True
    ) -> Sequence[float]:
        self.calls.append({"sentence_pairs": sentence_pairs, "normalize": normalize})
        return [float(len(passage)) for _query, passage in sentence_pairs]


@pytest.fixture(autouse=True)
def _clear_reranker_cache() -> Iterator[None]:
    rerank_module._load_reranker.cache_clear()
    yield
    rerank_module._load_reranker.cache_clear()


def test_empty_candidates_returns_empty_without_loading_a_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(model_id: str, backend: RerankerBackend) -> Any:
        raise AssertionError("must not load a model for an empty candidate list")

    monkeypatch.setattr(rerank_module, "_load_reranker", _fail)

    assert rerank("une question", []) == []


def test_scores_and_reorders_by_the_cross_encoder_score_descending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeScoringModel()
    monkeypatch.setattr(rerank_module, "_load_reranker", lambda model_id, backend: fake)
    candidates = [_candidate("short", "abc"), _candidate("long", "abcdefghij")]

    result = rerank("une question", candidates)

    assert [c.id for c in result] == ["long", "short"]
    assert [c.score for c in result] == [10.0, 3.0]


def test_pairs_are_built_from_the_query_and_each_candidate_s_text_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeScoringModel()
    monkeypatch.setattr(rerank_module, "_load_reranker", lambda model_id, backend: fake)

    rerank("une question", [_candidate("a", "le texte")])

    [call] = fake.calls
    assert call["sentence_pairs"] == [("une question", "le texte")]
    assert call["normalize"] is True


def test_register_payload_and_provenance_survive_the_rescore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeScoringModel()
    monkeypatch.setattr(rerank_module, "_load_reranker", lambda model_id, backend: fake)
    original = _candidate("a", "texte", score=0.1)

    [rescored] = rerank("q", [original])

    assert rescored.id == original.id
    assert rescored.register == original.register
    assert rescored.payload == original.payload
    assert rescored.provenance == original.provenance
    assert rescored.score != original.score


def test_model_id_and_backend_default_to_the_spec_primary_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def _load(model_id: str, backend: RerankerBackend) -> _FakeScoringModel:
        seen["model_id"] = model_id
        seen["backend"] = backend
        return _FakeScoringModel()

    monkeypatch.setattr(rerank_module, "_load_reranker", _load)

    rerank("q", [_candidate("a", "texte")])

    assert seen["model_id"] == DEFAULT_RERANKER_MODEL == "BAAI/bge-reranker-v2-m3"
    assert seen["backend"] is RerankerBackend.FP32


def test_model_id_and_backend_are_overridable_by_the_caller_without_a_code_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This ticket's own acceptance criterion: the cheap arm and int8 ONNX are reachable
    through `model_id`/`backend`, the same shape `expansion_depth`/`fiche_weights` already
    take (#29, #30)."""
    seen: dict[str, Any] = {}

    def _load(model_id: str, backend: RerankerBackend) -> _FakeScoringModel:
        seen["model_id"] = model_id
        seen["backend"] = backend
        return _FakeScoringModel()

    monkeypatch.setattr(rerank_module, "_load_reranker", _load)

    rerank(
        "q",
        [_candidate("a", "texte")],
        model_id=CHEAP_RERANKER_MODEL,
        backend=RerankerBackend.ONNX_INT8,
    )

    assert seen["model_id"] == CHEAP_RERANKER_MODEL == "Alibaba-NLP/gte-multilingual-reranker-base"
    assert seen["backend"] is RerankerBackend.ONNX_INT8


def test_jina_reranker_is_never_the_default_or_cheap_arm() -> None:
    """SPEC §9.4 / this ticket's own acceptance criterion —
    `jinaai/jina-reranker-v2-base-multilingual` is CC-BY-NC-4.0, out for a public repo."""
    banned = "jinaai/jina-reranker-v2-base-multilingual"
    assert banned != DEFAULT_RERANKER_MODEL
    assert banned != CHEAP_RERANKER_MODEL


class TestLoadRerankerDispatch:
    """`_load_reranker` is the one seam between `rerank()`'s pure logic and the two real
    loaders — proving it dispatches on `backend` is enough here; `_load_fp32`'s own body
    is exercised for real (against a fake `FlagAutoReranker`) below, the same way
    `embedder.py`'s tests swap in `_FakeBGEM3FlagModel`. `_load_onnx_int8`'s body needs
    `optimum`/`onnxruntime`, not installed by default (the `rerank-onnx` group,
    pyproject.toml), so it stays exercised only by hand — the same posture the `fetch`
    group's real network calls already take."""

    def test_fp32_backend_calls_the_fp32_loader(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(rerank_module, "_load_fp32", lambda model_id: calls.append(model_id))

        rerank_module._load_reranker(DEFAULT_RERANKER_MODEL, RerankerBackend.FP32)

        assert calls == [DEFAULT_RERANKER_MODEL]

    def test_onnx_int8_backend_calls_the_onnx_loader(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            rerank_module, "_load_onnx_int8", lambda model_id: calls.append(model_id)
        )

        rerank_module._load_reranker(CHEAP_RERANKER_MODEL, RerankerBackend.ONNX_INT8)

        assert calls == [CHEAP_RERANKER_MODEL]

    def test_caches_per_model_id_and_backend_pair(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(rerank_module, "_load_fp32", lambda model_id: calls.append(model_id))

        rerank_module._load_reranker(DEFAULT_RERANKER_MODEL, RerankerBackend.FP32)
        rerank_module._load_reranker(DEFAULT_RERANKER_MODEL, RerankerBackend.FP32)
        rerank_module._load_reranker(CHEAP_RERANKER_MODEL, RerankerBackend.FP32)

        assert calls == [DEFAULT_RERANKER_MODEL, CHEAP_RERANKER_MODEL]


class TestLoadFp32:
    """`_load_fp32`'s own body, for real, against a fake `FlagAutoReranker` — mirrors
    `test_embedder.py`'s `_FakeBGEM3FlagModel`: the deferred `from FlagEmbedding import
    FlagAutoReranker` re-resolves the attribute off the real `FlagEmbedding` package at
    call time, so patching it there is enough to intercept the call without ever
    downloading real weights."""

    def test_builds_the_cache_dir_and_forwards_model_id_fp32_and_cache_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        import FlagEmbedding

        calls: list[dict[str, Any]] = []

        class _FakeFlagAutoReranker:
            @classmethod
            def from_finetuned(cls, model_name_or_path: str, **kwargs: Any) -> _FakeScoringModel:
                calls.append({"model_name_or_path": model_name_or_path, **kwargs})
                return _FakeScoringModel()

        monkeypatch.setattr(FlagEmbedding, "FlagAutoReranker", _FakeFlagAutoReranker)
        monkeypatch.setattr(rerank_module, "MODEL_CACHE_DIR", tmp_path / "hf_cache")

        model = rerank_module._load_fp32(DEFAULT_RERANKER_MODEL)

        assert isinstance(model, _FakeScoringModel)
        [call] = calls
        assert call["model_name_or_path"] == DEFAULT_RERANKER_MODEL
        assert call["use_fp16"] is False
        assert call["cache_dir"] == str(tmp_path / "hf_cache")
        assert (tmp_path / "hf_cache").is_dir()


class TestOnnxScoringModel:
    """`_OnnxScoringModel` is the thin wrapper making an ORT session/tokenizer pair look
    like the same `compute_score` shape `FlagAutoReranker.from_finetuned` returns — its
    sigmoid-normalisation math is real torch, exercised without touching optimum/onnxruntime
    (not installed by default — see the `rerank-onnx` group, pyproject.toml)."""

    def test_normalize_true_applies_sigmoid(self) -> None:
        import torch

        class _FakeOutput:
            logits = torch.tensor([[0.0], [100.0]])

        class _FakeModel:
            def __call__(self, **inputs: Any) -> _FakeOutput:
                return _FakeOutput()

        class _FakeTokenizer:
            def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                return {}

        wrapper = _OnnxScoringModel(model=_FakeModel(), tokenizer=_FakeTokenizer())
        # sigmoid(0.0) == 0.5, sigmoid(100.0) ~= 1.0.
        scores = wrapper.compute_score([("q", "a"), ("q", "b")], normalize=True)

        assert scores[0] == pytest.approx(0.5)
        assert scores[1] == pytest.approx(1.0, abs=1e-6)

    def test_normalize_false_returns_raw_logits(self) -> None:
        import torch

        class _FakeOutput:
            logits = torch.tensor([[-2.0], [3.5]])

        class _FakeModel:
            def __call__(self, **inputs: Any) -> _FakeOutput:
                return _FakeOutput()

        class _FakeTokenizer:
            def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                return {}

        wrapper = _OnnxScoringModel(model=_FakeModel(), tokenizer=_FakeTokenizer())

        scores = wrapper.compute_score([("q", "a"), ("q", "b")], normalize=False)

        assert scores == pytest.approx([-2.0, 3.5])
