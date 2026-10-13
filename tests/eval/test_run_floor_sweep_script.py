"""`scripts/run_floor_sweep.py` plumbing (ADR-0027, #96): pool capture with the pipeline and
reranker faked — no Qdrant, no model. And the isolation guard: only the sweep reads the
calibration file."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from rag.eval.schema import GoldenItem
from rag.retrieval.candidates import Candidate, Provenance, Register
from rag.retrieval.pipeline import RetrievalResult
from rag.retrieval.short_circuit import ShortCircuitPath

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _article(point_id: str, score: float) -> Candidate:
    return Candidate(
        id=point_id,
        score=score,
        register=Register.ARTICLE,
        payload={"legiarti_cid": "G", "text": "texte"},
        provenance=frozenset({Provenance.SEARCH}),
    )


ITEM = GoldenItem(
    id="gs-001", question="q ?", history=(), expected_state="reponse", gold_fiches=("F1",),
    gold_spans=(), gold_articles=("G",), expected_points=("p",), tags=(),
)


def test_capture_item_pool_records_the_reranked_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _load("run_floor_sweep")
    monkeypatch.setattr(script, "rerank", lambda query, candidates: [_article("a#0", 0.7)])

    def fake_rung5(client: Any, embed: Any, question: str, keys: Any, *, rerank_fn: Any) -> RetrievalResult:
        rerank_fn(question, [_article("a#0", 0.0)])
        return RetrievalResult(short_circuit_path=ShortCircuitPath.NO_REFERENCE, contexts=[])

    monkeypatch.setattr(script, "retrieve_rung5", fake_rung5)

    pool = script.capture_item_pool(None, None, set(), ITEM)

    assert [(a.id, a.score) for a in pool.articles] == [("a#0", 0.7)]
    assert pool.short_circuit is False


def test_capture_item_pool_flags_a_short_circuited_item(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _load("run_floor_sweep")
    monkeypatch.setattr(
        script,
        "retrieve_rung5",
        lambda *a, **k: RetrievalResult(short_circuit_path=ShortCircuitPath.RESOLVED, contexts=[]),
    )

    assert script.capture_item_pool(None, None, set(), ITEM).short_circuit is True


def test_no_ladder_script_reads_the_floor_calibration_file() -> None:
    ladder_scripts = sorted((REPO_ROOT / "scripts").glob("run_ladder*.py"))
    ladder_modules = [
        REPO_ROOT / "src" / "rag" / "eval" / f"{name}.py"
        for name in ("run_experiment", "langfuse_sync", "retrieval_run", "ladder_registry", "compare", "paths")
    ]
    ladder_files = [*ladder_scripts, *ladder_modules]

    assert len(ladder_scripts) >= 2  # the guard must be looking at something
    offenders = [p.name for p in ladder_files if "floor-calibration" in p.read_text(encoding="utf-8")]
    assert offenders == []
