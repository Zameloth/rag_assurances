"""`run_retrieval_ladder` — SPEC §12.5/§12.11/§12.12, ADR-0011, ADR-0012, #35.

`run_retrieval_ladder` is exercised end to end against a hand-written fake `Langfuse`
client/dataset — real `ExperimentItemResult`/`ExperimentResult`/`DatasetItem` objects, but a
fake `run_experiment()` that actually invokes `task`/the evaluators synchronously instead of
going over the network. Same posture as `test_langfuse_sync.py`'s `_FakeLangfuseClient`: no
live Langfuse project needed for a unit test, and this fake is a closer stand-in than that
one, since `run_retrieval_ladder`'s own logic runs almost entirely *inside* the call to
`run_experiment()`.

Retrieval itself runs for real against `QdrantClient(":memory:")` (SPEC §6.3 — plumbing
assertions only, never a ranking/recall claim), the same fixtures `test_retrieval_pipeline.py`
uses.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from conftest import CreateCollection, raw_point, stub_embed
from langfuse.api import DatasetItem, DatasetStatus
from langfuse.experiment import Evaluation, ExperimentItemResult, ExperimentResult
from qdrant_client import QdrantClient

import rag.eval.run_experiment as run_experiment_module
from rag.config import Settings
from rag.eval.langfuse_sync import RETRIEVAL_DATASET_NAME, retrieval_dataset_items
from rag.eval.retrieval_metrics import ItemRetrievalScore
from rag.eval.retrieval_run import load_run
from rag.eval.run_experiment import MAX_CONCURRENCY, _score_evaluations, run_retrieval_ladder
from rag.eval.schema import GoldenItem, dump_golden_set
from rag.retrieval.legs import ARTICLES_ALIAS, FICHES_ALIAS


def make_score(**overrides: object) -> ItemRetrievalScore:
    defaults: dict[str, object] = dict(
        item_id="gs-001",
        short_circuit_path="none",
        fiche_recall_at_4=None,
        fiche_recall_at_10=None,
        fiche_recall_at_candidate=None,
        article_recall_at_4=None,
        article_recall_at_10=None,
        article_recall_at_candidate=None,
        zero_articles=None,
        floor_correct=None,
        span_containment_at_4=None,
    )
    defaults.update(overrides)
    return ItemRetrievalScore(**defaults)  # type: ignore[arg-type]


class TestScoreEvaluations:
    def test_skips_every_none_metric(self) -> None:
        assert _score_evaluations(make_score()) == []

    def test_one_evaluation_per_non_none_metric(self) -> None:
        evaluations = _score_evaluations(make_score(fiche_recall_at_4=1.0, article_recall_at_4=0.5))

        names = {evaluation.name for evaluation in evaluations}
        assert names == {"fiche_recall_at_4", "article_recall_at_4"}

    def test_numeric_metrics_keep_their_float_value_and_data_type(self) -> None:
        [evaluation] = _score_evaluations(make_score(fiche_recall_at_4=0.75))

        assert evaluation.name == "fiche_recall_at_4"
        assert evaluation.value == 0.75
        assert evaluation.data_type == "NUMERIC"

    def test_boolean_metrics_are_tagged_boolean_not_numeric(self) -> None:
        [evaluation] = _score_evaluations(make_score(floor_correct=True))

        assert evaluation.name == "floor_correct"
        assert evaluation.value is True
        assert evaluation.data_type == "BOOLEAN"

    def test_never_emits_an_evaluation_for_item_id_or_short_circuit_path(self) -> None:
        evaluations = _score_evaluations(
            make_score(fiche_recall_at_4=1.0, item_id="gs-999", short_circuit_path="resolved")
        )

        names = {evaluation.name for evaluation in evaluations}
        assert "item_id" not in names
        assert "short_circuit_path" not in names


def golden_item(
    id: str = "gs-001",
    *,
    question: str = "je suis locataire, dois-je m'assurer ?",
    expected_state: str = "reponse",
    gold_fiches: tuple[str, ...] = (),
    gold_spans: tuple[str, ...] = (),
    gold_articles: tuple[str, ...] = (),
) -> GoldenItem:
    return GoldenItem(
        id=id,
        question=question,
        history=(),
        expected_state=expected_state,
        gold_fiches=gold_fiches,
        gold_spans=gold_spans,
        gold_articles=gold_articles,
        expected_points=(),
        tags=(),
    )


def _fake_dataset_item(item: GoldenItem) -> DatasetItem:
    """A real `DatasetItem`, shaped exactly the way `sync_retrieval_dataset` would have
    written it (`retrieval_dataset_items`), plus the server-assigned bookkeeping fields a
    live dataset item always carries but which `run_retrieval_ladder` never reads."""
    [projected] = retrieval_dataset_items([item])
    now = datetime.now(UTC)
    return DatasetItem(
        id=projected.id,
        status=DatasetStatus.ACTIVE,  # type: ignore[arg-type]  # langfuse's StrEnum resolves to plain `str` under this project's mypy config; correct at runtime
        input=projected.input,
        expected_output=projected.expected_output,
        metadata=projected.metadata,
        dataset_id="ds-1",
        dataset_name=RETRIEVAL_DATASET_NAME,
        created_at=now,
        updated_at=now,
        media_references=[],
    )


class _FakeDataset:
    """Enough of `DatasetClient` for `run_retrieval_ladder`: `updated_at` for the run
    header, `run_experiment` that actually drives `task`/`evaluators` the way the real SDK
    does (`client.py`'s own evaluator-invocation code only ever passes it `input`/`output`/
    `expected_output`/`metadata`, never the raw `DatasetItem` — asserted here by construction
    rather than assumed)."""

    def __init__(self, items: list[DatasetItem], *, updated_at: datetime) -> None:
        self.items = items
        self.updated_at = updated_at
        self.run_experiment_calls: list[dict[str, Any]] = []

    def run_experiment(
        self, *, name: str, run_name: str, task: Any, evaluators: list[Any], max_concurrency: int
    ) -> ExperimentResult:
        self.run_experiment_calls.append(
            {"name": name, "run_name": run_name, "max_concurrency": max_concurrency}
        )
        item_results = []
        for item in self.items:
            output = task(item=item)
            evaluations: list[Evaluation] = []
            for evaluator in evaluators:
                produced = evaluator(
                    input=item.input, output=output, expected_output=item.expected_output, metadata=item.metadata
                )
                evaluations.extend(produced if isinstance(produced, list) else [produced])
            item_results.append(
                ExperimentItemResult(
                    item=item, output=output, evaluations=evaluations, trace_id=None, dataset_run_id=None
                )
            )
        return ExperimentResult(
            name=name,
            run_name=run_name,
            description=None,
            item_results=item_results,
            run_evaluations=[],
            experiment_id="fake-experiment-id",
        )


class _FakeLangfuseClient:
    def __init__(self, dataset: _FakeDataset, **constructor_kwargs: Any) -> None:
        self.dataset = dataset
        self.constructor_kwargs = constructor_kwargs
        self.get_dataset_calls: list[str] = []
        self.flushed = False

    def get_dataset(self, name: str) -> _FakeDataset:
        self.get_dataset_calls.append(name)
        return self.dataset

    def flush(self) -> None:
        self.flushed = True


def _init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)


def _commit_all(root: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=root, check=True, capture_output=True)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


@pytest.fixture
def fake_settings() -> Settings:
    return Settings(
        openrouter_api_key="",
        generation_model="",
        generation_provider="",
        condenser_model="",
        condenser_provider="",
        judge_model="",
        judge_provider="",
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_base_url="https://fake.langfuse.example",
        langfuse_tracing=False,  # deliberately off — proves it's never consulted
        qdrant_url="http://localhost:6333",
    )


class TestRunRetrievalLadder:
    def test_writes_a_run_scored_against_real_retrieval(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qdrant: QdrantClient,
        create_collection: CreateCollection,
        fake_settings: Settings,
    ) -> None:
        create_collection(qdrant, FICHES_ALIAS)
        create_collection(qdrant, ARTICLES_ALIAS)
        qdrant.upsert(FICHES_ALIAS, points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"fiche_id": "F1"})])

        item = golden_item("gs-001", gold_fiches=("F1",))
        dataset_item = _fake_dataset_item(item)
        dataset = _FakeDataset([dataset_item], updated_at=datetime(2026, 9, 1, tzinfo=UTC))

        monkeypatch.setattr(run_experiment_module, "load_settings", lambda: fake_settings)
        monkeypatch.setattr(run_experiment_module, "Langfuse", lambda **kwargs: _FakeLangfuseClient(dataset, **kwargs))

        _init_git_repo(tmp_path)
        golden_set_path = tmp_path / "golden-set.yaml"
        dump_golden_set([item], golden_set_path)
        golden_set_sha = _commit_all(tmp_path, "add golden set")

        runs_dir = tmp_path / "runs"
        run = run_retrieval_ladder(
            client=qdrant,
            embed=stub_embed([1.0, 0.0, 0.0, 0.0]),
            lookup_keys=set(),
            arm="rung1",
            rung="rung1",
            run_id="rung1-test-run",
            repo_root=tmp_path,
            golden_set_path=golden_set_path,
            runs_dir=runs_dir,
            retrieval_config={"embedder_id": "stub"},
        )

        assert run.header.run_id == "rung1-test-run"
        assert run.header.rung == "rung1"
        assert run.header.arm == "rung1"
        assert run.header.langfuse_run_name == "rung1-test-run"
        assert run.header.golden_set_git_sha == golden_set_sha
        assert run.header.langfuse_dataset_version == "2026-09-01T00:00:00+00:00"
        assert run.header.retrieval_config == {"embedder_id": "stub"}

        # A value, not `1.0` specifically: `QdrantClient(":memory:")` is for plumbing
        # assertions only, never a recall/ranking claim (SPEC §6.3, `CODING_STANDARDS.md`) —
        # this only proves `task`/the evaluator wired the retrieved contexts into `score_item`
        # at all, not that the in-memory engine ranked them any particular way.
        [score] = run.items
        assert score.item_id == "gs-001"
        assert score.fiche_recall_at_4 is not None
        assert score.article_recall_at_4 is None  # no gold_articles on this item

        # Actually persisted, not just returned — `load_run` is `write_run`'s own inverse.
        assert load_run(runs_dir / "rung1-test-run.json") == run

    def test_forces_tracing_on_regardless_of_the_dev_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qdrant: QdrantClient,
        create_collection: CreateCollection,
        fake_settings: Settings,
    ) -> None:
        create_collection(qdrant, FICHES_ALIAS)
        create_collection(qdrant, ARTICLES_ALIAS)
        qdrant.upsert(FICHES_ALIAS, points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"fiche_id": "F1"})])

        item = golden_item("gs-001", gold_fiches=("F1",))
        dataset = _FakeDataset([_fake_dataset_item(item)], updated_at=datetime(2026, 9, 1, tzinfo=UTC))
        constructor_calls: list[dict[str, Any]] = []

        def fake_langfuse_constructor(**kwargs: Any) -> _FakeLangfuseClient:
            constructor_calls.append(kwargs)
            return _FakeLangfuseClient(dataset, **kwargs)

        monkeypatch.setattr(run_experiment_module, "load_settings", lambda: fake_settings)
        monkeypatch.setattr(run_experiment_module, "Langfuse", fake_langfuse_constructor)

        _init_git_repo(tmp_path)
        golden_set_path = tmp_path / "golden-set.yaml"
        dump_golden_set([item], golden_set_path)
        _commit_all(tmp_path, "add golden set")

        run_retrieval_ladder(
            client=qdrant,
            embed=stub_embed([1.0, 0.0, 0.0, 0.0]),
            lookup_keys=set(),
            arm="rung1",
            rung="rung1",
            run_id="rung1-test-run",
            repo_root=tmp_path,
            golden_set_path=golden_set_path,
            runs_dir=tmp_path / "runs",
            retrieval_config={},
        )

        assert fake_settings.langfuse_tracing is False  # the dev default this run ignores
        [kwargs] = constructor_calls
        assert kwargs["tracing_enabled"] is True
        assert kwargs["base_url"] == "https://fake.langfuse.example"

    def test_passes_run_id_as_both_name_and_run_name_and_a_concurrency_within_range(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qdrant: QdrantClient,
        create_collection: CreateCollection,
        fake_settings: Settings,
    ) -> None:
        create_collection(qdrant, FICHES_ALIAS)
        create_collection(qdrant, ARTICLES_ALIAS)

        item = golden_item("gs-001", gold_fiches=("F1",))
        dataset = _FakeDataset([_fake_dataset_item(item)], updated_at=datetime(2026, 9, 1, tzinfo=UTC))

        monkeypatch.setattr(run_experiment_module, "load_settings", lambda: fake_settings)
        monkeypatch.setattr(run_experiment_module, "Langfuse", lambda **kwargs: _FakeLangfuseClient(dataset, **kwargs))

        _init_git_repo(tmp_path)
        golden_set_path = tmp_path / "golden-set.yaml"
        dump_golden_set([item], golden_set_path)
        _commit_all(tmp_path, "add golden set")

        run_retrieval_ladder(
            client=qdrant,
            embed=stub_embed([1.0, 0.0, 0.0, 0.0]),
            lookup_keys=set(),
            arm="rung1",
            rung="rung1",
            run_id="rung1-test-run",
            repo_root=tmp_path,
            golden_set_path=golden_set_path,
            runs_dir=tmp_path / "runs",
            retrieval_config={},
        )

        [call] = dataset.run_experiment_calls
        assert call["name"] == "rung1-test-run"
        assert call["run_name"] == "rung1-test-run"
        assert 5 <= call["max_concurrency"] <= MAX_CONCURRENCY

    def test_pipeline_arm_overrides_which_retrieval_arm_runs_while_header_arm_keeps_its_own_label(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qdrant: QdrantClient,
        create_collection: CreateCollection,
        fake_settings: Settings,
    ) -> None:
        """SPEC §12.8 / #38 — the pre-ladder A/Bs run a fixed pipeline (`pipeline_arm`)
        under a header `arm` label ("incumbent"/"challenger") that names which collection
        served the query, not which `RETRIEVAL_ARMS` entry ran."""
        create_collection(qdrant, FICHES_ALIAS)
        create_collection(qdrant, ARTICLES_ALIAS)
        qdrant.upsert(FICHES_ALIAS, points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"fiche_id": "F1"})])

        item = golden_item("gs-001", gold_fiches=("F1",))
        dataset = _FakeDataset([_fake_dataset_item(item)], updated_at=datetime(2026, 9, 1, tzinfo=UTC))

        monkeypatch.setattr(run_experiment_module, "load_settings", lambda: fake_settings)
        monkeypatch.setattr(run_experiment_module, "Langfuse", lambda **kwargs: _FakeLangfuseClient(dataset, **kwargs))

        _init_git_repo(tmp_path)
        golden_set_path = tmp_path / "golden-set.yaml"
        dump_golden_set([item], golden_set_path)
        _commit_all(tmp_path, "add golden set")

        run = run_retrieval_ladder(
            client=qdrant,
            embed=stub_embed([1.0, 0.0, 0.0, 0.0]),
            lookup_keys=set(),
            arm="incumbent",
            rung="ab_fiche_header",
            pipeline_arm="rung1",
            run_id="ab-fiche-header-incumbent",
            repo_root=tmp_path,
            golden_set_path=golden_set_path,
            runs_dir=tmp_path / "runs",
            retrieval_config={},
        )

        # `RETRIEVAL_ARMS` has no "incumbent" entry — had `arm` been passed straight to
        # `retrieve()` this would have raised `KeyError` instead of running rung 1's pipeline.
        assert run.header.arm == "incumbent"
        assert run.header.rung == "ab_fiche_header"
        [score] = run.items
        assert score.fiche_recall_at_4 is not None

    def test_flushes_the_client_before_returning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qdrant: QdrantClient,
        create_collection: CreateCollection,
        fake_settings: Settings,
    ) -> None:
        create_collection(qdrant, FICHES_ALIAS)
        create_collection(qdrant, ARTICLES_ALIAS)

        item = golden_item("gs-001", gold_fiches=("F1",))
        dataset = _FakeDataset([_fake_dataset_item(item)], updated_at=datetime(2026, 9, 1, tzinfo=UTC))
        clients: list[_FakeLangfuseClient] = []

        def fake_langfuse_constructor(**kwargs: Any) -> _FakeLangfuseClient:
            client = _FakeLangfuseClient(dataset, **kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(run_experiment_module, "load_settings", lambda: fake_settings)
        monkeypatch.setattr(run_experiment_module, "Langfuse", fake_langfuse_constructor)

        _init_git_repo(tmp_path)
        golden_set_path = tmp_path / "golden-set.yaml"
        dump_golden_set([item], golden_set_path)
        _commit_all(tmp_path, "add golden set")

        run_retrieval_ladder(
            client=qdrant,
            embed=stub_embed([1.0, 0.0, 0.0, 0.0]),
            lookup_keys=set(),
            arm="rung1",
            rung="rung1",
            run_id="rung1-test-run",
            repo_root=tmp_path,
            golden_set_path=golden_set_path,
            runs_dir=tmp_path / "runs",
            retrieval_config={},
        )

        [client] = clients
        assert client.flushed is True
