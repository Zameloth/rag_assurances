"""`run_generation_eval` — SPEC §12.5/§12.9/§12.11/§12.12, ADR-0009, #46.

Exercised end to end against a hand-written fake `Langfuse` client/dataset — real
`ExperimentItemResult`/`ExperimentResult`/`DatasetItem` objects, but a fake `run_experiment()`
that invokes `task`/the evaluators synchronously instead of going over the network. Same
posture as `test_run_experiment.py`. Retrieval runs for real against
`QdrantClient(":memory:")` (SPEC §6.3 — plumbing only); condensation and generation are both
fakes injected through `condense_fn`/`generate_fn`, so no OpenRouter call happens here either.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from conftest import CreateCollection, raw_point, stub_embed
from langfuse.api import DatasetItem, DatasetStatus
from langfuse.experiment import Evaluation, ExperimentItemResult, ExperimentResult
from qdrant_client import QdrantClient, models

import rag.eval.run_generation_experiment as run_generation_experiment_module
from rag.condensation.pipeline import CondenseFn
from rag.condensation.prompt import Message as CondensationMessage
from rag.condensation.schema import CondenserOutput
from rag.config import Settings
from rag.eval.generation_metrics import ItemGenerationScore
from rag.eval.generation_run import load_generation_run
from rag.eval.judge import (
    FaithfulnessOutput,
    Judge,
    JudgeOutputError,
    PointCoverageOutput,
    PointVerdict,
    PromptLanguage,
)
from rag.eval.langfuse_sync import GENERATION_DATASET_NAME, generation_dataset_items
from rag.eval.run_generation_experiment import (
    MAX_CONCURRENCY,
    JudgeRunError,
    _score_evaluations,
    run_generation_eval,
)
from rag.eval.schema import GoldenItem, dump_golden_set
from rag.generation.pipeline import GenerateFn
from rag.generation.prompt import Message as GenerationMessage
from rag.generation.schema import Envelope, FondementJuridique, Reponse
from rag.ingest.upsert import Embedding
from rag.retrieval.legs import ARTICLES_ALIAS, FICHES_ALIAS


def make_score(**overrides: object) -> ItemGenerationScore:
    defaults: dict[str, object] = dict(
        item_id="gs-001",
        expected_state="reponse",
        actual_state="reponse",
        state_correct=True,
        citation_valid=True,
        fabricated_ids=(),
        citation_correctness=None,
    )
    defaults.update(overrides)
    return ItemGenerationScore(**defaults)  # type: ignore[arg-type]


class TestScoreEvaluations:
    def test_skips_citation_correctness_when_none(self) -> None:
        evaluations = _score_evaluations(make_score(citation_correctness=None))
        names = {e.name for e in evaluations}
        assert names == {"state_correct", "citation_valid"}

    def test_emits_all_three_when_every_metric_applies(self) -> None:
        evaluations = _score_evaluations(make_score(citation_correctness=0.5))
        names = {e.name for e in evaluations}
        assert names == {"state_correct", "citation_valid", "citation_correctness"}

    def test_boolean_metrics_are_tagged_boolean(self) -> None:
        [state_eval] = [e for e in _score_evaluations(make_score()) if e.name == "state_correct"]
        assert state_eval.value is True
        assert state_eval.data_type == "BOOLEAN"

    def test_numeric_metric_keeps_its_float_value_and_data_type(self) -> None:
        [eval_] = [
            e for e in _score_evaluations(make_score(citation_correctness=0.75)) if e.name == "citation_correctness"
        ]
        assert eval_.value == 0.75
        assert eval_.data_type == "NUMERIC"

    def test_never_emits_an_evaluation_for_item_id_or_states(self) -> None:
        names = {e.name for e in _score_evaluations(make_score())}
        assert "item_id" not in names
        assert "expected_state" not in names
        assert "actual_state" not in names
        assert "fabricated_ids" not in names


def golden_item(
    id: str = "gs-001",
    *,
    question: str = "je suis locataire, dois-je m'assurer ?",
    history: tuple[dict[str, str], ...] = (),
    expected_state: str = "reponse",
    gold_articles: tuple[str, ...] = (),
    expected_points: tuple[str, ...] = (),
) -> GoldenItem:
    return GoldenItem(
        id=id,
        question=question,
        history=history,
        expected_state=expected_state,
        gold_fiches=(),
        gold_spans=(),
        gold_articles=gold_articles,
        expected_points=expected_points,
        tags=(),
    )


def _fake_condense_fn(requete: str) -> CondenseFn:
    def fn(messages: list[CondensationMessage]) -> CondenserOutput:
        return CondenserOutput(requete=requete)

    return fn


def _fake_generate_fn(envelope: Envelope) -> GenerateFn:
    def fn(messages: list[GenerationMessage]) -> Envelope:
        return envelope

    return fn


def _fake_dataset_item(item: GoldenItem) -> DatasetItem:
    """A real `DatasetItem`, shaped exactly the way `sync_generation_dataset` would have
    written it (`generation_dataset_items`), plus the server-assigned bookkeeping fields a
    live dataset item always carries but which `run_generation_eval` never reads."""
    [projected] = generation_dataset_items([item])
    now = datetime.now(UTC)
    return DatasetItem(
        id=projected.id,
        status=DatasetStatus.ACTIVE,  # type: ignore[arg-type]  # see test_run_experiment.py's own note
        input=projected.input,
        expected_output=projected.expected_output,
        metadata=projected.metadata,
        dataset_id="ds-1",
        dataset_name=GENERATION_DATASET_NAME,
        created_at=now,
        updated_at=now,
        media_references=[],
    )


class _FakeDataset:
    def __init__(self, items: list[DatasetItem], *, updated_at: datetime) -> None:
        self.items = items
        self.updated_at = updated_at
        self.run_experiment_calls: list[dict[str, Any]] = []
        self.last_evaluations: list[Evaluation] = []

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
            self.last_evaluations = evaluations
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


class TestRunGenerationEval:
    def test_writes_a_run_scored_end_to_end(
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

        item = golden_item("gs-001", expected_state="reponse")
        dataset = _FakeDataset([_fake_dataset_item(item)], updated_at=datetime(2026, 9, 25, tzinfo=UTC))

        monkeypatch.setattr(run_generation_experiment_module, "load_settings", lambda: fake_settings)
        monkeypatch.setattr(
            run_generation_experiment_module, "Langfuse", lambda **kwargs: _FakeLangfuseClient(dataset, **kwargs)
        )

        _init_git_repo(tmp_path)
        golden_set_path = tmp_path / "golden-set.yaml"
        dump_golden_set([item], golden_set_path)
        golden_set_sha = _commit_all(tmp_path, "add golden set")

        envelope = Reponse(explanation="Oui, sous conditions.", fondement_juridique=[])
        runs_dir = tmp_path / "runs"
        run = run_generation_eval(
            client=qdrant,
            embed=stub_embed([1.0, 0.0, 0.0, 0.0]),
            lookup_keys=set(),
            condense_fn=_fake_condense_fn("ignored — no history"),
            generate_fn=_fake_generate_fn(envelope),
            arm="mistral-large-2512",
            run_id="generation-test-run",
            repo_root=tmp_path,
            golden_set_path=golden_set_path,
            runs_dir=runs_dir,
            generation_model="mistralai/mistral-large-2512",
            generation_provider="mistral",
            retrieval_config={"embedder_id": "stub"},
        )

        assert run.header.run_id == "generation-test-run"
        assert run.header.rung == "generation"
        assert run.header.arm == "mistral-large-2512"
        assert run.header.langfuse_run_name == "generation-test-run"
        assert run.header.golden_set_git_sha == golden_set_sha
        assert run.header.langfuse_dataset_version == "2026-09-25T00:00:00+00:00"
        assert run.header.retrieval_config == {"embedder_id": "stub"}
        assert run.header.generation_model == "mistralai/mistral-large-2512"
        assert run.header.generation_provider == "mistral"

        [score] = run.items
        assert score.item_id == "gs-001"
        assert score.expected_state == "reponse"
        assert score.actual_state == "reponse"
        assert score.state_correct is True

        # Actually persisted, not just returned.
        assert load_generation_run(runs_dir / "generation-test-run.json") == run

    def test_persisted_item_id_is_the_bare_golden_id_not_the_langfuse_dataset_item_id(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qdrant: QdrantClient,
        create_collection: CreateCollection,
        fake_settings: Settings,
    ) -> None:
        """#46, confirmed against a live Langfuse project: `generation_dataset_item_id`
        prefixes the golden id (item ids are unique per project, not per dataset, so this
        dataset can't reuse the retrieval dataset's bare-id namespace) — `run.items[].item_id`
        must still come out as the plain `gs-XXX` golden id, read from `metadata["golden_id"]`,
        not from the prefixed `DatasetItem.id`."""
        create_collection(qdrant, FICHES_ALIAS)
        create_collection(qdrant, ARTICLES_ALIAS)

        item = golden_item("gs-042")
        dataset_item = _fake_dataset_item(item)
        assert dataset_item.id != "gs-042"  # the Langfuse-specific id really is prefixed
        dataset = _FakeDataset([dataset_item], updated_at=datetime(2026, 9, 25, tzinfo=UTC))

        monkeypatch.setattr(run_generation_experiment_module, "load_settings", lambda: fake_settings)
        monkeypatch.setattr(
            run_generation_experiment_module, "Langfuse", lambda **kwargs: _FakeLangfuseClient(dataset, **kwargs)
        )

        _init_git_repo(tmp_path)
        golden_set_path = tmp_path / "golden-set.yaml"
        dump_golden_set([item], golden_set_path)
        _commit_all(tmp_path, "add golden set")

        run = run_generation_eval(
            client=qdrant,
            embed=stub_embed([1.0, 0.0, 0.0, 0.0]),
            lookup_keys=set(),
            condense_fn=_fake_condense_fn("ignored"),
            generate_fn=_fake_generate_fn(Reponse(explanation="...", fondement_juridique=[])),
            arm="mistral-large-2512",
            run_id="generation-test-run",
            repo_root=tmp_path,
            golden_set_path=golden_set_path,
            runs_dir=tmp_path / "runs",
            generation_model="mistralai/mistral-large-2512",
            generation_provider="mistral",
            retrieval_config={},
        )

        [score] = run.items
        assert score.item_id == "gs-042"

    def test_condenses_before_retrieving_when_history_is_non_empty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qdrant: QdrantClient,
        create_collection: CreateCollection,
        fake_settings: Settings,
    ) -> None:
        """SPEC §12.1 — the ten `multi_turn` items are the only ones measuring the condenser
        at all, so the task must run `condense()` (which resolves the query `retrieve()`
        actually gets) before retrieval, not skip straight to it."""
        create_collection(qdrant, FICHES_ALIAS)
        create_collection(qdrant, ARTICLES_ALIAS)
        qdrant.upsert(FICHES_ALIAS, points=[raw_point(1, [0.0, 1.0, 0.0, 0.0], {"fiche_id": "F-condensed"})])

        item = golden_item(
            "gs-031",
            question="Et si c'était lui l'ivre ?",
            history=({"role": "user", "content": "Mon assurance me couvre ?"},),
        )
        dataset = _FakeDataset([_fake_dataset_item(item)], updated_at=datetime(2026, 9, 25, tzinfo=UTC))

        monkeypatch.setattr(run_generation_experiment_module, "load_settings", lambda: fake_settings)
        monkeypatch.setattr(
            run_generation_experiment_module, "Langfuse", lambda **kwargs: _FakeLangfuseClient(dataset, **kwargs)
        )

        _init_git_repo(tmp_path)
        golden_set_path = tmp_path / "golden-set.yaml"
        dump_golden_set([item], golden_set_path)
        _commit_all(tmp_path, "add golden set")

        # `retrieve()` embeds whatever text it's handed with the same stub vector regardless
        # of content — the condensed-query dependency is asserted through `embed`'s call log
        # instead, since two different query strings would otherwise be indistinguishable to
        # a stub that ignores its input.
        embedded_texts: list[str] = []

        def embed(texts: Sequence[str]) -> list[Embedding]:
            embedded_texts.extend(texts)
            return [([0.0, 1.0, 0.0, 0.0], models.SparseVector(indices=[], values=[]))] * len(texts)

        run_generation_eval(
            client=qdrant,
            embed=embed,
            lookup_keys=set(),
            condense_fn=_fake_condense_fn("question condensée autonome"),
            generate_fn=_fake_generate_fn(Reponse(explanation="...", fondement_juridique=[])),
            arm="mistral-large-2512",
            run_id="generation-test-run",
            repo_root=tmp_path,
            golden_set_path=golden_set_path,
            runs_dir=tmp_path / "runs",
            generation_model="mistralai/mistral-large-2512",
            generation_provider="mistral",
            retrieval_config={},
        )

        assert "question condensée autonome" in embedded_texts

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

        item = golden_item("gs-001")
        dataset = _FakeDataset([_fake_dataset_item(item)], updated_at=datetime(2026, 9, 25, tzinfo=UTC))
        constructor_calls: list[dict[str, Any]] = []

        def fake_langfuse_constructor(**kwargs: Any) -> _FakeLangfuseClient:
            constructor_calls.append(kwargs)
            return _FakeLangfuseClient(dataset, **kwargs)

        monkeypatch.setattr(run_generation_experiment_module, "load_settings", lambda: fake_settings)
        monkeypatch.setattr(run_generation_experiment_module, "Langfuse", fake_langfuse_constructor)

        _init_git_repo(tmp_path)
        golden_set_path = tmp_path / "golden-set.yaml"
        dump_golden_set([item], golden_set_path)
        _commit_all(tmp_path, "add golden set")

        run_generation_eval(
            client=qdrant,
            embed=stub_embed([1.0, 0.0, 0.0, 0.0]),
            lookup_keys=set(),
            condense_fn=_fake_condense_fn("ignored"),
            generate_fn=_fake_generate_fn(Reponse(explanation="...", fondement_juridique=[])),
            arm="mistral-large-2512",
            run_id="generation-test-run",
            repo_root=tmp_path,
            golden_set_path=golden_set_path,
            runs_dir=tmp_path / "runs",
            generation_model="mistralai/mistral-large-2512",
            generation_provider="mistral",
            retrieval_config={},
        )

        assert fake_settings.langfuse_tracing is False
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

        item = golden_item("gs-001")
        dataset = _FakeDataset([_fake_dataset_item(item)], updated_at=datetime(2026, 9, 25, tzinfo=UTC))

        monkeypatch.setattr(run_generation_experiment_module, "load_settings", lambda: fake_settings)
        monkeypatch.setattr(
            run_generation_experiment_module, "Langfuse", lambda **kwargs: _FakeLangfuseClient(dataset, **kwargs)
        )

        _init_git_repo(tmp_path)
        golden_set_path = tmp_path / "golden-set.yaml"
        dump_golden_set([item], golden_set_path)
        _commit_all(tmp_path, "add golden set")

        run_generation_eval(
            client=qdrant,
            embed=stub_embed([1.0, 0.0, 0.0, 0.0]),
            lookup_keys=set(),
            condense_fn=_fake_condense_fn("ignored"),
            generate_fn=_fake_generate_fn(Reponse(explanation="...", fondement_juridique=[])),
            arm="mistral-large-2512",
            run_id="generation-test-run",
            repo_root=tmp_path,
            golden_set_path=golden_set_path,
            runs_dir=tmp_path / "runs",
            generation_model="mistralai/mistral-large-2512",
            generation_provider="mistral",
            retrieval_config={},
        )

        [call] = dataset.run_experiment_calls
        assert call["name"] == "generation-test-run"
        assert call["run_name"] == "generation-test-run"
        assert 5 <= call["max_concurrency"] <= MAX_CONCURRENCY

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

        item = golden_item("gs-001")
        dataset = _FakeDataset([_fake_dataset_item(item)], updated_at=datetime(2026, 9, 25, tzinfo=UTC))
        clients: list[_FakeLangfuseClient] = []

        def fake_langfuse_constructor(**kwargs: Any) -> _FakeLangfuseClient:
            client = _FakeLangfuseClient(dataset, **kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(run_generation_experiment_module, "load_settings", lambda: fake_settings)
        monkeypatch.setattr(run_generation_experiment_module, "Langfuse", fake_langfuse_constructor)

        _init_git_repo(tmp_path)
        golden_set_path = tmp_path / "golden-set.yaml"
        dump_golden_set([item], golden_set_path)
        _commit_all(tmp_path, "add golden set")

        run_generation_eval(
            client=qdrant,
            embed=stub_embed([1.0, 0.0, 0.0, 0.0]),
            lookup_keys=set(),
            condense_fn=_fake_condense_fn("ignored"),
            generate_fn=_fake_generate_fn(Reponse(explanation="...", fondement_juridique=[])),
            arm="mistral-large-2512",
            run_id="generation-test-run",
            repo_root=tmp_path,
            golden_set_path=golden_set_path,
            runs_dir=tmp_path / "runs",
            generation_model="mistralai/mistral-large-2512",
            generation_provider="mistral",
            retrieval_config={},
        )

        [client] = clients
        assert client.flushed is True

    def test_citation_correctness_reflects_cited_vs_gold_articles(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qdrant: QdrantClient,
        create_collection: CreateCollection,
        fake_settings: Settings,
    ) -> None:
        """SPEC §12.9 — distinct from article recall: this asks whether the model *cited*
        the gold article, not whether it reached the prompt."""
        create_collection(qdrant, FICHES_ALIAS)
        create_collection(qdrant, ARTICLES_ALIAS)
        qdrant.upsert(
            ARTICLES_ALIAS,
            points=[
                raw_point(
                    1,
                    [1.0, 0.0, 0.0, 0.0],
                    {"legiarti_cid": "CID1", "citation_id": "L113-3", "lookup_key": None},
                )
            ],
        )

        item = golden_item("gs-001", gold_articles=("CID1",))
        dataset = _FakeDataset([_fake_dataset_item(item)], updated_at=datetime(2026, 9, 25, tzinfo=UTC))

        monkeypatch.setattr(run_generation_experiment_module, "load_settings", lambda: fake_settings)
        monkeypatch.setattr(
            run_generation_experiment_module, "Langfuse", lambda **kwargs: _FakeLangfuseClient(dataset, **kwargs)
        )

        _init_git_repo(tmp_path)
        golden_set_path = tmp_path / "golden-set.yaml"
        dump_golden_set([item], golden_set_path)
        _commit_all(tmp_path, "add golden set")

        envelope = Reponse(
            explanation="...",
            fondement_juridique=[FondementJuridique(article_id="L113-3", gloss="résiliation")],
        )
        run = run_generation_eval(
            client=qdrant,
            embed=stub_embed([1.0, 0.0, 0.0, 0.0]),
            lookup_keys=set(),
            condense_fn=_fake_condense_fn("ignored"),
            generate_fn=_fake_generate_fn(envelope),
            arm="mistral-large-2512",
            run_id="generation-test-run",
            repo_root=tmp_path,
            golden_set_path=golden_set_path,
            runs_dir=tmp_path / "runs",
            generation_model="mistralai/mistral-large-2512",
            generation_provider="mistral",
            retrieval_config={},
        )

        [score] = run.items
        assert score.citation_valid is True
        assert score.citation_correctness == 1.0


class _ScriptedJudgeCall:
    """Answers faithfulness with a fixed score and point coverage with every point asserted
    except the last — enough to tell the two metrics apart in what gets persisted."""

    def __init__(self, provider: str = "Anthropic") -> None:
        self.provider = provider
        self.prompts: list[str] = []

    def __call__(self, prompt: str, schema: type[Any]) -> tuple[Any, str]:
        self.prompts.append(prompt)
        if schema is FaithfulnessOutput:
            return FaithfulnessOutput(score=0.8, reasoning="4 of 5 grounded"), self.provider
        points = [line for line in prompt.splitlines() if line[:1].isdigit() and ". " in line]
        verdicts = [
            PointVerdict(index=i, asserted=i < len(points), reasoning="r") for i in range(1, len(points) + 1)
        ]
        return PointCoverageOutput(verdicts=verdicts), self.provider


class _FailingOnFaithfulness(_ScriptedJudgeCall):
    """Point coverage succeeds; faithfulness raises the way a response with no resolved
    provider does (`rag.eval.judge_chain`)."""

    def __call__(self, prompt: str, schema: type[Any]) -> tuple[Any, str]:
        if schema is FaithfulnessOutput:
            raise JudgeOutputError("OpenRouter reported no resolved provider for the judge call")
        return super().__call__(prompt, schema)


class TestJudgedEvaluators:
    """#47 — faithfulness and point coverage, wired as evaluators over the same task output
    as the three deterministic metrics, persisted per item with the resolved provider."""

    def _run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qdrant: QdrantClient,
        create_collection: CreateCollection,
        fake_settings: Settings,
        items: list[GoldenItem],
        judge: Judge | None,
    ) -> tuple[Any, _FakeDataset]:
        create_collection(qdrant, FICHES_ALIAS)
        create_collection(qdrant, ARTICLES_ALIAS)
        qdrant.upsert(FICHES_ALIAS, points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"fiche_id": "F1", "text": "Texte de fiche."})])
        dataset = _FakeDataset([_fake_dataset_item(i) for i in items], updated_at=datetime(2026, 9, 25, tzinfo=UTC))
        monkeypatch.setattr(run_generation_experiment_module, "load_settings", lambda: fake_settings)
        monkeypatch.setattr(
            run_generation_experiment_module, "Langfuse", lambda **kwargs: _FakeLangfuseClient(dataset, **kwargs)
        )
        _init_git_repo(tmp_path)
        golden_set_path = tmp_path / "golden-set.yaml"
        dump_golden_set(items, golden_set_path)
        _commit_all(tmp_path, "add golden set")

        run = run_generation_eval(
            client=qdrant,
            embed=stub_embed([1.0, 0.0, 0.0, 0.0]),
            lookup_keys=set(),
            condense_fn=_fake_condense_fn("ignored"),
            generate_fn=_fake_generate_fn(Reponse(explanation="Oui.", fondement_juridique=[])),
            arm="mistral-large-2512",
            run_id="generation-test-run",
            repo_root=tmp_path,
            golden_set_path=golden_set_path,
            runs_dir=tmp_path / "runs",
            generation_model="mistralai/mistral-large-2512",
            generation_provider="mistral",
            retrieval_config={},
            judge=judge,
        )
        return run, dataset

    def _judge(self, call: _ScriptedJudgeCall) -> Judge:
        return Judge(
            call=call,
            model="anthropic/claude-sonnet-5",
            faithfulness_language=PromptLanguage.EN,
            point_coverage_language=PromptLanguage.FR,
        )

    def test_persists_both_judged_scores_per_item(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qdrant: QdrantClient,
        create_collection: CreateCollection,
        fake_settings: Settings,
    ) -> None:
        item = golden_item("gs-001", expected_points=("p1", "p2"))
        run, _ = self._run(
            tmp_path, monkeypatch, qdrant, create_collection, fake_settings, [item], self._judge(_ScriptedJudgeCall())
        )

        [score] = run.items
        assert score.faithfulness == 0.8
        assert score.point_coverage == 0.5
        assert load_generation_run(tmp_path / "runs" / "generation-test-run.json") == run

    def test_judge_sees_the_context_the_generator_saw(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qdrant: QdrantClient,
        create_collection: CreateCollection,
        fake_settings: Settings,
    ) -> None:
        call = _ScriptedJudgeCall()
        self._run(tmp_path, monkeypatch, qdrant, create_collection, fake_settings, [golden_item()], self._judge(call))

        [faithfulness_prompt] = call.prompts
        assert "Texte de fiche." in faithfulness_prompt
        assert "Oui." in faithfulness_prompt

    def test_header_pins_judge_model_resolved_provider_and_prompt_languages(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qdrant: QdrantClient,
        create_collection: CreateCollection,
        fake_settings: Settings,
    ) -> None:
        run, _ = self._run(
            tmp_path,
            monkeypatch,
            qdrant,
            create_collection,
            fake_settings,
            [golden_item()],
            self._judge(_ScriptedJudgeCall(provider="Google")),
        )

        assert run.header.judge_model == "anthropic/claude-sonnet-5"
        assert run.header.judge_providers == ("Google",)
        assert run.header.judge_prompt_languages == {"faithfulness": "en", "point_coverage": "fr"}

    def test_item_without_expected_points_has_no_coverage_score(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qdrant: QdrantClient,
        create_collection: CreateCollection,
        fake_settings: Settings,
    ) -> None:
        run, dataset = self._run(
            tmp_path, monkeypatch, qdrant, create_collection, fake_settings, [golden_item()], self._judge(_ScriptedJudgeCall())
        )

        [score] = run.items
        assert score.point_coverage is None
        assert score.faithfulness == 0.8

    def test_scores_reach_langfuse_with_reasoning_and_provider(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qdrant: QdrantClient,
        create_collection: CreateCollection,
        fake_settings: Settings,
    ) -> None:
        run, dataset = self._run(
            tmp_path,
            monkeypatch,
            qdrant,
            create_collection,
            fake_settings,
            [golden_item(expected_points=("p1",))],
            self._judge(_ScriptedJudgeCall()),
        )

        [faithfulness] = [e for e in dataset.last_evaluations if e.name == "faithfulness"]
        assert faithfulness.comment == "4 of 5 grounded"
        assert faithfulness.metadata == {"provider": "Anthropic"}
        assert faithfulness.data_type == "NUMERIC"
        assert {e.name for e in dataset.last_evaluations} >= {"faithfulness", "point_coverage"}

    def test_without_a_judge_the_run_stays_deterministic_only(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qdrant: QdrantClient,
        create_collection: CreateCollection,
        fake_settings: Settings,
    ) -> None:
        run, _ = self._run(tmp_path, monkeypatch, qdrant, create_collection, fake_settings, [golden_item()], None)

        [score] = run.items
        assert score.faithfulness is None
        assert run.header.judge_model == ""
        assert run.header.judge_providers == ()

    def test_a_judge_failure_is_recorded_per_item_and_fails_the_run_after_persisting_it(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qdrant: QdrantClient,
        create_collection: CreateCollection,
        fake_settings: Settings,
    ) -> None:
        """`dataset.run_experiment()` swallows evaluator exceptions — a judge failure must
        not reach `eval/runs/` looking like "no judge ran" or "nothing to cover"."""
        item = golden_item("gs-007", expected_points=("p1", "p2"))

        with pytest.raises(JudgeRunError, match="gs-007"):
            self._run(
                tmp_path,
                monkeypatch,
                qdrant,
                create_collection,
                fake_settings,
                [item],
                self._judge(_FailingOnFaithfulness()),
            )

        [score] = load_generation_run(tmp_path / "runs" / "generation-test-run.json").items
        assert score.faithfulness is None
        assert score.judge_error is not None
        assert "faithfulness" in score.judge_error
        assert "no resolved provider" in score.judge_error
        assert score.point_coverage == 0.5  # the metric that succeeded is kept

    def test_a_clean_judged_run_records_no_judge_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qdrant: QdrantClient,
        create_collection: CreateCollection,
        fake_settings: Settings,
    ) -> None:
        run, _ = self._run(
            tmp_path, monkeypatch, qdrant, create_collection, fake_settings, [golden_item()], self._judge(_ScriptedJudgeCall())
        )

        [score] = run.items
        assert score.judge_error is None
