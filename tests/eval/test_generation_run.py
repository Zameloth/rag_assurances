"""`GenerationRun` persistence — SPEC §12.9, §12.11, #46. Reuses `RunHeader`/`resolve_git_sha`
from `rag.eval.retrieval_run`, tested there — this file covers only what's new: the
generation-shaped items and `generation_model`/`generation_provider` on the header."""

from __future__ import annotations

import json
from pathlib import Path

from rag.eval.generation_metrics import ItemGenerationScore
from rag.eval.generation_run import GenerationRun, load_generation_run, write_generation_run
from rag.eval.retrieval_run import RunHeader


def make_header(**overrides: object) -> RunHeader:
    defaults: dict[str, object] = dict(
        run_id="generation-2026-09-25",
        rung="generation",
        arm="mistral-large-2512",
        golden_set_git_sha="a" * 40,
        langfuse_dataset_version="2026-09-25T00:00:00Z",
        retrieval_config={"embedder": "BAAI/bge-m3", "retrieval_arm": "rung1"},
        code_git_sha="b" * 40,
        timestamp="2026-09-25T12:00:00Z",
        langfuse_run_name="generation-2026-09-25",
        generation_model="mistralai/mistral-large-2512",
        generation_provider="mistral",
    )
    defaults.update(overrides)
    return RunHeader(**defaults)  # type: ignore[arg-type]


def make_item(item_id: str = "gs-001") -> ItemGenerationScore:
    return ItemGenerationScore(
        item_id=item_id,
        expected_state="reponse",
        actual_state="reponse",
        state_correct=True,
        citation_valid=True,
        fabricated_ids=(),
        citation_correctness=1.0,
    )


class TestWriteGenerationRun:
    def test_writes_to_runs_dir_slash_run_id_dot_json(self, tmp_path: Path) -> None:
        run = GenerationRun(header=make_header(), items=(make_item(),))
        path = write_generation_run(run, tmp_path)
        assert path == tmp_path / "generation-2026-09-25.json"
        assert path.exists()

    def test_round_trips_through_load_generation_run(self, tmp_path: Path) -> None:
        run = GenerationRun(header=make_header(), items=(make_item("gs-001"), make_item("gs-002")))
        path = write_generation_run(run, tmp_path)
        reloaded = load_generation_run(path)
        assert reloaded == run

    def test_header_pins_the_generation_model_and_resolved_provider(self, tmp_path: Path) -> None:
        run = GenerationRun(header=make_header(), items=())
        path = write_generation_run(run, tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["header"]["generation_model"] == "mistralai/mistral-large-2512"
        assert payload["header"]["generation_provider"] == "mistral"

    def test_creates_the_runs_dir_if_missing(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        run = GenerationRun(header=make_header(), items=())
        write_generation_run(run, runs_dir)
        assert (runs_dir / "generation-2026-09-25.json").exists()


class TestJudgedFields:
    """#47 — the two judged metrics and the judge's pinned configuration."""

    def test_judged_scores_round_trip(self, tmp_path: Path) -> None:
        from dataclasses import replace

        item = replace(make_item(), faithfulness=0.75, point_coverage=None)
        header = make_header(
            judge_model="anthropic/claude-sonnet-5",
            judge_provider="Anthropic",
            judge_prompt_languages={"faithfulness": "en", "point_coverage": "fr"},
        )
        run = GenerationRun(header=header, items=(item,))

        assert load_generation_run(write_generation_run(run, tmp_path)) == run

    def test_a_run_written_before_the_judge_existed_still_loads(self, tmp_path: Path) -> None:
        """`eval/runs/generation-*.json` from #46 carries neither judged field."""
        run = GenerationRun(header=make_header(), items=(make_item(),))
        path = write_generation_run(run, tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload["items"]:
            del item["faithfulness"], item["point_coverage"]
        del payload["header"]["judge_prompt_languages"]
        path.write_text(json.dumps(payload), encoding="utf-8")

        [reloaded] = load_generation_run(path).items
        assert reloaded.faithfulness is None
        assert reloaded.point_coverage is None
