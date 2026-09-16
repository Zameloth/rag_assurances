"""`RunHeader`/`RetrievalRun` persistence and the git-sha helpers pinning them — SPEC §12.11,
ADR-0010, #35."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from rag.eval.retrieval_metrics import ItemRetrievalScore
from rag.eval.retrieval_run import RetrievalRun, RunHeader, resolve_git_sha, write_run


def make_header(**overrides: object) -> RunHeader:
    defaults: dict[str, object] = dict(
        run_id="rung1-2026-09-16",
        rung="rung1",
        arm="rung1",
        golden_set_git_sha="a" * 40,
        langfuse_dataset_version="2026-09-16T00:00:00Z",
        retrieval_config={"embedder_id": "BAAI/bge-m3", "leg_top_k": 20},
        code_git_sha="b" * 40,
        timestamp="2026-09-16T12:00:00Z",
        langfuse_run_name="rung1-baseline",
    )
    defaults.update(overrides)
    return RunHeader(**defaults)  # type: ignore[arg-type]


def make_item(item_id: str = "gs-001") -> ItemRetrievalScore:
    return ItemRetrievalScore(
        item_id=item_id,
        short_circuit_path="no_reference",
        fiche_recall_at_4=1.0,
        fiche_recall_at_10=1.0,
        fiche_recall_at_candidate=1.0,
        article_recall_at_4=None,
        article_recall_at_10=None,
        article_recall_at_candidate=None,
        zero_articles=None,
        floor_correct=None,
        span_containment_at_4=None,
    )


def init_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)


def commit_all(root: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=root, check=True, capture_output=True)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


class TestWriteRun:
    def test_writes_to_runs_dir_slash_run_id_dot_json(self, tmp_path: Path) -> None:
        run = RetrievalRun(header=make_header(), items=(make_item(),))
        path = write_run(run, tmp_path)
        assert path == tmp_path / "rung1-2026-09-16.json"
        assert path.exists()

    def test_round_trips_through_load_run(self, tmp_path: Path) -> None:
        from rag.eval.retrieval_run import load_run

        run = RetrievalRun(header=make_header(), items=(make_item("gs-001"), make_item("gs-002")))
        path = write_run(run, tmp_path)
        reloaded = load_run(path)
        assert reloaded == run

    def test_is_valid_json_with_the_header_and_items_keys(self, tmp_path: Path) -> None:
        run = RetrievalRun(header=make_header(), items=(make_item(),))
        path = write_run(run, tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert set(payload) == {"header", "items"}
        assert payload["header"]["run_id"] == "rung1-2026-09-16"
        assert payload["items"][0]["item_id"] == "gs-001"

    def test_creates_the_runs_dir_if_missing(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        run = RetrievalRun(header=make_header(), items=())
        write_run(run, runs_dir)
        assert (runs_dir / "rung1-2026-09-16.json").exists()


class TestResolveGitSha:
    def test_code_sha_is_the_repo_head(self, tmp_path: Path) -> None:
        init_repo(tmp_path)
        (tmp_path / "a.txt").write_text("one", encoding="utf-8")
        sha = commit_all(tmp_path, "first")
        assert resolve_git_sha(tmp_path) == sha

    def test_code_sha_moves_on_any_commit(self, tmp_path: Path) -> None:
        init_repo(tmp_path)
        (tmp_path / "a.txt").write_text("one", encoding="utf-8")
        commit_all(tmp_path, "first")
        (tmp_path / "b.txt").write_text("two", encoding="utf-8")
        second = commit_all(tmp_path, "second")
        assert resolve_git_sha(tmp_path) == second

    def test_path_scoped_sha_only_moves_when_that_path_changes(self, tmp_path: Path) -> None:
        """`golden_set_git_sha` (ADR-0010) must not move on a code-only commit — a
        code-scoped and a label-scoped sha have to be able to diverge."""
        init_repo(tmp_path)
        golden = tmp_path / "golden-set.yaml"
        golden.write_text("- id: gs-001", encoding="utf-8")
        golden_sha = commit_all(tmp_path, "add golden set")

        (tmp_path / "unrelated.py").write_text("x = 1", encoding="utf-8")
        code_sha = commit_all(tmp_path, "unrelated code change")

        assert code_sha != golden_sha
        assert resolve_git_sha(tmp_path) == code_sha
        assert resolve_git_sha(tmp_path, path=golden) == golden_sha

    def test_raises_when_the_path_has_no_commits(self, tmp_path: Path) -> None:
        init_repo(tmp_path)
        (tmp_path / "a.txt").write_text("one", encoding="utf-8")
        commit_all(tmp_path, "first")
        with pytest.raises(ValueError):
            resolve_git_sha(tmp_path, path=tmp_path / "never-committed.yaml")
