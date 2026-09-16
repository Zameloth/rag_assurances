"""Per-run persistence for the retrieval ladder — SPEC §12.11, ADR-0010, ADR-0011, #35.

`RunHeader` pins everything SPEC §12.11 says could move a score; `RetrievalRun` pairs it
with the per-item rows `rag.eval.retrieval_metrics.score_item` produces. Both are plain,
JSON-shaped dataclasses on purpose — `eval/runs/<run-id>.json` is machine-written and never
hand-edited (SPEC §12.11: "JSON, not YAML" is exactly this call, made once for the whole
project), so there is no human-readability requirement pulling toward YAML's block style
the way `rag.eval.schema.dump_golden_set` has.

This module has nothing to do with *how* a run is produced — no Qdrant, no Langfuse, no
`dataset.run_experiment()`. That orchestration is the harness script's job; this is only the
shape it writes into and the two git-sha lookups (SPEC §12.11's "golden-set git sha" and
"code git sha") it needs to fill the header with.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rag.eval.retrieval_metrics import ItemRetrievalScore

__all__ = [
    "RetrievalRun",
    "RunHeader",
    "load_run",
    "resolve_git_sha",
    "write_run",
]


@dataclass(frozen=True)
class RunHeader:
    """SPEC §12.11: "the run header pins everything that could move a score." `judge_model`
    and `judge_provider` default empty — a pure retrieval run has no judge in the loop at
    all (ADR-0011: "the ladder is fully deterministic and API-free"); they exist on this
    header only so the generation-eval harness (#46) can reuse the same shape rather than
    inventing a second one for a single pair of fields."""

    run_id: str
    rung: str
    arm: str
    golden_set_git_sha: str
    langfuse_dataset_version: str
    retrieval_config: dict[str, Any]
    code_git_sha: str
    timestamp: str
    langfuse_run_name: str
    judge_model: str = ""
    judge_provider: str = ""


@dataclass(frozen=True)
class RetrievalRun:
    header: RunHeader
    items: tuple[ItemRetrievalScore, ...]


def write_run(run: RetrievalRun, runs_dir: Path) -> Path:
    """Write `<runs_dir>/<run.header.run_id>.json` (SPEC §12.11's `eval/runs/<run-id>.json`),
    creating `runs_dir` if needed, and return the path written."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run.header.run_id}.json"
    path.write_text(json.dumps(asdict(run), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_run(path: Path) -> RetrievalRun:
    """The inverse of `write_run` — reconstructs the dataclasses `compare.py` (#36) reads
    per-item scores through, rather than handing back bare JSON dicts."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    header = RunHeader(**raw["header"])
    items = tuple(ItemRetrievalScore(**item) for item in raw["items"])
    return RetrievalRun(header=header, items=items)


def resolve_git_sha(repo_root: Path, *, path: Path | None = None) -> str:
    """The sha of the most recent commit touching `path`, or of `HEAD` when `path` is
    omitted. The header needs both flavours: `code_git_sha` moves on any commit,
    `golden_set_git_sha` (ADR-0010) must move only when the labels themselves change — a
    plain `HEAD` for both would let an unrelated code commit spuriously invalidate "did the
    labels change?".

    Raises `ValueError` if `path` has never been committed — a silent empty sha in the run
    header would be indistinguishable from a real, if unlikely, all-zero commit id.
    """
    args = ["git", "-C", str(repo_root), "log", "-1", "--format=%H"]
    if path is not None:
        args.extend(["--", str(path)])
    completed = subprocess.run(args, capture_output=True, text=True, check=True)
    sha = completed.stdout.strip()
    if not sha:
        raise ValueError(f"no commit touches {path if path is not None else repo_root}")
    return sha
