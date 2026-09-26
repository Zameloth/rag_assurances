"""Per-run persistence for the generation eval harness — SPEC §12.9, §12.11, #46.

Mirrors `rag.eval.retrieval_run`'s own split: `GenerationRun` pairs the shared `RunHeader`
(SPEC §12.11's header pins the same facts regardless of regime) with the per-item rows
`rag.eval.generation_metrics.score_item` produces. Reuses `RunHeader`/`resolve_git_sha`
verbatim rather than inventing a second header shape — the two regimes differ in what gets
scored, not in what has to be pinned to reproduce a run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rag.eval.generation_metrics import ItemGenerationScore
from rag.eval.retrieval_run import RunHeader

__all__ = ["GenerationRun", "load_generation_run", "write_generation_run"]


@dataclass(frozen=True)
class GenerationRun:
    header: RunHeader
    items: tuple[ItemGenerationScore, ...]


def write_generation_run(run: GenerationRun, runs_dir: Path) -> Path:
    """Write `<runs_dir>/<run.header.run_id>.json` (SPEC §12.11's `eval/runs/<run-id>.json`),
    creating `runs_dir` if needed, and return the path written."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run.header.run_id}.json"
    path.write_text(json.dumps(asdict(run), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_generation_run(path: Path) -> GenerationRun:
    """The inverse of `write_generation_run` — reconstructs the dataclasses rather than
    handing back bare JSON dicts, the same posture `rag.eval.retrieval_run.load_run` takes.

    `fabricated_ids` needs an explicit `tuple(...)`, unlike every other field here: JSON has
    no tuple type, so `json.loads` hands it back as a list, and `ItemGenerationScore` (unlike
    `ItemRetrievalScore`, which is bools/floats/strings only) has this one field that would
    otherwise fail the round trip on type alone.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    header = RunHeader(**raw["header"])
    items = tuple(
        ItemGenerationScore(**{**item, "fabricated_ids": tuple(item["fabricated_ids"])})
        for item in raw["items"]
    )
    return GenerationRun(header=header, items=items)
