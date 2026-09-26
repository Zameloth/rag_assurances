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
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rag.config import PromptLanguage
from rag.eval.judge import JudgedMetric
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
    and `judge_providers` default empty — a pure retrieval run has no judge in the loop at
    all (ADR-0011: "the ladder is fully deterministic and API-free"), and neither does a
    generation run made without one (#46's). #47's judged runs fill them, `judge_providers`
    with every provider OpenRouter *resolved* across the run's judge calls — read off each
    response (`rag.eval.judge_chain`), not the one requested; one in practice, since
    `allow_fallbacks: false` leaves nowhere else to route — plus `judge_prompt_languages`,
    the per-metric prompt language (SPEC §12.10's FR/EN question), since the same judge
    model behind a different prompt is a different instrument.

    `generation_model`/`generation_provider` are the #46 counterpart for the generation arm
    itself: "the generation model is an ablatable arm" (SPEC §10.1) is exactly the fact a
    retrieval run has nothing to pin (it never calls a generation model), so those two
    default empty there and are filled by `rag.eval.run_generation_experiment`.

    `generation_provider` pins the *requested* `provider.order[0]` (SPEC §10.1's pinned
    routing), not a value read back from the response — see
    `rag.eval.run_generation_experiment.run_generation_eval`'s own docstring for why: the
    installed `langchain_openai` never surfaces OpenRouter's own `provider` response field
    through `with_structured_output()`, so an independently-resolved value isn't available
    to pin here yet. `allow_fallbacks: false` is what makes the requested value trustworthy
    for a *successful* call — OpenRouter has nowhere else to have routed it.
    """

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
    judge_providers: tuple[str, ...] = ()
    generation_model: str = ""
    generation_provider: str = ""
    judge_prompt_languages: dict[JudgedMetric, PromptLanguage] = field(default_factory=dict)

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> RunHeader:
        """The inverse of `asdict` + JSON: tuples and enums back from lists and strings.

        Runs written before #47 carry a single `judge_provider` string — always empty, since
        no judge had run yet — instead of `judge_providers`; it is read as one provider when
        non-empty and as none otherwise, so every committed run in `eval/runs/` still loads.
        """
        fields = dict(raw)
        legacy_provider = fields.pop("judge_provider", None)
        providers = fields.pop("judge_providers", [legacy_provider] if legacy_provider else [])
        languages = fields.pop("judge_prompt_languages", {})
        return cls(
            **fields,
            judge_providers=tuple(providers),
            judge_prompt_languages={
                JudgedMetric(metric): PromptLanguage(language) for metric, language in languages.items()
            },
        )


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
    header = RunHeader.from_json(raw["header"])
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
