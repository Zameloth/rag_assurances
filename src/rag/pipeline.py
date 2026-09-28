"""The full chain for one turn — condense, retrieve, generate (CONTEXT.md: "the pipeline is a
library", #46, #50).

The one code path the generation eval (`rag.eval.run_generation_experiment`), the
calibration authoring helper (#47) and the web app (`rag.app`) all run. It lives here, above
both `rag.eval` and `rag.app`, so the served system can never become a second code path
next to the measured one.

It also announces its own **stage events** (SPEC §13.2, #51) through `on_stage`, so the
app's progress indicator shows the pipeline that actually ran rather than a timed animation.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet

from qdrant_client import QdrantClient

from rag.condensation.pipeline import CondenseFn, condense
from rag.condensation.prompt import HistoryTurn, trim_history
from rag.condensation.prompt import Message as CondensationMessage
from rag.condensation.schema import CondenserOutput
from rag.generation.pipeline import GenerateFn, GenerationResult, generate
from rag.generation.prompt import HistoryTurn as GenerationHistoryTurn
from rag.ingest.upsert import EmbedFn
from rag.retrieval.pipeline import retrieve

__all__ = ["GENERATION_RETRIEVAL_ARM", "Stage", "StageFn", "ignore_stage", "run_chain"]

# ADR-0024 — rung 1 stands. What a generation run sits on top of, what the calibration
# authoring helper's real answers come from, and what the app serves: one constant, so the
# answers calibrated against and the answers people are shown are answers the generation
# eval would actually score.
GENERATION_RETRIEVAL_ARM = "rung1"


class Stage(enum.Enum):
    """SPEC §13.2's staged indicator, one member per wait a visitor can sit through.
    `CHARGEMENT` is the first question reading the lookup keys (BGE-M3 loads at app
    startup) and is announced by the app, which owns that read; the other three by
    `run_chain` as it reaches them."""

    CHARGEMENT = "chargement"
    CONDENSATION = "condensation"
    RECHERCHE = "recherche"
    GENERATION = "generation"


StageFn = Callable[[Stage], None]


def ignore_stage(stage: Stage) -> None:
    """The `StageFn` for a caller with no one to show progress to — the eval, say."""


def run_chain(
    raw_turn: str,
    history: Sequence[HistoryTurn],
    *,
    client: QdrantClient,
    embed: EmbedFn,
    lookup_keys: AbstractSet[str],
    condense_fn: CondenseFn,
    generate_fn: GenerateFn,
    retrieval_arm: str = GENERATION_RETRIEVAL_ARM,
    on_stage: StageFn = ignore_stage,
) -> GenerationResult:
    """Condense, retrieve, generate — exactly as the eval task runs it.

    `history` is trimmed once, here, for both stages (SPEC §8.7): it is untrusted client
    input on the app path, and `condense()`'s own trim only bounds what the condenser sees,
    not the generation prompt. The golden set's scripted histories sit inside the window,
    so on the eval path this is a no-op.

    `on_stage` hears each stage as it starts. `CONDENSATION` is announced from inside the
    condenser call itself, so a turn that skips condensation (no history, or a resolved
    short-circuit — ADR-0008) never claims to have condensed.
    """
    history = trim_history(history)

    def announced_condense_fn(messages: list[CondensationMessage]) -> CondenserOutput:
        on_stage(Stage.CONDENSATION)
        return condense_fn(messages)

    condensation = condense(raw_turn, history, lookup_keys, announced_condense_fn)
    on_stage(Stage.RECHERCHE)
    retrieval = retrieve(client, embed, condensation.query, lookup_keys, arm=retrieval_arm)
    generation_history = tuple(
        GenerationHistoryTurn(role=turn.role, content=turn.content) for turn in history
    )
    on_stage(Stage.GENERATION)
    return generate(raw_turn, retrieval, generate_fn, history=generation_history)
