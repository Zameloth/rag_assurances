"""`(raw_turn, history)` -> `CondensationResult`: the two skips, the sanitizer, the
code-computed `condense_status` (SPEC §8.1, §8.8, ADR-0008, #43).

`condense_fn` is the seam a real `with_structured_output()` LangChain chain plugs into —
the same shape `GenerateFn`/`RerankFn`/`EmbedFn` already take
(`rag.generation.pipeline`, `rag.retrieval.rerank`, `rag.ingest.upsert`). Tests inject a
fake; `rag.condensation.chain.make_condense_fn` builds the real one.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from rag.condensation.prompt import HistoryTurn, Message, build_messages, trim_history
from rag.condensation.sanitizer import sanitize_condensed
from rag.condensation.schema import CondenserOutput
from rag.retrieval.short_circuit import ShortCircuitPath, resolve_short_circuit

__all__ = ["CondenseFn", "CondenseStatus", "CondensationResult", "condense"]

# `Message` in, a `CondenserOutput` out — the real implementation runs `build_messages`'s
# output through a `with_structured_output` chain pinned to `CONDENSER_MODEL` (SPEC §8.2).
CondenseFn = Callable[[list[Message]], CondenserOutput]


class CondenseStatus(enum.Enum):
    """SPEC §8.8's six paths — computed here by code, never by the model. Fallback and
    passthrough rates are derived views over this field, never a second source of truth
    (`CODING_STANDARDS.md`: "a typed field beats an inference")."""

    SKIPPED_NO_HISTORY = "skipped_no_history"
    SKIPPED_SHORT_CIRCUIT = "skipped_short_circuit"
    PASSTHROUGH = "passthrough"
    REWRITTEN = "rewritten"
    FALLBACK_SANITIZER = "fallback_sanitizer"
    FALLBACK_ERROR = "fallback_error"


@dataclass(frozen=True)
class CondensationResult:
    """`query` is what the retriever should actually be called with — `raw_turn` on every
    path except `REWRITTEN`. `condensed_query` is `None` only when no condensation call
    was made at all (both skip paths, and `FALLBACK_ERROR` where the call produced no
    output); on `FALLBACK_SANITIZER` it still carries the model's rejected text, since a
    call *was* made and the rejected text is exactly what SPEC §8.8's "undiagnosable
    months later" observability concern is about."""

    query: str
    condensed_query: str | None
    condense_status: CondenseStatus


def condense(
    raw_turn: str,
    history: Sequence[HistoryTurn],
    lookup_keys: AbstractSet[str],
    condense_fn: CondenseFn,
) -> CondensationResult:
    """SPEC §8.1's two skips, then the condenser call and its sanitizer.

    `history` is trimmed first (`rag.condensation.prompt.trim_history`, SPEC §8.7's
    "server-side enforcement is not optional") — before either skip check, so a client
    posting 200 turns still only ever costs this function a slice, never a 200-message
    call. Trimming a non-empty `history` never produces an empty one, so trimming first
    doesn't change which turns take the `SKIPPED_NO_HISTORY` branch below.

    `lookup_keys` is the same collection-loaded membership set
    `rag.retrieval.short_circuit.resolve_short_circuit` already takes
    (`rag.retrieval.lookup.load_lookup_keys`) — this function re-runs that same check on
    `raw_turn` for the sole purpose of deciding whether to skip condensation (SPEC §8.1's
    second skip); it is not a second short-circuit, `rag.retrieval.pipeline.retrieve` still
    runs its own.
    """
    history = trim_history(history)
    if not history:
        return CondensationResult(
            query=raw_turn, condensed_query=None, condense_status=CondenseStatus.SKIPPED_NO_HISTORY
        )

    short_circuit = resolve_short_circuit(raw_turn, lookup_keys)
    if short_circuit.path is ShortCircuitPath.RESOLVED:
        return CondensationResult(
            query=raw_turn,
            condensed_query=None,
            condense_status=CondenseStatus.SKIPPED_SHORT_CIRCUIT,
        )

    messages = build_messages(raw_turn, history)
    try:
        output = condense_fn(messages)
    except Exception:
        # SPEC §8.3: "call error, timeout, pinned endpoint down" are all a caller-side
        # exception from the LangChain chain, of no fixed type; every one of them falls
        # back to the raw turn exactly like a sanitizer rejection.
        return CondensationResult(
            query=raw_turn, condensed_query=None, condense_status=CondenseStatus.FALLBACK_ERROR
        )

    sanitized = sanitize_condensed(raw_turn, output.requete)
    if sanitized is None:
        return CondensationResult(
            query=raw_turn,
            condensed_query=output.requete,
            condense_status=CondenseStatus.FALLBACK_SANITIZER,
        )

    if sanitized == raw_turn.strip():
        return CondensationResult(
            query=raw_turn, condensed_query=sanitized, condense_status=CondenseStatus.PASSTHROUGH
        )

    return CondensationResult(
        query=sanitized, condensed_query=sanitized, condense_status=CondenseStatus.REWRITTEN
    )
