"""The faulted-twin authoring helper's library half (SPEC §12.10, #47).

`scripts/author_calibration_pair.py` is the interactive part — run the real chain for one
golden item, show the answer and the four archetypes, open `$EDITOR` on the twin. This
module is everything that doesn't need a terminal: drafting the pair from a real
`GenerationResult`, the envelope's editable YAML form, and the validated write.

**The clean side is always a real pipeline answer** (SPEC §12.10: "keep the real pipeline
answer and hand-author a faulted twin") — `draft_pair` takes a `GenerationResult`, never a
hand-written envelope, and records the context section and citation ids that answer was
generated from, so the pair stays replayable without Qdrant.

**Nothing invalid reaches the file.** `upsert_pair` validates the whole resulting set
against the golden set before writing, so the helper can't leave `judge-set.yaml` in a
state `rag.eval.calibration.load_calibration_set`/`validate_calibration_set` would reject.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml
from pydantic import ValidationError

from rag.eval.calibration import (
    ENVELOPE_ADAPTER,
    CalibrationAnswer,
    CalibrationPair,
    CalibrationSetError,
    FaultArchetype,
    HumanLabel,
    block_yaml,
    dump_calibration_set,
    load_calibration_set,
    validate_calibration_set,
)
from rag.eval.schema import GoldenItem
from rag.generation.citation import retrieved_citation_ids
from rag.generation.pipeline import GenerationResult
from rag.generation.prompt import build_context_section
from rag.generation.schema import Envelope

__all__ = [
    "ARCHETYPE_GUIDANCE",
    "draft_pair",
    "envelope_from_yaml",
    "envelope_to_yaml",
    "parse_label",
    "upsert_pair",
]

# What the author is asked to do to the clean answer — one fault, nothing else changed, so
# a score drop can only mean the judge saw that fault.
ARCHETYPE_GUIDANCE: Mapping[FaultArchetype, str] = {
    FaultArchetype.FABRICATED_CITATION: (
        "Add or swap one `fondement_juridique` entry for a real-looking article id that is NOT "
        "in the retrieved context (see retrieved_citation_ids), with a plausible gloss. Leave "
        "the explanation as it is."
    ),
    FaultArchetype.CLAIM_ABSENT_FROM_CONTEXT: (
        "Add one plausible, specific claim to the explanation (a delay, an amount, a condition) "
        "that the context does not support. Change nothing else."
    ),
    FaultArchetype.DROPPED_EXPECTED_POINT: (
        "Remove what asserts one expected_point from the explanation, keeping the rest fluent. "
        "Don't add anything."
    ),
    FaultArchetype.REFUSAL_WITHOUT_EXPLANATION: (
        "Keep the refusal and its motif, and cut the informational half of the explanation — "
        "the part that still explains the rule (SPEC §10.4). Needs a clean answer that is a "
        "refus."
    ),
}


def draft_pair(item: GoldenItem, result: GenerationResult, archetype: FaultArchetype) -> CalibrationPair:
    """The pair before the fault is planted: the real answer and a copy of it for the
    author to edit into the twin. The `pass`/`fail` labels here are only the defaults the
    helper offers — the author confirms or overrides each (`parse_label`) before the pair is
    written, since nothing guarantees the real answer is actually clean."""
    return CalibrationPair(
        golden_id=item.id,
        archetype=archetype,
        question=item.question,
        expected_points=item.expected_points,
        retrieved_citation_ids=tuple(sorted(retrieved_citation_ids(result.contexts))),
        context=build_context_section(result.contexts),
        clean=CalibrationAnswer(result.envelope, HumanLabel.PASS),
        faulted=CalibrationAnswer(result.envelope, HumanLabel.FAIL),
    )


def parse_label(raw: str, *, default: HumanLabel) -> HumanLabel | None:
    """An author's answer to "pass or fail?": `p`/`pass`, `f`/`fail` (any case), empty for
    `default`, `None` for anything else so the caller can ask again."""
    answer = raw.strip().lower()
    if not answer:
        return default
    for label in HumanLabel:
        if answer in (label.value, label.value[0]):
            return label
    return None


def envelope_to_yaml(envelope: Envelope) -> str:
    return block_yaml(ENVELOPE_ADAPTER.dump_python(envelope, mode="json"))


def envelope_from_yaml(text: str) -> Envelope:
    """Parse an edited envelope back, raising `CalibrationSetError` with the schema's own reason
    — the author is looking at the editor, not a traceback."""
    try:
        return ENVELOPE_ADAPTER.validate_python(yaml.safe_load(text))
    except (yaml.YAMLError, ValidationError) as error:
        raise CalibrationSetError(f"not a valid envelope: {error}") from None


def upsert_pair(path: Path, pair: CalibrationPair, golden_set: Sequence[GoldenItem]) -> None:
    """Write `pair` into the calibration set at `path`, replacing any pair for the same golden
    item, after validating the whole resulting set. Raises `CalibrationSetError` — and writes
    nothing — if any violation remains."""
    pairs = [existing for existing in load_calibration_set(path) if existing.golden_id != pair.golden_id]
    pairs.append(pair)
    violations = validate_calibration_set(pairs, golden_set)
    if violations:
        raise CalibrationSetError("\n".join(violations))
    dump_calibration_set(pairs, path)
