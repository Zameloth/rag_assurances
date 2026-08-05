"""Golden-set item schema (SPEC §12.1, ADR-0010, #33).

`eval/golden/golden-set.yaml` is one repo-canonical YAML file: a list of items, each with
the nine fields SPEC §12.1 fixes — `id`, `question`, `history`, `expected_state`,
`gold_fiches`, `gold_spans`, `gold_articles`, `expected_points`, `tags`. This module is
shape only: every field present and of the right container type. Whether the *values*
resolve against the committed corpus, and whether `expected_state` and its label-emptiness
pattern agree, is `validate.py`'s job (#33) — a different question, checked against
different inputs (the corpus, not the YAML alone).

`history` turns are kept loosely typed (`Mapping[str, str]`, any keys) rather than a fixed
`role`/`content` dataclass: no ticket has fixed that contract yet, and inventing one here
would risk disagreeing with whatever #8/#17's condenser and #13's app actually settle on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "EXPECTED_STATES",
    "FIELDS",
    "GoldenItem",
    "GoldenSetSchemaError",
    "dump_golden_set",
    "load_golden_set",
]

# SPEC §10.2/§10.3, mirrored field for field — the answer envelope's five terminal shapes,
# `type` + `motif` collapsed into one string so state comparison is direct (SPEC §12.1).
EXPECTED_STATES = frozenset(
    {
        "reponse",
        "reponse_sans_article",
        "refus:recommandation_produit",
        "refus:conseil_action",
        "refus:hors_corpus",
    }
)

# Declaration order doubles as the on-disk field order (SPEC §12.1's example) — `as_dict`
# and `dump_golden_set` below both walk this tuple rather than repeating the list.
FIELDS = (
    "id",
    "question",
    "history",
    "expected_state",
    "gold_fiches",
    "gold_spans",
    "gold_articles",
    "expected_points",
    "tags",
)

_STRING_LIST_FIELDS = ("gold_fiches", "gold_spans", "gold_articles", "expected_points", "tags")


class GoldenSetSchemaError(Exception):
    """One or more golden-set items don't shape into `GoldenItem` — SPEC §12.1's fields
    missing, or present with the wrong container type. Raised with every violation found,
    not just the first (same reasoning as `ingest.assertions.CorpusAssertionError`)."""


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    history: tuple[Mapping[str, str], ...]
    expected_state: str
    gold_fiches: tuple[str, ...]
    gold_spans: tuple[str, ...]
    gold_articles: tuple[str, ...]
    expected_points: tuple[str, ...]
    tags: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "history": [dict(turn) for turn in self.history],
            "expected_state": self.expected_state,
            "gold_fiches": list(self.gold_fiches),
            "gold_spans": list(self.gold_spans),
            "gold_articles": list(self.gold_articles),
            "expected_points": list(self.expected_points),
            "tags": list(self.tags),
        }


def load_golden_set(path: Path) -> list[GoldenItem]:
    """Parse `eval/golden/golden-set.yaml` into `GoldenItem`s.

    A missing or empty file reads as zero items — the correct answer for a fresh checkout
    or a golden set with nothing annotated yet, not an error.
    """
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise GoldenSetSchemaError(f"{path}: expected a YAML list of items, got {type(raw).__name__}")

    items: list[GoldenItem] = []
    violations: list[str] = []
    for index, entry in enumerate(raw):
        label = entry.get("id") if isinstance(entry, Mapping) else None
        ref = label or f"index {index}"
        entry_violations = _validate_entry(entry, ref)
        if entry_violations:
            violations.extend(entry_violations)
            continue
        items.append(_to_item(entry))

    if violations:
        raise GoldenSetSchemaError(f"{len(violations)} golden-set schema violation(s):\n" + "\n".join(violations))
    return items


def dump_golden_set(items: list[GoldenItem], path: Path) -> None:
    """Write `items` back as valid, human-reviewable YAML — block style, `FIELDS` order,
    never flow-style `{...}`/`[...]` (SPEC §12.11: the point of repo-canonical storage is
    `git diff`/`git blame` on ground truth, and a flow-style dump reads as one opaque line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [item.as_dict() for item in items]
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, default_flow_style=False, width=100),
        encoding="utf-8",
    )


def _validate_entry(entry: Any, ref: str) -> list[str]:
    if not isinstance(entry, Mapping):
        return [f"{ref}: item is a {type(entry).__name__}, not a mapping"]

    violations = []
    missing = [field for field in FIELDS if field not in entry]
    if missing:
        violations.append(f"{ref}: missing field(s) {missing}")
        return violations  # further checks would just repeat "field absent"

    if not isinstance(entry["id"], str):
        violations.append(f"{ref}: id must be a string")
    if not isinstance(entry["question"], str):
        violations.append(f"{ref}: question must be a string")
    if not isinstance(entry["expected_state"], str):
        violations.append(f"{ref}: expected_state must be a string")

    if not isinstance(entry["history"], list) or not all(isinstance(turn, Mapping) for turn in entry["history"]):
        violations.append(f"{ref}: history must be a list of mappings")
    elif any(not all(isinstance(v, str) for v in turn.values()) for turn in entry["history"]):
        violations.append(f"{ref}: history turn values must be strings")

    for field in _STRING_LIST_FIELDS:
        value = entry[field]
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            violations.append(f"{ref}: {field} must be a list of strings")

    return violations


def _to_item(entry: Mapping[str, Any]) -> GoldenItem:
    return GoldenItem(
        id=entry["id"],
        question=entry["question"],
        history=tuple(dict(turn) for turn in entry["history"]),
        expected_state=entry["expected_state"],
        gold_fiches=tuple(entry["gold_fiches"]),
        gold_spans=tuple(entry["gold_spans"]),
        gold_articles=tuple(entry["gold_articles"]),
        expected_points=tuple(entry["expected_points"]),
        tags=tuple(entry["tags"]),
    )
