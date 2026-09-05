"""The retrieval candidate shape and its merge rule (SPEC §7.5, §9, #28).

`register` and `provenance` are retrieval **annotations** — attached here, on the way out
of a query, never carried by a stored payload (SPEC §7.5). `Candidate` is the one place
both get attached; downstream (the retriever's `Document.metadata`, later tickets) is a
straight copy of these fields, never a second computation of them.

`provenance` is a **set, not a scalar** (SPEC §7.5): the same point id can be reached by
more than one path in one query — search and expansion today, and in principle more paths
later — so candidates are deduped by point id with their provenance sets **unioned**. A
scalar would make "which path wins" depend on merge order, which is exactly the bug SPEC
§7.5 rules out.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

__all__ = [
    "Candidate",
    "Provenance",
    "Register",
    "merge_candidates",
]


class Register(enum.Enum):
    """CONTEXT.md's `register` — which of the two document populations a candidate
    belongs to. Set from the collection queried, never read off the payload."""

    FICHE = "fiche"
    ARTICLE = "article"


class Provenance(enum.Enum):
    """How a candidate was found. SPEC §7.5 names two values (expansion vs search);
    `LOOKUP` is this ticket's third — the short-circuit resolves by metadata lookup, which
    is neither a ranked search hit nor a `<dc:source>` expansion hit, so folding it into
    either label would misdescribe how the candidate was actually found. CONTEXT.md's
    `provenance` entry is updated alongside this to name all three."""

    SEARCH = "search"
    EXPANSION = "expansion"
    LOOKUP = "lookup"


@dataclass(frozen=True)
class Candidate:
    """One retrieval candidate: a point id, its score, the register it was queried under,
    its payload verbatim, and the provenance set it was reached by.

    `score` has no fixed scale across provenances — a cosine similarity from a search leg
    and a fixed sentinel from a metadata lookup are not comparable, which is exactly why
    the short-circuit path (SPEC §9.1) never merges its result with a search pool.
    """

    id: str
    score: float
    register: Register
    payload: Mapping[str, Any]
    provenance: frozenset[Provenance]


def merge_candidates(*pools: Iterable[Candidate]) -> list[Candidate]:
    """Dedupe candidates by point id across `pools`, unioning provenance (SPEC §7.5).

    The first pool to mention an id decides the kept `score`/`register`/`payload` — at
    rung 1 no id can appear in more than one pool (the fiche leg and the article leg query
    disjoint collections), so this only matters once expansion lands and a point can arrive
    by more than one path. Order is otherwise preserved: pools are concatenated in the
    order given, first occurrence order within that.
    """
    merged: dict[str, Candidate] = {}
    for pool in pools:
        for candidate in pool:
            existing = merged.get(candidate.id)
            if existing is None:
                merged[candidate.id] = candidate
            else:
                merged[candidate.id] = replace(
                    existing, provenance=existing.provenance | candidate.provenance
                )
    return list(merged.values())
