"""`/health` — three clauses (SPEC §14.3, §15.6, ADR-0013, #53).

    models loaded  AND  Qdrant reachable  AND  alias target == index_lock.json

Sablier closes its waiting page on this signal, so **anything the definition omits is
handed to a visitor as a failure**: without models-loaded they land on an app still
loading; without Qdrant-reachable the first query fails; without the alias check a
forgotten restore serves a stale or empty index.

The third clause costs one call: a generation's name carries its release tag
(`rag.restore.generation_name`), so reading the aliases is reading which release is served.
It is the same call that proves Qdrant reachable. In dev the aliases point at ladder arms,
not releases, so dev `/health` reports the index clause false — correctly: dev is not
serving the published index.

Each clause is reported separately so a failing probe says which thing broke.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient

from rag.ingest.arms import REGISTER_ALIASES
from rag.restore import release_tag_of

__all__ = ["AliasProbe", "HealthReport", "check_health", "qdrant_aliases"]

# Every alias Qdrant holds, name -> collection; raises when Qdrant cannot be reached.
AliasProbe = Callable[[], Mapping[str, str]]


@dataclass(frozen=True)
class HealthReport:
    models_loaded: bool
    qdrant_reachable: bool
    release_tag: str
    # Per register, the collection its alias points at — `None` when there is no alias
    # (or Qdrant could not be asked) — and the release tag that name carries, `None` when
    # it is not a release generation of that register. Both, because the two `None`s of
    # `served` have different fixes: run restore, versus find who flipped the alias.
    targets: Mapping[str, str | None]
    served: Mapping[str, str | None]

    @property
    def index_matches_lock(self) -> bool:
        return self.qdrant_reachable and all(tag == self.release_tag for tag in self.served.values())

    @property
    def healthy(self) -> bool:
        return self.models_loaded and self.qdrant_reachable and self.index_matches_lock

    def as_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "models_loaded": self.models_loaded,
            "qdrant_reachable": self.qdrant_reachable,
            "index_matches_lock": self.index_matches_lock,
            "release_tag": self.release_tag,
            "served": dict(self.served),
            "targets": dict(self.targets),
        }


def check_health(*, models_loaded: bool, aliases: AliasProbe, release_tag: str) -> HealthReport:
    try:
        found = aliases()
    except Exception:  # noqa: BLE001 — any failure to reach Qdrant means the same thing here
        nothing: dict[str, str | None] = dict.fromkeys(REGISTER_ALIASES)
        return HealthReport(
            models_loaded=models_loaded,
            qdrant_reachable=False,
            release_tag=release_tag,
            targets=nothing,
            served=nothing,
        )
    targets = {register: found.get(alias) for register, alias in REGISTER_ALIASES.items()}
    served = {
        register: release_tag_of(register, target) if target is not None else None
        for register, target in targets.items()
    }
    return HealthReport(
        models_loaded=models_loaded,
        qdrant_reachable=True,
        release_tag=release_tag,
        targets=targets,
        served=served,
    )


def qdrant_aliases(client: QdrantClient) -> AliasProbe:
    def probe() -> Mapping[str, str]:
        return {a.alias_name: a.collection_name for a in client.get_aliases().aliases}

    return probe
