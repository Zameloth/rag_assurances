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

from rag.ingest.arms import ARTICLES_ALIAS, FICHES_ALIAS
from rag.restore import release_tag_of

__all__ = ["AliasProbe", "HealthReport", "check_health", "qdrant_aliases"]

# Every alias Qdrant holds, name -> collection; raises when Qdrant cannot be reached.
AliasProbe = Callable[[], Mapping[str, str]]

_REGISTER_ALIASES = {"fiches": FICHES_ALIAS, "articles": ARTICLES_ALIAS}


@dataclass(frozen=True)
class HealthReport:
    models_loaded: bool
    qdrant_reachable: bool
    release_tag: str
    # Per register, the release tag its alias target carries — `None` when there is no
    # alias, or it points at something that is not a release generation of that register.
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
        }


def check_health(*, models_loaded: bool, aliases: AliasProbe, release_tag: str) -> HealthReport:
    try:
        targets = aliases()
    except Exception:  # noqa: BLE001 — any failure to reach Qdrant means the same thing here
        return HealthReport(
            models_loaded=models_loaded,
            qdrant_reachable=False,
            release_tag=release_tag,
            served={register: None for register in _REGISTER_ALIASES},
        )
    served: dict[str, str | None] = {}
    for register, alias in _REGISTER_ALIASES.items():
        target = targets.get(alias)
        served[register] = release_tag_of(register, target) if target is not None else None
    return HealthReport(
        models_loaded=models_loaded, qdrant_reachable=True, release_tag=release_tag, served=served
    )


def qdrant_aliases(client: QdrantClient) -> AliasProbe:
    def probe() -> Mapping[str, str]:
        return {a.alias_name: a.collection_name for a in client.get_aliases().aliases}

    return probe
