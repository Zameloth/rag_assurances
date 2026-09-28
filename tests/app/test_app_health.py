"""`GET /health` — three clauses (SPEC §14.3, §15.6, #53).

    models loaded  AND  Qdrant reachable  AND  alias target == index_lock.json

Sablier closes its waiting page on this signal, so each clause is pinned failing on its
own: anything the definition omits is handed to a visitor as a failure.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from test_app_routes import ATTRIBUTION
from test_app_stream import _staged_answer

from rag.app.health import AliasProbe, check_health, qdrant_aliases
from rag.app.main import create_app
from rag.ingest.arms import ensure_articles_collection, ensure_fiches_collection, flip_alias

TAG = "index-2026-09-28"
LIVE = {"fiches": f"fiches__{TAG}", "articles": f"articles__{TAG}"}


def _probe(aliases: Mapping[str, str]) -> AliasProbe:
    return lambda: aliases


def _unreachable() -> Mapping[str, str]:
    raise ConnectionError("qdrant is still waking")


class TestCheckHealth:
    def test_healthy_when_all_three_clauses_hold(self) -> None:
        report = check_health(models_loaded=True, aliases=_probe(LIVE), release_tag=TAG)

        assert report.healthy
        assert report.models_loaded and report.qdrant_reachable and report.index_matches_lock

    def test_models_not_loaded_is_unhealthy_on_its_own(self) -> None:
        report = check_health(models_loaded=False, aliases=_probe(LIVE), release_tag=TAG)

        assert not report.healthy
        assert report.qdrant_reachable and report.index_matches_lock

    def test_unreachable_qdrant_is_unhealthy_and_says_which_clause_failed(self) -> None:
        report = check_health(models_loaded=True, aliases=_unreachable, release_tag=TAG)

        assert not report.healthy
        assert not report.qdrant_reachable
        assert not report.index_matches_lock

    def test_an_alias_on_another_release_is_unhealthy(self) -> None:
        """A pulled image whose restore was forgotten: the store serves last release."""
        stale = {**LIVE, "articles": "articles__index-2026-08-03"}

        report = check_health(models_loaded=True, aliases=_probe(stale), release_tag=TAG)

        assert not report.index_matches_lock
        assert report.served == {"fiches": TAG, "articles": "index-2026-08-03"}

    @pytest.mark.parametrize(
        "aliases",
        [
            {"fiches": f"fiches__{TAG}"},  # no articles alias at all: an empty store
            {**LIVE, "fiches": "fiches__m3__c512__v1"},  # a dev ladder arm, never a release
            {"fiches": f"articles__{TAG}", "articles": f"fiches__{TAG}"},  # crossed registers
        ],
    )
    def test_an_alias_that_is_not_this_release_s_generation_is_unhealthy(
        self, aliases: Mapping[str, str]
    ) -> None:
        report = check_health(models_loaded=True, aliases=_probe(aliases), release_tag=TAG)

        assert not report.index_matches_lock

    def test_no_alias_and_an_alias_on_a_non_release_are_told_apart(self) -> None:
        """Different fixes: run restore, versus find who flipped the alias by hand."""
        aliases = {"fiches": "fiches__m3__c512__v1"}

        report = check_health(models_loaded=True, aliases=_probe(aliases), release_tag=TAG)

        assert report.targets == {"fiches": "fiches__m3__c512__v1", "articles": None}

    def test_the_report_serializes_each_clause_separately(self) -> None:
        report = check_health(models_loaded=True, aliases=_probe(LIVE), release_tag=TAG)

        assert report.as_dict() == {
            "healthy": True,
            "models_loaded": True,
            "qdrant_reachable": True,
            "index_matches_lock": True,
            "release_tag": TAG,
            "served": {"fiches": TAG, "articles": TAG},
            "targets": LIVE,
        }


def test_qdrant_aliases_reads_every_alias_target(qdrant: QdrantClient) -> None:
    ensure_fiches_collection(qdrant, f"fiches__{TAG}", dense_dim=4)
    ensure_articles_collection(qdrant, f"articles__{TAG}", dense_dim=4)
    flip_alias(qdrant, "fiches", f"fiches__{TAG}")
    flip_alias(qdrant, "articles", f"articles__{TAG}")

    assert qdrant_aliases(qdrant)() == LIVE


class TestRoute:
    def _client(self, aliases: AliasProbe, *, load_models: Callable[[], None] | None = None) -> TestClient:
        return TestClient(
            create_app(
                answer=_staged_answer(),
                attribution=ATTRIBUTION,
                load_models=load_models or (lambda: None),
                aliases=aliases,
                release_tag=TAG,
            )
        )

    def test_200_when_healthy(self) -> None:
        with self._client(_probe(LIVE)) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["healthy"] is True

    def test_503_when_any_clause_fails(self) -> None:
        with self._client(_unreachable) as client:
            response = client.get("/health")

        assert response.status_code == 503
        assert response.json()["qdrant_reachable"] is False

    def test_models_are_loaded_once_at_startup_before_the_app_serves(self) -> None:
        """SPEC §14.2: eager, and never unloaded — the container's lifetime is the model's."""
        loads: list[str] = []

        with self._client(_probe(LIVE), load_models=lambda: loads.append("embedder")) as client:
            assert loads == ["embedder"]
            client.get("/health")
            client.post("/api/ask", json={"question": "Puis-je résilier ?"})

        assert loads == ["embedder"]

    def test_the_models_clause_is_false_until_startup_has_loaded_them(self) -> None:
        """Without the lifespan having run, nothing has loaded — and /health must say so."""
        client = self._client(_probe(LIVE))  # not entered: startup never ran

        response = client.get("/health")

        assert response.status_code == 503
        assert response.json()["models_loaded"] is False

    def test_a_model_load_that_fails_fails_startup(self) -> None:
        """A container that cannot load its models dies, rather than sitting unhealthy
        forever: Sablier starts it afresh on the next visit."""

        def broken() -> None:
            raise OSError("weights not found")

        with pytest.raises(OSError, match="weights"), self._client(_probe(LIVE), load_models=broken):
            pass
