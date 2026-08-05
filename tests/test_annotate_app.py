"""The annotation helper's routes (SPEC §12.1-§12.4, ADR-0010, #33) — wired against the
real committed corpus (read-only) with a throwaway golden-set/counter under `tmp_path`, so
a test run never touches `eval/golden/golden-set.yaml`.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.eval.annotate_app import create_app
from rag.eval.schema import load_golden_set

REPO_ROOT = Path(__file__).resolve().parents[1]
FICHES_DIR = REPO_ROOT / "data" / "corpus" / "fiches"
ARTICLES_PATH = REPO_ROOT / "data" / "corpus" / "articles.jsonl"

# Same F1124 fixture as test_eval_corpus.py / test_golden_validate.py.
FICHE_ID = "F1124"
VALID_ARTICLE_CID = "LEGIARTI000006792738"  # L127-1, under F1124's dc:source section


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_app(
        fiches_dir=FICHES_DIR,
        articles_path=ARTICLES_PATH,
        golden_set_path=tmp_path / "golden-set.yaml",
        id_counter_path=tmp_path / ".next_id",
    )
    return TestClient(app)


def _payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = dict(
        question="je suis locataire, je dois vraiment prendre une assurance ?",
        expected_state="reponse",
        gold_fiches=[FICHE_ID],
        gold_articles=[VALID_ARTICLE_CID],
        gold_spans=[],
        expected_points=["la responsabilité civile locative est obligatoire"],
        tags=[],
        history=[],
    )
    base.update(overrides)
    return base


def test_index_lists_fiches(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert FICHE_ID in response.text


def test_fiche_detail_renders_body_sections_and_states(client: TestClient) -> None:
    response = client.get(f"/fiche/{FICHE_ID}")
    assert response.status_code == 200
    assert "Assurance des associations" in response.text
    assert "reponse_sans_article" in response.text  # one of the closed-vocabulary options
    assert "dc:source" in response.text.lower() or "suggestions" in response.text.lower()


def test_fiche_detail_404_for_unknown_fiche(client: TestClient) -> None:
    response = client.get("/fiche/F0000000")
    assert response.status_code == 404


def test_api_articles_exposes_the_full_corpus(client: TestClient) -> None:
    response = client.get("/api/articles")
    assert response.status_code == 200
    body = response.json()
    assert VALID_ARTICLE_CID in body
    assert body[VALID_ARTICLE_CID]["citation_id"] == "L127-1"
    assert response.headers["cache-control"] == "public, max-age=3600"


def test_save_writes_a_validated_item(client: TestClient, tmp_path: Path) -> None:
    response = client.post("/save", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert body == {"ok": True, "id": "gs-001"}

    items = load_golden_set(tmp_path / "golden-set.yaml")
    assert len(items) == 1
    assert items[0].id == "gs-001"
    assert items[0].gold_fiches == (FICHE_ID,)


def test_save_rejects_an_unresolvable_article_and_writes_nothing(client: TestClient, tmp_path: Path) -> None:
    response = client.post("/save", json=_payload(gold_articles=["LEGIARTI000000000000"]))
    assert response.status_code == 422
    body = response.json()
    assert body["ok"] is False
    assert "gold_articles" in body["detail"]
    assert not (tmp_path / "golden-set.yaml").exists()


def test_save_never_reuses_an_id_even_after_a_failed_save(client: TestClient, tmp_path: Path) -> None:
    failed = client.post("/save", json=_payload(gold_articles=["LEGIARTI000000000000"]))
    assert failed.status_code == 422
    assert failed.json()["id"] == "gs-001"

    succeeded = client.post("/save", json=_payload())
    assert succeeded.status_code == 200
    assert succeeded.json()["id"] == "gs-002"  # gs-001 was burned, not reused


def test_save_second_item_gets_the_next_id(client: TestClient) -> None:
    first = client.post("/save", json=_payload())
    second = client.post("/save", json=_payload(question="une autre question ?"))
    assert first.json()["id"] == "gs-001"
    assert second.json()["id"] == "gs-002"


def test_helper_never_writes_to_the_corpus(client: TestClient) -> None:
    fiches_before = {p: p.read_bytes() for p in FICHES_DIR.glob("*.xml")}
    articles_before = ARTICLES_PATH.read_bytes()

    client.get("/")
    client.get(f"/fiche/{FICHE_ID}")
    client.get("/api/articles")
    client.post("/save", json=_payload())

    assert articles_before == ARTICLES_PATH.read_bytes()
    assert fiches_before == {p: p.read_bytes() for p in FICHES_DIR.glob("*.xml")}
