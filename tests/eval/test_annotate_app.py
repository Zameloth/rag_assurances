"""The annotation helper's routes (SPEC §12.1-§12.4, ADR-0010, #33) — wired against the
real committed corpus (read-only) with a throwaway golden-set/counter under `tmp_path`, so
a test run never touches `eval/golden/golden-set.yaml`.
"""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import rag.eval.annotate_app as annotate_app_module
from rag.config import Settings
from rag.eval.annotate_app import create_app
from rag.eval.corpus import fiche_chunk_texts
from rag.eval.schema import load_golden_set

REPO_ROOT = Path(__file__).resolve().parents[2]
FICHES_DIR = REPO_ROOT / "data" / "corpus" / "fiches"
ARTICLES_PATH = REPO_ROOT / "data" / "corpus" / "articles.jsonl"

# Same F1124 fixture as test_eval_corpus.py / test_golden_validate.py.
FICHE_ID = "F1124"
VALID_ARTICLE_CID = "LEGIARTI000006792738"  # L127-1, under F1124's dc:source section
# Committed, but outside every one of F1124's dc:source sections (verified against
# articles.jsonl) — used to exercise the "lexical only" provenance path.
OFF_SECTION_ARTICLE_CID = "LEGIARTI000006785773"


def _settings(**overrides: Any) -> Settings:
    fields: dict[str, Any] = {
        "openrouter_api_key": "sk-or-test",
        "generation_model": "mistralai/mistral-large-2512",
        "generation_provider": "",
        "condenser_model": "",
        "condenser_provider": "",
        "judge_model": "",
        "judge_provider": "",
        "langfuse_public_key": "",
        "langfuse_secret_key": "",
        "langfuse_base_url": "https://cloud.langfuse.com",
        "langfuse_tracing": False,
        "qdrant_url": "http://localhost:6333",
    }
    fields.update(overrides)
    return Settings(**fields)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChatOpenAI:
    """Records the prompt it was invoked with and hands back a canned response —
    never touches the network. `response_content` is set per test before the request
    that triggers the LLM call."""

    response_content: str = ""
    last_prompt: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        pass

    def invoke(self, messages: list[Any]) -> _FakeResponse:
        _FakeChatOpenAI.last_prompt = messages[0].content
        return _FakeResponse(_FakeChatOpenAI.response_content)


@pytest.fixture(autouse=True)
def _fake_chat_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeChatOpenAI.response_content = ""
    _FakeChatOpenAI.last_prompt = None
    monkeypatch.setattr(annotate_app_module, "ChatOpenAI", _FakeChatOpenAI)


def _make_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, settings: Settings) -> TestClient:
    monkeypatch.setattr(annotate_app_module, "load_settings", lambda: settings)
    app = create_app(
        fiches_dir=FICHES_DIR,
        articles_path=ARTICLES_PATH,
        golden_set_path=tmp_path / "golden-set.yaml",
        id_counter_path=tmp_path / ".next_id",
    )
    return TestClient(app)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Credentials present — the default for every test that isn't specifically about
    the missing-credentials path, which builds its own client with `_settings(...)`."""
    return _make_client(tmp_path, monkeypatch, settings=_settings())


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


# --- /api/propose-gold-articles (#70, ADR-0020) ------------------------------


def test_propose_gold_articles_tags_dc_source_and_lexical_provenance(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real lexical scorer is unit-tested in test_propose.py; pinning it here keeps
    # this test's expected shortlist independent of the full 2,377-article corpus.
    monkeypatch.setattr(annotate_app_module, "lexical_shortlist", lambda *a, **k: [OFF_SECTION_ARTICLE_CID])
    _FakeChatOpenAI.response_content = (
        f"{VALID_ARTICLE_CID}: couvre le cas décrit dans la question\n"
        f"{OFF_SECTION_ARTICLE_CID}: complète le contexte\n"
    )
    response = client.post("/api/propose-gold-articles", json={"fiche_id": FICHE_ID, "question": "une question ?"})
    assert response.status_code == 200
    by_cid = {c["cid"]: c for c in response.json()["candidates"]}
    assert set(by_cid) == {VALID_ARTICLE_CID, OFF_SECTION_ARTICLE_CID}
    assert by_cid[VALID_ARTICLE_CID]["provenance"] == ["dc:source"]
    assert by_cid[VALID_ARTICLE_CID]["justification"] == "couvre le cas décrit dans la question"
    assert by_cid[OFF_SECTION_ARTICLE_CID]["provenance"] == ["lexical"]


def test_propose_gold_articles_drops_a_cid_outside_the_shortlist(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(annotate_app_module, "lexical_shortlist", lambda *a, **k: [])
    _FakeChatOpenAI.response_content = (
        f"{VALID_ARTICLE_CID}: pertinent\nLEGIARTI000000000000: cid hors liste, jamais du corpus réel\n"
    )
    response = client.post("/api/propose-gold-articles", json={"fiche_id": FICHE_ID, "question": "une question ?"})
    assert response.status_code == 200
    cids = [c["cid"] for c in response.json()["candidates"]]
    assert cids == [VALID_ARTICLE_CID]  # the hallucinated cid never reaches the UI


def test_propose_gold_articles_deduplicates_a_cid_named_on_two_lines(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(annotate_app_module, "lexical_shortlist", lambda *a, **k: [])
    _FakeChatOpenAI.response_content = f"{VALID_ARTICLE_CID}: premier avis\n{VALID_ARTICLE_CID}: avis redondant\n"
    response = client.post("/api/propose-gold-articles", json={"fiche_id": FICHE_ID, "question": "une question ?"})
    assert response.status_code == 200
    candidates = response.json()["candidates"]
    assert len(candidates) == 1
    assert candidates[0]["justification"] == "premier avis"  # the first mention wins


def test_propose_gold_articles_404_for_unknown_fiche(client: TestClient) -> None:
    response = client.post("/api/propose-gold-articles", json={"fiche_id": "F0000000", "question": "?"})
    assert response.status_code == 404


def test_propose_gold_articles_502_when_credentials_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(tmp_path, monkeypatch, settings=_settings(openrouter_api_key=""))
    response = client.post("/api/propose-gold-articles", json={"fiche_id": FICHE_ID, "question": "une question ?"})
    assert response.status_code == 502
    assert response.json()["ok"] is False


# --- /api/propose-gold-spans (#70, ADR-0020) ---------------------------------


def test_propose_gold_spans_keeps_only_verbatim_candidates(client: TestClient) -> None:
    chunks = fiche_chunk_texts(FICHE_ID, FICHES_DIR)
    real_excerpt = chunks[0][:40]
    _FakeChatOpenAI.response_content = f'"{real_excerpt}"\nceci n\'est écrit nulle part dans la fiche\n'
    response = client.post("/api/propose-gold-spans", json={"fiche_id": FICHE_ID, "question": "une question ?"})
    assert response.status_code == 200
    spans = response.json()["spans"]
    assert spans == [real_excerpt]  # the fabricated line is silently dropped, never shown


def test_propose_gold_spans_404_for_unknown_fiche(client: TestClient) -> None:
    response = client.post("/api/propose-gold-spans", json={"fiche_id": "F0000000", "question": "?"})
    assert response.status_code == 404


# --- /api/propose-expected-points (#70, ADR-0020) ----------------------------


def test_propose_expected_points_uses_spans_and_articles_as_context(client: TestClient) -> None:
    _FakeChatOpenAI.response_content = "la responsabilité civile locative est obligatoire\n"
    response = client.post(
        "/api/propose-expected-points",
        json={
            "question": "je dois vraiment prendre une assurance ?",
            "gold_spans": ["l'assurance est obligatoire"],
            "gold_articles": [VALID_ARTICLE_CID],
        },
    )
    assert response.status_code == 200
    assert response.json()["points"] == ["la responsabilité civile locative est obligatoire"]
    assert "l'assurance est obligatoire" in (_FakeChatOpenAI.last_prompt or "")
    assert "L127-1" in (_FakeChatOpenAI.last_prompt or "")


def test_propose_expected_points_caps_at_three(client: TestClient) -> None:
    _FakeChatOpenAI.response_content = "\n".join(f"point {i}" for i in range(5))
    response = client.post("/api/propose-expected-points", json={"question": "?"})
    assert response.status_code == 200
    assert len(response.json()["points"]) == 3


def test_propose_expected_points_502_when_credentials_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(tmp_path, monkeypatch, settings=_settings(generation_model=""))
    response = client.post("/api/propose-expected-points", json={"question": "?"})
    assert response.status_code == 502
    assert response.json()["ok"] is False


# --- ai_assisted tag on save (#70, ADR-0020) ---------------------------------


def test_save_appends_ai_assisted_tag_when_flagged(client: TestClient, tmp_path: Path) -> None:
    response = client.post("/save", json=_payload(ai_assisted=True))
    assert response.status_code == 200
    items = load_golden_set(tmp_path / "golden-set.yaml")
    assert "ai_assisted" in items[0].tags


def test_save_omits_ai_assisted_tag_by_default(client: TestClient, tmp_path: Path) -> None:
    response = client.post("/save", json=_payload())
    assert response.status_code == 200
    items = load_golden_set(tmp_path / "golden-set.yaml")
    assert "ai_assisted" not in items[0].tags


def test_save_does_not_duplicate_an_already_present_ai_assisted_tag(client: TestClient, tmp_path: Path) -> None:
    response = client.post("/save", json=_payload(ai_assisted=True, tags=["ai_assisted"]))
    assert response.status_code == 200
    items = load_golden_set(tmp_path / "golden-set.yaml")
    assert items[0].tags.count("ai_assisted") == 1
