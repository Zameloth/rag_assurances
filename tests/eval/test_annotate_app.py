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
from rag.eval.schema import EXPECTED_STATES, load_golden_set

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


def test_api_fiches_exposes_id_and_title_for_client_side_search(client: TestClient) -> None:
    response = client.get("/api/fiches")
    assert response.status_code == 200
    by_id = {row["id"]: row["title"] for row in response.json()}
    assert FICHE_ID in by_id
    assert by_id[FICHE_ID]  # a nonempty title, not just the id echoed back


def test_api_progress_targets_the_full_sixty_item_composition(client: TestClient) -> None:
    """#44 — once behavioural items land in the same golden-set.yaml, `total` counts all
    of them (SPEC §12.1: 60 hand-annotated items), so its target must be 60, not just #34's
    38-item retrieval-bearing slice."""
    response = client.get("/api/progress")
    assert response.status_code == 200
    body = response.json()
    assert body["target_total"] == 60
    assert body["target_refus_regulated"] == 12
    assert body["target_refus_hors_corpus"] == 10
    assert body["target_multi_turn"] == 10


def test_manual_marks_regulated_refus_and_no_points_states_in_option_data(client: TestClient) -> None:
    """The regulated-refus pair and the hors_corpus/no-expected_points rule are read by the
    page's JS from the option's own data attributes — one source of truth with the server's
    `_expected_state_options()`, not a second hardcoded list in the template's script."""
    response = client.get("/manual")
    assert response.status_code == 200
    assert 'value="refus:recommandation_produit" data-regulated-refus="true" data-no-points="false"' in response.text
    assert 'value="refus:conseil_action" data-regulated-refus="true" data-no-points="false"' in response.text
    assert 'value="refus:hors_corpus" data-regulated-refus="false" data-no-points="true"' in response.text
    assert 'value="reponse" data-regulated-refus="false" data-no-points="false"' in response.text


def test_manual_renders_all_five_expected_states_with_distinguishing_labels(client: TestClient) -> None:
    """#44 — the no-fiche-anchor authoring page must offer every EXPECTED_STATES value,
    and the two empty-cell states must read as visibly different choices rather than two
    bare identifiers an annotator could swap by mistake."""
    response = client.get("/manual")
    assert response.status_code == 200
    for state in EXPECTED_STATES:
        assert f'value="{state}"' in response.text
    # reponse_sans_article ("empty is correct") and refus:hors_corpus ("nothing exists")
    # must not read as the same kind of blank — each option's own label says which.
    assert "aucun article ne suffit" in response.text or "article ne suffit" in response.text
    assert "rien n'existe" in response.text or "rien n" in response.text.lower()


def test_fiche_detail_also_uses_the_distinguishing_expected_state_labels(client: TestClient) -> None:
    response = client.get(f"/fiche/{FICHE_ID}")
    assert response.status_code == 200
    for state in EXPECTED_STATES:
        assert f'value="{state}"' in response.text


# --- multi_turn tag derivation on save (#44) ----------------------------------


def test_save_adds_multi_turn_tag_when_history_is_nonempty(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/save",
        json=_payload(history=[{"role": "user", "content": "je loue un appartement"}]),
    )
    assert response.status_code == 200
    items = load_golden_set(tmp_path / "golden-set.yaml")
    assert "multi_turn" in items[0].tags


def test_save_does_not_duplicate_multi_turn_tag_already_present(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/save",
        json=_payload(
            history=[{"role": "user", "content": "je loue un appartement"}],
            tags=["multi_turn"],
        ),
    )
    assert response.status_code == 200
    items = load_golden_set(tmp_path / "golden-set.yaml")
    assert items[0].tags.count("multi_turn") == 1


def test_save_strips_a_stray_multi_turn_tag_when_history_is_empty(client: TestClient, tmp_path: Path) -> None:
    """The tag is derived from `history`, never trusted from the client — a stray tag on a
    single-turn item would otherwise defeat #44's cross-cutting-tag invariant."""
    response = client.post("/save", json=_payload(tags=["multi_turn"]))
    assert response.status_code == 200
    items = load_golden_set(tmp_path / "golden-set.yaml")
    assert "multi_turn" not in items[0].tags


# --- end-to-end behavioural/multi-turn items through /save (#44) -------------


def test_save_a_hors_corpus_item_with_no_fiche_anchor(client: TestClient, tmp_path: Path) -> None:
    """#44 — a behavioural item authored with no `<dc:source>`-anchored fiche at all:
    `refus:hors_corpus` means nothing exists, so every gold context stays empty."""
    response = client.post(
        "/save",
        json=_payload(
            question="pouvez-vous m'aider à résilier mon abonnement de streaming ?",
            expected_state="refus:hors_corpus",
            gold_fiches=[],
            gold_articles=[],
            gold_spans=[],
            expected_points=[],
        ),
    )
    assert response.status_code == 200
    items = load_golden_set(tmp_path / "golden-set.yaml")
    assert items[0].expected_state == "refus:hors_corpus"
    assert items[0].gold_fiches == ()
    assert items[0].gold_articles == ()


def test_save_a_regulated_refus_item_keeps_populated_gold_contexts(client: TestClient, tmp_path: Path) -> None:
    """#44 — regulated-act refusals are full ladder items: retrieval succeeds, so
    gold_fiches/gold_articles stay populated even though the answer is a refusal."""
    response = client.post(
        "/save",
        json=_payload(
            question="quelle assurance dois-je prendre pour mon studio ?",
            expected_state="refus:recommandation_produit",
            gold_fiches=[FICHE_ID],
            gold_articles=[VALID_ARTICLE_CID],
            gold_spans=[],
            expected_points=["le refus oriente vers un comparateur, sans recommander un produit précis"],
        ),
    )
    assert response.status_code == 200
    items = load_golden_set(tmp_path / "golden-set.yaml")
    assert items[0].expected_state == "refus:recommandation_produit"
    assert items[0].gold_fiches == (FICHE_ID,)
    assert items[0].gold_articles == (VALID_ARTICLE_CID,)


def test_save_a_multi_turn_item_with_stripped_history(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/save",
        json=_payload(
            question="et si je résilie avant la fin ?",
            history=[
                {"role": "user", "content": "je loue un appartement, dois-je m'assurer ?"},
                {"role": "assistant", "content": "oui, l'assurance habitation est obligatoire pour un locataire."},
            ],
        ),
    )
    assert response.status_code == 200
    items = load_golden_set(tmp_path / "golden-set.yaml")
    assert "multi_turn" in items[0].tags
    assert len(items[0].history) == 2


def test_save_rejects_an_unstripped_assistant_turn(client: TestClient, tmp_path: Path) -> None:
    """SPEC §8.7 — history must already be in the pipeline's stripped form; an assistant
    turn still carrying the raw envelope's `fondement_juridique` field is caught at
    save time, not silently written."""
    response = client.post(
        "/save",
        json=_payload(
            history=[
                {"role": "user", "content": "je loue un appartement"},
                {
                    "role": "assistant",
                    "content": '{"fondement_juridique": [{"article_id": "L127-1"}]}',
                },
            ],
        ),
    )
    assert response.status_code == 422
    assert "fondement_juridique" in response.json()["detail"]
    assert not (tmp_path / "golden-set.yaml").exists()


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
