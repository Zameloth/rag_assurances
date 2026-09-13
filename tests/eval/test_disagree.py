"""`rag.eval.disagree`'s disagreement-detector pass (ADR-0010, ADR-0020, #34).

`section_shortlist` / `find_outside_section_items` run against the real committed corpus,
same fixture shape as `test_eval_corpus.py`. The LLM-calling half (`independent_section_pick`
/ `run_disagreement_pass`) is exercised with `ChatOpenAI` monkeypatched to a canned fake,
same pattern as `test_annotate_app.py` — no network call in a test run.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import rag.eval.disagree as disagree_module
from rag.config import Settings
from rag.eval.corpus import ArticleRow, load_articles
from rag.eval.disagree import (
    Disagreement,
    DisagreementCallError,
    find_outside_section_items,
    independent_section_pick,
    run_disagreement_pass,
    section_shortlist,
)
from rag.eval.schema import GoldenItem

REPO_ROOT = Path(__file__).resolve().parents[2]
FICHES_DIR = REPO_ROOT / "data" / "corpus" / "fiches"
ARTICLES_PATH = REPO_ROOT / "data" / "corpus" / "articles.jsonl"

# Same F1124 fixture as test_eval_corpus.py / test_annotate_app.py.
FICHE_ID = "F1124"
VALID_ARTICLE_CID = "LEGIARTI000006792738"  # L127-1, under F1124's dc:source section
OFF_SECTION_ARTICLE_CID = "LEGIARTI000006785773"  # committed, outside every F1124 section


@pytest.fixture(scope="module")
def articles() -> Sequence[ArticleRow]:
    return load_articles(ARTICLES_PATH)


def _item(
    item_id: str, *, gold_fiches: tuple[str, ...], gold_articles: tuple[str, ...] = ()
) -> GoldenItem:
    return GoldenItem(
        id=item_id,
        question="une question ?",
        history=(),
        expected_state="reponse" if gold_articles else "reponse_sans_article",
        gold_fiches=gold_fiches,
        gold_spans=(),
        gold_articles=gold_articles,
        expected_points=("un point",),
        tags=(),
    )


def test_disagree_module_never_imports_the_retrieval_pipeline_or_the_lexical_shortlist() -> None:
    """This pass must share no code path with `propose.py`'s during-annotation shortlist
    (ADR-0020: reusing it would make the pass "redundant with itself, not independent"),
    checked statically against the actual imports rather than the prose describing them."""
    import_lines = [
        line
        for line in inspect.getsource(disagree_module).splitlines()
        if line.startswith(("import ", "from "))
    ]
    assert not any("rag.retrieval" in line for line in import_lines)
    assert not any("propose" in line for line in import_lines)


# --- section_shortlist / find_outside_section_items (real corpus, no LLM) ----


def test_section_shortlist_is_the_union_over_every_gold_fiche(
    articles: Sequence[ArticleRow],
) -> None:
    item = _item("gs-001", gold_fiches=(FICHE_ID,))
    cids = section_shortlist(item, FICHES_DIR, articles)
    assert VALID_ARTICLE_CID in cids
    assert OFF_SECTION_ARTICLE_CID not in cids


def test_find_outside_section_items_flags_a_gold_article_the_fiche_never_cited(
    articles: Sequence[ArticleRow],
) -> None:
    items = [
        _item("gs-001", gold_fiches=(FICHE_ID,), gold_articles=(VALID_ARTICLE_CID,)),
        _item("gs-002", gold_fiches=(FICHE_ID,), gold_articles=(OFF_SECTION_ARTICLE_CID,)),
    ]
    outside = find_outside_section_items(items, FICHES_DIR, articles)
    assert set(outside) == {"gs-002"}
    assert outside["gs-002"] == frozenset({OFF_SECTION_ARTICLE_CID})


def test_find_outside_section_items_ignores_items_with_empty_gold_articles(
    articles: Sequence[ArticleRow],
) -> None:
    items = [_item("gs-001", gold_fiches=(FICHE_ID,), gold_articles=())]
    assert find_outside_section_items(items, FICHES_DIR, articles) == {}


# --- Disagreement dataclass ---------------------------------------------------


def test_disagreement_explained_by_outside_section_excludes_the_expected_case() -> None:
    d = Disagreement(
        item_id="gs-001",
        gold_articles=frozenset({"A1", "A2"}),
        model_picks=frozenset({"A1"}),
        section_cids=frozenset({"A1"}),  # A2 was never in the pool shown to the model
    )
    assert d.only_in_gold == frozenset({"A2"})
    assert d.explained_by_outside_section == frozenset({"A2"})
    assert d.needs_review is False


def test_disagreement_needs_review_when_model_missed_an_in_pool_article() -> None:
    d = Disagreement(
        item_id="gs-001",
        gold_articles=frozenset({"A1"}),
        model_picks=frozenset(),
        section_cids=frozenset({"A1", "A2"}),  # A1 was shown but not picked by the model
    )
    assert d.explained_by_outside_section == frozenset()
    assert d.needs_review is True


def test_disagreement_needs_review_when_model_picks_something_the_annotator_didnt() -> None:
    d = Disagreement(
        item_id="gs-001",
        gold_articles=frozenset({"A1"}),
        model_picks=frozenset({"A1", "A2"}),
        section_cids=frozenset({"A1", "A2"}),
    )
    assert d.needs_review is True


# --- independent_section_pick / run_disagreement_pass (fake LLM) -------------


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
    response_content: str = ""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def invoke(self, messages: list[Any]) -> _FakeResponse:
        return _FakeResponse(_FakeChatOpenAI.response_content)


@pytest.fixture(autouse=True)
def _fake_chat_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(disagree_module, "ChatOpenAI", _FakeChatOpenAI)


def test_independent_section_pick_returns_empty_for_an_empty_shortlist() -> None:
    assert independent_section_pick("une question ?", set(), {}, _settings()) == set()


def test_independent_section_pick_drops_a_cid_outside_the_shortlist(
    articles: Sequence[ArticleRow],
) -> None:
    articles_by_cid = {row["cid"]: row for row in articles}
    _FakeChatOpenAI.response_content = (
        f"{VALID_ARTICLE_CID} - pertinent\nLEGIARTI000000000000 - hors liste\n"
    )
    picks = independent_section_pick(
        "une question ?", {VALID_ARTICLE_CID}, articles_by_cid, _settings()
    )
    assert picks == {VALID_ARTICLE_CID}


def test_run_disagreement_pass_skips_items_with_no_gold_fiches(
    articles: Sequence[ArticleRow],
) -> None:
    items = [_item("gs-001", gold_fiches=())]
    assert run_disagreement_pass(items, FICHES_DIR, articles, _settings()) == []


def test_run_disagreement_pass_reports_nothing_when_model_matches_gold(
    articles: Sequence[ArticleRow],
) -> None:
    _FakeChatOpenAI.response_content = f"{VALID_ARTICLE_CID} - pertinent\n"
    items = [_item("gs-001", gold_fiches=(FICHE_ID,), gold_articles=(VALID_ARTICLE_CID,))]
    assert run_disagreement_pass(items, FICHES_DIR, articles, _settings()) == []


def test_run_disagreement_pass_flags_a_mismatch(articles: Sequence[ArticleRow]) -> None:
    _FakeChatOpenAI.response_content = "aucun\n"
    items = [_item("gs-001", gold_fiches=(FICHE_ID,), gold_articles=(VALID_ARTICLE_CID,))]
    disagreements = run_disagreement_pass(items, FICHES_DIR, articles, _settings())
    assert len(disagreements) == 1
    assert disagreements[0].item_id == "gs-001"
    assert disagreements[0].only_in_gold == frozenset({VALID_ARTICLE_CID})


def test_run_disagreement_pass_wraps_a_dead_llm_call(
    articles: Sequence[ArticleRow], monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BrokenChatOpenAI(_FakeChatOpenAI):
        def invoke(self, messages: list[Any]) -> _FakeResponse:
            raise RuntimeError("network down")

    monkeypatch.setattr(disagree_module, "ChatOpenAI", _BrokenChatOpenAI)
    items = [_item("gs-001", gold_fiches=(FICHE_ID,), gold_articles=(VALID_ARTICLE_CID,))]
    with pytest.raises(DisagreementCallError):
        run_disagreement_pass(items, FICHES_DIR, articles, _settings())
