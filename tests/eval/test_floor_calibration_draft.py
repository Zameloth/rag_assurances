"""`rag.eval.floor_calibration_draft` — the floor-calibration drafter (#95, #84, ADR-0020).

The LLM round trips are the existing annotation-helper propose functions, replaced here by
canned fakes: no network call, no spend in a test run.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

import rag.eval.annotate_app as annotate_app_module
from rag.config import Settings
from rag.eval.annotate_app import _article_shortlist
from rag.eval.corpus import fiche_chunk_texts, load_articles
from rag.eval.disagree import Disagreement
from rag.eval.floor_calibration_draft import (
    Draft,
    DraftRun,
    draft_fiche,
    draft_floor_calibration,
    sample_unused_fiches,
    write_drafts,
)
from rag.eval.paths import GOLDEN_SET_PATH
from rag.eval.schema import GoldenItem, load_golden_set

# Modules whose presence in `sys.modules` after importing the drafter would mean an import
# path reaches the pipeline under test (the retrieval package, its Qdrant client).
_FORBIDDEN_PREFIXES = ("rag.retrieval", "qdrant_client")


def test_importing_the_drafter_never_loads_the_retrieval_pipeline() -> None:
    """ADR-0020's central constraint, checked on the *transitive* import graph — a fresh
    interpreter imports the drafter, then we look at what got loaded. Gold labels chosen by
    the reranker would calibrate the reranker against itself."""
    probe = (
        "import sys, rag.eval.floor_calibration_draft;"
        f"bad = sorted(m for m in sys.modules if m.startswith({_FORBIDDEN_PREFIXES!r}));"
        "print(bad)"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


# --- sample_unused_fiches ----------------------------------------------------


def _fiches_dir(tmp_path: Path, ids: list[str]) -> Path:
    fiches = tmp_path / "fiches"
    fiches.mkdir()
    for fiche_id in ids:
        (fiches / f"{fiche_id}.xml").write_text("<x/>")
    return fiches


def _golden(item_id: str, gold_fiches: tuple[str, ...]) -> GoldenItem:
    return GoldenItem(
        id=item_id,
        question="q ?",
        history=(),
        expected_state="reponse",
        gold_fiches=gold_fiches,
        gold_spans=(),
        gold_articles=("A1",),
        expected_points=("p",),
        tags=(),
    )


def test_sample_never_includes_a_fiche_used_by_a_golden_item(tmp_path: Path) -> None:
    fiches = _fiches_dir(tmp_path, [f"F{n}" for n in range(1, 21)])
    used = [_golden("gs-001", ("F1", "F2")), _golden("gs-002", ("F3",))]
    sample = sample_unused_fiches(fiches, used, n=17, seed=0)
    assert len(sample) == 17
    assert not {"F1", "F2", "F3"} & set(sample)


def test_sample_is_reproducible_for_a_seed_and_differs_across_seeds(tmp_path: Path) -> None:
    fiches = _fiches_dir(tmp_path, [f"F{n}" for n in range(1, 41)])
    first = sample_unused_fiches(fiches, [], n=10, seed=7)
    assert sample_unused_fiches(fiches, [], n=10, seed=7) == first
    assert sample_unused_fiches(fiches, [], n=10, seed=8) != first


def test_sample_caps_at_the_available_fiches(tmp_path: Path) -> None:
    fiches = _fiches_dir(tmp_path, ["F1", "F2", "F3"])
    assert sorted(sample_unused_fiches(fiches, [_golden("gs-001", ("F1",))], n=45, seed=0)) == ["F2", "F3"]


# --- draft_fiche (LLM faked at the ChatOpenAI boundary) ----------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
FICHES_DIR = REPO_ROOT / "data" / "corpus" / "fiches"
ARTICLES_PATH = REPO_ROOT / "data" / "corpus" / "articles.jsonl"
FICHE_ID = "F1124"
IN_SECTION_CID = "LEGIARTI000006792738"  # L127-1, under F1124's dc:source section


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _RoutingChatOpenAI:
    """Answers by which propose prompt it was handed — one canned reply per round trip."""

    replies: dict[str, str] = {}

    def __init__(self, **kwargs: Any) -> None:
        pass

    def invoke(self, messages: list[Any]) -> _FakeResponse:
        prompt = messages[0].content
        for marker, reply in _RoutingChatOpenAI.replies.items():
            if marker in prompt:
                return _FakeResponse(reply)
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    # The reject-justification round trip answers nothing unless a test says otherwise.
    _RoutingChatOpenAI.replies = {"Articles écartés": ""}
    monkeypatch.setattr(annotate_app_module, "ChatOpenAI", _RoutingChatOpenAI)
    return _RoutingChatOpenAI.replies


@pytest.fixture(scope="module")
def articles() -> list[dict[str, Any]]:
    return load_articles(ARTICLES_PATH)


def _settings() -> Settings:
    return Settings(
        openrouter_api_key="sk-or-test",
        generation_model="m",
        generation_provider="",
        condenser_model="",
        condenser_provider="",
        judge_model="",
        judge_provider="",
        langfuse_public_key="",
        langfuse_secret_key="",
        langfuse_base_url="https://cloud.langfuse.com",
        langfuse_tracing=False,
        qdrant_url="http://localhost:6333",
    )


def _draft(articles: list[dict[str, Any]]) -> Draft:
    return draft_fiche(FICHE_ID, "gs-901", fiches_dir=FICHES_DIR, articles=articles, settings=_settings())


def test_draft_with_a_pick_is_an_ai_assisted_reponse_candidate(
    llm: dict[str, str], articles: list[dict[str, Any]]
) -> None:
    real_span = fiche_chunk_texts(FICHE_ID, FICHES_DIR)[0][:60].strip()
    llm["Génère 2 à 3 questions"] = "Première question ?\nDeuxième question ?"
    llm["Articles candidats"] = f"{IN_SECTION_CID} - définit l'obligation"
    llm["Recopie 2 à 4 extraits"] = f"{real_span}\nun extrait inventé par le modèle"
    llm["affirmations"] = "Un point vérifiable"

    item = _draft(articles).item

    assert item.id == "gs-901"
    assert item.question == "Première question ?"
    assert item.expected_state == "reponse"
    assert item.gold_fiches == (FICHE_ID,)
    assert item.gold_articles == (IN_SECTION_CID,)
    assert item.gold_spans == (real_span,)  # the invented one fails the verbatim check
    assert item.expected_points == ("Un point vérifiable",)
    assert item.history == ()
    assert item.tags == ("ai_assisted",)


def test_draft_with_no_pick_is_a_reponse_sans_article_candidate(
    llm: dict[str, str], articles: list[dict[str, Any]]
) -> None:
    llm["Génère 2 à 3 questions"] = "Une question ?"
    llm["Articles candidats"] = "aucun"
    llm["Recopie 2 à 4 extraits"] = "un extrait inventé"
    llm["affirmations"] = "Un point"

    item = _draft(articles).item

    assert item.expected_state == "reponse_sans_article"
    assert item.gold_articles == ()
    assert item.gold_fiches == (FICHE_ID,)
    assert item.tags == ("ai_assisted",)


def test_draft_keeps_the_whole_shortlist_with_a_justification_per_candidate(
    llm: dict[str, str], articles: list[dict[str, Any]]
) -> None:
    llm["Génère 2 à 3 questions"] = "Une question ?"
    llm["Articles candidats"] = f"{IN_SECTION_CID} - définit l'obligation"
    llm["Recopie 2 à 4 extraits"] = ""
    llm["affirmations"] = "Un point"
    section_cids, lexical_cids = _article_shortlist(FICHE_ID, "Une question ?", FICHES_DIR, articles)
    rejected_cid = sorted((section_cids | lexical_cids) - {IN_SECTION_CID})[0]
    llm["Articles écartés"] = f"{rejected_cid} - parle d'autre chose"

    considered = {c.cid: c for c in _draft(articles).considered}

    assert set(considered) == section_cids | lexical_cids
    assert considered[IN_SECTION_CID].verdict == "pick"
    assert considered[IN_SECTION_CID].justification == "définit l'obligation"
    assert "dc:source" in considered[IN_SECTION_CID].provenance
    assert considered[rejected_cid].verdict == "reject"
    assert considered[rejected_cid].justification == "parle d'autre chose"
    assert all(c.verdict == "reject" for cid, c in considered.items() if cid != IN_SECTION_CID)


# --- draft_floor_calibration / write_drafts ----------------------------------


def _fake_pass(items: Sequence[GoldenItem], *_: Any) -> list[Disagreement]:
    """Stands in for `rag.eval.disagree.run_disagreement_pass`: disagrees on the first item."""
    first = items[0]
    return [
        Disagreement(
            item_id=first.id,
            gold_articles=frozenset(first.gold_articles),
            model_picks=frozenset({"LEGIARTI-model"}),
            section_cids=frozenset(),
        )
    ]


def _run(articles: list[dict[str, Any]], *, n: int = 3, seed: int = 1) -> DraftRun:
    return draft_floor_calibration(
        fiches_dir=FICHES_DIR,
        articles=articles,
        golden_items=load_golden_set(GOLDEN_SET_PATH),
        settings=_settings(),
        n=n,
        seed=seed,
        disagreement_pass=_fake_pass,
    )


def _canned(llm: dict[str, str]) -> None:
    llm["Génère 2 à 3 questions"] = "Une question ?"
    llm["Articles candidats"] = "aucun"
    llm["Recopie 2 à 4 extraits"] = ""
    llm["affirmations"] = "Un point"


def test_run_drafts_the_seeded_sample_outside_the_golden_set_with_sequential_ids(
    llm: dict[str, str], articles: list[dict[str, Any]]
) -> None:
    _canned(llm)
    golden = load_golden_set(GOLDEN_SET_PATH)
    golden_fiches = {f for item in golden for f in item.gold_fiches}

    run = _run(articles)

    items = [d.item for d in run.drafts]
    assert run.seed == 1
    assert [i.id for i in items] == ["gs-901", "gs-902", "gs-903"]
    assert not golden_fiches & {f for i in items for f in i.gold_fiches}
    assert [i.gold_fiches[0] for i in items] == sample_unused_fiches(FICHES_DIR, golden, n=3, seed=1)
    assert all(i.tags == ("ai_assisted",) for i in items)


def test_run_reports_disagreements_per_item(llm: dict[str, str], articles: list[dict[str, Any]]) -> None:
    _canned(llm)
    run = _run(articles)
    assert set(run.disagreements) == {"gs-901"}
    assert run.disagreements["gs-901"].model_picks == frozenset({"LEGIARTI-model"})


def test_run_records_a_fiche_whose_draft_fails_and_carries_on(
    llm: dict[str, str], articles: list[dict[str, Any]]
) -> None:
    _canned(llm)
    llm["Génère 2 à 3 questions"] = ""  # the model drafted no question at all

    run = _run(articles, n=2)

    assert run.drafts == []
    assert len(run.failures) == 2


def test_write_drafts_emits_a_golden_schema_file_and_a_review_sidecar(
    llm: dict[str, str], articles: list[dict[str, Any]], tmp_path: Path
) -> None:
    _canned(llm)
    run = _run(articles)
    golden_path = tmp_path / "floor-calibration.yaml"
    review_path = tmp_path / "floor-calibration.review.yaml"

    write_drafts(run, golden_path, review_path)

    assert "seed: 1" in golden_path.read_text(encoding="utf-8").splitlines()[0]
    assert [i.id for i in load_golden_set(golden_path)] == ["gs-901", "gs-902", "gs-903"]
    review = yaml.safe_load(review_path.read_text(encoding="utf-8"))
    assert review["seed"] == 1
    first = review["items"][0]
    assert first["id"] == "gs-901"
    assert first["disagreement"]["model_picks"] == ["LEGIARTI-model"]
    assert review["items"][1]["disagreement"] is None
    assert "considered" in first
