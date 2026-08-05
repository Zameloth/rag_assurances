"""`rag.eval.corpus` against the committed corpus (#33) — the same read-only surface the
golden-set validator and the annotation helper both build on.
"""

from collections.abc import Sequence
from pathlib import Path

import pytest

from rag.eval.corpus import (
    ArticleRow,
    article_cids,
    fiche_chunk_texts,
    fiche_ids,
    fiche_sections,
    fiche_summaries,
    fiche_title,
    load_articles,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FICHES_DIR = REPO_ROOT / "data" / "corpus" / "fiches"
ARTICLES_PATH = REPO_ROOT / "data" / "corpus" / "articles.jsonl"

# F1124 "Assurance des associations" — `<dc:source>` names three LEGISCTA sections: one
# (LEGISCTA000006157261, Code des assurances Chapitre VII) carries 12 in-force articles,
# one (LEGISCTA000006174242) carries 3, and one (LEGISCTA000032021488, Code civil) carries
# none in this corpus — verified directly against `articles.jsonl`.
FICHE_ID = "F1124"
SECTION_WITH_ARTICLES = "LEGISCTA000006157261"
EMPTY_SECTION = "LEGISCTA000032021488"


@pytest.fixture(scope="module")
def articles() -> Sequence[ArticleRow]:
    return load_articles(ARTICLES_PATH)


def test_fiche_ids_includes_the_committed_corpus() -> None:
    ids = fiche_ids(FICHES_DIR)
    assert FICHE_ID in ids
    assert len(ids) == len(list(FICHES_DIR.glob("*.xml")))


def test_article_cids_matches_row_count(articles: Sequence[ArticleRow]) -> None:
    cids = article_cids(articles)
    assert len(cids) == len(articles)  # SPEC assertion 1 — one version per chronicle


def test_fiche_title() -> None:
    xml_bytes = (FICHES_DIR / f"{FICHE_ID}.xml").read_bytes()
    assert fiche_title(xml_bytes) == "Assurance des associations"


def test_fiche_summaries_cover_every_fiche() -> None:
    summaries = fiche_summaries(FICHES_DIR)
    assert len(summaries) == len(list(FICHES_DIR.glob("*.xml")))
    by_id = {s.fiche_id: s.title for s in summaries}
    assert by_id[FICHE_ID] == "Assurance des associations"


def test_fiche_chunk_texts_nonempty_and_reassembles_real_prose() -> None:
    texts = fiche_chunk_texts(FICHE_ID, FICHES_DIR)
    assert texts
    assert any("association" in text.lower() for text in texts)


def test_fiche_sections_is_a_reading_list_not_a_label_source(articles: Sequence[ArticleRow]) -> None:
    sections = fiche_sections(FICHE_ID, FICHES_DIR, articles)
    section_ids = {s.section_id for s in sections}
    # dc:source names 3 sections; only one resolves to in-force articles in this corpus.
    assert len(sections) == 3
    assert SECTION_WITH_ARTICLES in section_ids

    populated = next(s for s in sections if s.section_id == SECTION_WITH_ARTICLES)
    assert len(populated.articles) == 12
    assert all(row["sectionParentId"] == SECTION_WITH_ARTICLES for row in populated.articles)
    citation_ids = {row["citation_id"] for row in populated.articles}
    assert "L127-1" in citation_ids

    empty = next(s for s in sections if s.section_id == EMPTY_SECTION)
    assert empty.articles == ()
    assert empty.title == EMPTY_SECTION  # no article to source a title from
