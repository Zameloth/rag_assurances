"""SPEC §3.2 / #20 acceptance criteria — the committed article corpus itself.

Runs against `data/corpus/articles.jsonl` as it sits in git, the same file ingest will
read. A corpus that fails these can never land on `main` (the fetch script runs the same
assertions before it writes, see scripts/fetch_articles.py) — this test is what keeps that
true after the fact, e.g. if the file were ever hand-edited.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from rag.ingest.assertions import run_article_assertions

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTICLES_PATH = REPO_ROOT / "data" / "corpus" / "articles.jsonl"
MANIFEST_PATH = REPO_ROOT / "data" / "corpus" / "corpus_manifest.json"
LICENSE_PATH = REPO_ROOT / "data" / "corpus" / "LICENSE.md"


@pytest.fixture(scope="module")
def articles() -> list[dict[str, Any]]:
    lines = ARTICLES_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return result


def test_articles_jsonl_is_committed_and_non_empty() -> None:
    assert ARTICLES_PATH.exists(), "data/corpus/articles.jsonl is not committed"
    assert ARTICLES_PATH.stat().st_size > 0


def test_ingest_assertions_1_to_4_pass(articles: list[dict[str, Any]]) -> None:
    run_article_assertions(articles)


def test_sorted_by_cid(articles: list[dict[str, Any]]) -> None:
    cids = [row["cid"] for row in articles]
    assert cids == sorted(cids)


def test_every_row_carries_texte_and_texte_html(articles: list[dict[str, Any]]) -> None:
    """SPEC §3.2's hard requirement — texteHtml, not only texte."""
    missing = [row["cid"] for row in articles if not row.get("texte") or not row.get("texteHtml")]
    assert not missing, f"{len(missing)} row(s) missing texte or texteHtml: {missing[:5]}"


def test_one_line_per_article_no_blank_lines() -> None:
    raw_lines = ARTICLES_PATH.read_text(encoding="utf-8").splitlines()
    assert all(line.strip() for line in raw_lines)


class TestCorpusManifest:
    """SPEC §16.4 — script-emitted, DILA/LO 2.0 attribution fixed regardless of the route."""

    def test_committed(self) -> None:
        assert MANIFEST_PATH.exists()

    def test_names_dila_and_licence_ouverte_regardless_of_the_route(self, manifest: dict[str, Any]) -> None:
        entry = manifest["articles"]
        assert entry["producer"] == "DILA"
        assert entry["licence"] == "Licence Ouverte 2.0"

    def test_carries_the_provenance_pin(self, manifest: dict[str, Any]) -> None:
        entry = manifest["articles"]
        for field in ("download_url", "filename", "file_date", "retrieved_at", "sha256", "document_count"):
            assert entry.get(field), f"{field} missing or empty"

    def test_document_count_matches_the_committed_file(
        self, manifest: dict[str, Any], articles: list[dict[str, Any]]
    ) -> None:
        assert manifest["articles"]["document_count"] == len(articles)

    def test_carries_mirror_of_when_the_download_is_not_dila_s_own(self, manifest: dict[str, Any]) -> None:
        entry = manifest["articles"]
        if "legifrance.gouv.fr" not in entry["download_url"] and "dila.gouv.fr" not in entry["download_url"]:
            assert entry.get("mirror_of"), "third-party download_url without mirror_of"


class TestDataCorpusLicense:
    """SPEC §16.2 — the human-readable licence file, split from `LICENSE` (code, MIT)."""

    def test_committed(self) -> None:
        assert LICENSE_PATH.exists()

    def test_names_licence_ouverte_and_the_manifest(self) -> None:
        text = LICENSE_PATH.read_text(encoding="utf-8")
        assert "Licence Ouverte" in text
        assert "corpus_manifest.json" in text
