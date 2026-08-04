"""SPEC §1.2 / §3.2 / #21 acceptance criteria — the committed fiche corpus itself.

Runs against `data/corpus/fiches/*.xml` and `data/corpus/articles.jsonl` as they sit in
git, the same files ingest will read. A corpus that fails these can never land on `main`
(the fetch script runs assertion 5 before it writes, see scripts/fetch_fiches.py) — this
test is what keeps that true after the fact.

Completeness against the *full* DILA flux (no in-scope fiche left out) is not checked
here: the flux itself is a gitignored, `-latest`-URL download with no version to pin (SPEC
§3.2), so it is not reproducible inside the test suite. What is checked is the direction
that matters for a corpus that must never contain a document the rule wouldn't pick: every
committed file independently satisfies the scope rule.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from rag.ingest.assertions import run_fiche_assertions
from rag.ingest.fiches import ParsedFiche, in_scope, parse_fiche

REPO_ROOT = Path(__file__).resolve().parents[1]
FICHES_DIR = REPO_ROOT / "data" / "corpus" / "fiches"
ARTICLES_PATH = REPO_ROOT / "data" / "corpus" / "articles.jsonl"
MANIFEST_PATH = REPO_ROOT / "data" / "corpus" / "corpus_manifest.json"

_FICHE_FILENAME_RE = re.compile(r"^F\d+\.xml$")


@pytest.fixture(scope="module")
def fiche_paths() -> list[Path]:
    return sorted(FICHES_DIR.glob("*.xml"))


@pytest.fixture(scope="module")
def parsed_fiches(fiche_paths: list[Path]) -> list[ParsedFiche]:
    return [parse_fiche(path.read_bytes()) for path in fiche_paths]


@pytest.fixture(scope="module")
def articles() -> list[dict[str, Any]]:
    lines = ARTICLES_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@pytest.fixture(scope="module")
def article_section_ids(articles: list[dict[str, Any]]) -> set[str]:
    return {row["sectionParentId"] for row in articles if row.get("sectionParentId")}


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return result


def test_exactly_87_fiches_committed(fiche_paths: list[Path]) -> None:
    assert len(fiche_paths) == 87


def test_filenames_are_original_dila_names(fiche_paths: list[Path]) -> None:
    offenders = [path.name for path in fiche_paths if not _FICHE_FILENAME_RE.match(path.name)]
    assert not offenders, f"non-DILA-shaped filename(s): {offenders}"


def test_filename_matches_the_fiche_s_own_publication_id(
    fiche_paths: list[Path], parsed_fiches: list[ParsedFiche]
) -> None:
    offenders = [
        path.name
        for path, parsed in zip(fiche_paths, parsed_fiches, strict=True)
        if path.stem != parsed.fiche_id
    ]
    assert not offenders, f"filename/Publication-ID mismatch: {offenders}"


def test_every_committed_fiche_is_in_scope_by_the_rule_itself(
    parsed_fiches: list[ParsedFiche], article_section_ids: set[str]
) -> None:
    """SPEC §1.2 — a rule, not a list: every committed file must independently satisfy it."""
    offenders = [parsed.fiche_id for parsed in parsed_fiches if not in_scope(parsed.section_ids, article_section_ids)]
    assert not offenders, f"committed fiche(s) failing the scope rule: {offenders}"


def test_no_duplicate_fiche_ids(parsed_fiches: list[ParsedFiche]) -> None:
    committed_ids = {parsed.fiche_id for parsed in parsed_fiches}
    assert len(committed_ids) == len(parsed_fiches), "duplicate fiche_id across committed files"


def test_ingest_assertion_5_passes(parsed_fiches: list[ParsedFiche], articles: list[dict[str, Any]]) -> None:
    run_fiche_assertions(
        ({"fiche_id": parsed.fiche_id, "section_ids": parsed.section_ids} for parsed in parsed_fiches),
        articles,
    )


class TestCorpusManifest:
    """SPEC §16.4 — same fields, same DILA/LO 2.0 attribution rule as the articles entry."""

    def test_fiches_entry_present(self, manifest: dict[str, Any]) -> None:
        assert "fiches" in manifest

    def test_names_dila_and_licence_ouverte(self, manifest: dict[str, Any]) -> None:
        entry = manifest["fiches"]
        assert entry["producer"] == "DILA"
        assert entry["licence"] == "Licence Ouverte 2.0"

    def test_carries_the_provenance_pin(self, manifest: dict[str, Any]) -> None:
        entry = manifest["fiches"]
        for field in ("download_url", "filename", "file_date", "retrieved_at", "sha256", "document_count"):
            assert entry.get(field), f"{field} missing or empty"

    def test_document_count_matches_the_committed_files(
        self, manifest: dict[str, Any], fiche_paths: list[Path]
    ) -> None:
        assert manifest["fiches"]["document_count"] == len(fiche_paths)

    def test_does_not_disturb_the_articles_entry(self, manifest: dict[str, Any]) -> None:
        assert manifest["articles"]["producer"] == "DILA"
