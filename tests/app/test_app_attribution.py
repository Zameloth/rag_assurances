"""`rag.app.attribution` — the footer's DILA / Licence Ouverte 2.0 surface and the corpus
snapshot stamp, read from `corpus_manifest.json` (SPEC §13.5, §16.4, #50)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from rag.app.attribution import CORPUS_MANIFEST_PATH, load_attribution


def _source(**overrides: str) -> dict[str, object]:
    source: dict[str, object] = {
        "producer": "DILA",
        "licence": "Licence Ouverte 2.0",
        "licence_url": "https://www.etalab.gouv.fr/licence-ouverte",
        "download_url": "https://example.org/dump.zip",
        "filename": "dump.zip",
        "file_date": "2026-08-04",
        "retrieved_at": "2026-08-04T19:09:49Z",
        "sha256": "0" * 64,
        "document_count": 87,
    }
    source.update(overrides)
    return source


def _write(tmp_path: Path, manifest: dict[str, object]) -> Path:
    path = tmp_path / "corpus_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_each_source_carries_the_licence_three_and_its_name() -> None:
    attribution = load_attribution(CORPUS_MANIFEST_PATH)
    assert {s.name for s in attribution.sources} == {"Code des assurances", "Fiches service-public.fr"}
    for source in attribution.sources:
        assert source.producer == "DILA"
        assert source.licence == "Licence Ouverte 2.0"
        assert source.download_url.startswith("https://")
        assert source.filename
        assert isinstance(source.file_date, date)


def test_mirror_of_is_kept_only_where_the_download_was_not_dila_s_own(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "articles": _source(mirror_of="https://www.legifrance.gouv.fr/codes/texte_lc/X/"),
            "fiches": _source(),
        },
    )
    by_name = {s.name: s for s in load_attribution(path).sources}
    assert by_name["Code des assurances"].mirror_of == "https://www.legifrance.gouv.fr/codes/texte_lc/X/"
    assert by_name["Fiches service-public.fr"].mirror_of is None


def test_snapshot_date_is_the_oldest_source_file(tmp_path: Path) -> None:
    """One stamp for the whole corpus (SPEC §10.5): as old as its oldest content — a mirror
    fetched yesterday still carries the text of its own file date."""
    path = _write(
        tmp_path,
        {
            "articles": _source(file_date="2025-09-21", retrieved_at="2026-08-03T18:58:40Z"),
            "fiches": _source(file_date="2026-08-04", retrieved_at="2026-08-04T19:09:49Z"),
        },
    )
    assert load_attribution(path).snapshot_date == date(2025, 9, 21)
