#!/usr/bin/env python3
"""Fetch the in-scope service-public.fr fiches into `data/corpus/fiches/` (SPEC §1.2, §3, #21).

**A dev tool, run by hand — never imported from `rag` and never called by the pipeline**
(SPEC §3.3), same posture as `fetch_articles.py`. It reads `data/corpus/articles.jsonl` from
git rather than re-fetching the code — the scope rule is defined *against the committed
articles*, so a fiches refresh never silently drifts from whichever article corpus is
actually on disk.

    uv run --group fetch python scripts/fetch_fiches.py

**Route**: `vosdroits-latest.zip`, DILA's own daily-refreshed flux for service-public.fr
(`docs/research/corpus-sources.md` §2.1) — no PISTE key, no third-party mirror, so the
manifest entry carries no `mirror_of`. It is a `-latest` URL with no version to pin (SPEC
§3.2), which is exactly why the *selected* fiches are pinned in git instead: the ZIP itself
stays gitignored in `data/raw/`.

**The scope rule (SPEC §1.2), applied mechanically, not read off a list**: a fiche is in
scope iff at least one `LEGISCTA` id in its `<dc:source>` matches the `sectionParentId` of
some in-force article. `rag.ingest.fiches` does the parsing; this script does the
intersection and the write.

Before writing anything, this runs ingest assertion 5 (SPEC §7.4) over the selected set — a
fiche whose `<dc:source>` resolves to nothing in the article corpus would make expansion
silently yield nothing for it, and a corpus that fails its own assertions is not written.
Every run, not only the first — a refresh that would break the join fails loudly here.

**Refresh (SPEC §3.3, #22)**: also before writing, this diffs the freshly-selected fiches
against whatever is already committed under `data/corpus/fiches/` and reports three counts
— added, removed, and changed-text-under-the-same-`fiche_id`, keyed by the verbatim XML
bytes. This script is never scheduled — refresh is a manual, reviewed commit.

Writes `data/corpus/fiches/F*.xml` (verbatim bytes, original filenames — clearing any files
already there first, so a refresh's diff is the whole truth) and merges a `fiches` entry
into `corpus_manifest.json`, leaving the `articles` entry untouched.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

from rag.ingest.assertions import CorpusAssertionError, run_fiche_assertions
from rag.ingest.fiches import ParsedFiche, in_scope, parse_fiche
from rag.ingest.refresh_diff import diff_corpus, load_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_CORPUS = REPO_ROOT / "data" / "corpus"

VOSDROITS_URL = "https://lecomarquage.service-public.gouv.fr/vdd/3.5/part/zip/vosdroits-latest.zip"
LICENCE_OUVERTE_URL = "https://www.etalab.gouv.fr/wp-content/uploads/2017/04/ETALAB-Licence-Ouverte-v2.0.pdf"

_FICHE_NAME_RE = re.compile(r"^F\d+\.xml$")


def main() -> None:
    retrieved_at = datetime.now(UTC)
    filename = VOSDROITS_URL.rsplit("/", 1)[-1]

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_RAW / filename
    print(f"Downloading {VOSDROITS_URL}")
    sha256, last_modified = _download(VOSDROITS_URL, raw_path)
    file_date = _to_date(last_modified)
    print(f"  -> {raw_path} ({raw_path.stat().st_size:,} bytes, sha256={sha256}, file_date={file_date})")

    articles = load_jsonl(DATA_CORPUS / "articles.jsonl")
    article_section_ids = {row["sectionParentId"] for row in articles if row.get("sectionParentId")}
    print(f"{len(article_section_ids)} distinct article sectionParentId value(s)")

    with zipfile.ZipFile(raw_path) as archive:
        fiche_names = sorted(name for name in archive.namelist() if _FICHE_NAME_RE.match(name))
        print(f"{len(fiche_names)} F* fiche(s) in the flux")
        selected: list[tuple[str, bytes, ParsedFiche]] = []
        for name in fiche_names:
            data = archive.read(name)
            parsed = parse_fiche(data)
            if in_scope(parsed.section_ids, article_section_ids):
                selected.append((name, data, parsed))
    print(f"{len(selected)} fiche(s) in scope (SPEC §1.2)")

    try:
        run_fiche_assertions(
            ({"fiche_id": parsed.fiche_id, "section_ids": parsed.section_ids} for _, _, parsed in selected),
            articles,
        )
    except CorpusAssertionError as exc:
        raise SystemExit(f"Refusing to write a corpus that fails its own assertions:\n{exc}") from exc
    print(f"Ingest assertion 5 passed over {len(selected)} fiches")

    fiches_dir = DATA_CORPUS / "fiches"
    _report_refresh_diff(fiches_dir, selected)

    fiches_dir.mkdir(parents=True, exist_ok=True)
    for existing in fiches_dir.glob("F*.xml"):
        existing.unlink()
    for name, data, _ in selected:
        (fiches_dir / name).write_bytes(data)
    print(f"Wrote {len(selected)} file(s) to {fiches_dir}")

    manifest_path = DATA_CORPUS / "corpus_manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["fiches"] = {
        "producer": "DILA",
        "licence": "Licence Ouverte 2.0",
        "licence_url": LICENCE_OUVERTE_URL,
        "download_url": VOSDROITS_URL,
        "filename": filename,
        "file_date": file_date,
        "retrieved_at": retrieved_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": sha256,
        "document_count": len(selected),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


def _download(url: str, dest: Path) -> tuple[str, str | None]:
    """Stream `url` to `dest`, returning `(sha256, Last-Modified header value)`."""
    digest = hashlib.sha256()
    with httpx.stream("GET", url, timeout=120, follow_redirects=True) as response:
        response.raise_for_status()
        last_modified = response.headers.get("Last-Modified")
        with dest.open("wb") as f:
            for chunk in response.iter_bytes():
                digest.update(chunk)
                f.write(chunk)
    return digest.hexdigest(), last_modified


def _to_date(http_date: str | None) -> str | None:
    """RFC 2822 `Last-Modified` header -> `YYYY-MM-DD`, the LO 2.0 attribution's "file date"."""
    if http_date is None:
        return None
    return parsedate_to_datetime(http_date).date().isoformat()


def _report_refresh_diff(fiches_dir: Path, selected: list[tuple[str, bytes, ParsedFiche]]) -> None:
    """SPEC §3.3 — added / removed / changed-text-under-the-same-`fiche_id`, before writing anything.

    Compares against whatever is already committed under `fiches_dir`; empty on a first
    ingest, which correctly reports every fiche as added and nothing as changed.
    """
    old_by_fiche_id = (
        {path.stem: path.read_bytes() for path in sorted(fiches_dir.glob("F*.xml"))} if fiches_dir.exists() else {}
    )
    new_by_fiche_id = {parsed.fiche_id: data for _, data, parsed in selected}
    diff = diff_corpus(old_by_fiche_id, new_by_fiche_id)
    print(f"Refresh diff: {diff.summary()}")
    if diff.changed:
        print(f"{len(diff.changed)} fiche(s) changed text under the same fiche_id — re-review gold labels touching these:")
        for fiche_id in diff.changed:
            print(f"  {fiche_id}")


if __name__ == "__main__":
    main()
